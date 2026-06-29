import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from transformers.cache_utils import Cache
import numpy as np
# from Experiments import copy_atten, copy_atten_gemma

ATTN_ATTR_CANDS = ("self_attn", "attn", "attention")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def my_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    return attn_weights, value_states



def _has_qkv_like(attn) -> bool:
    # 三投影 或 融合投影 均视为注意力单元
    three = all(hasattr(attn, x) for x in ("q_proj", "k_proj", "v_proj"))
    fused = any(hasattr(attn, x) for x in ("qkv_proj", "query_key_value"))
    return three or fused



def discover_attn_sites(model):
    """
    深度扫描全模型，返回所有注意力挂点。
    输出：List[ (parent_module, attr_name, attn_module) ]
    已按 (parent, attr) 去重；不会出现同一层 ['self_attn','self_attn'] 的重复。
    """
    sites = []
    seen_parent_attr = set()  # 去重 (id(parent), attr)
    for parent in model.modules():
        for attr in ATTN_ATTR_CANDS:
            if hasattr(parent, attr):
                attn = getattr(parent, attr)
                if _has_qkv_like(attn):
                    key = (id(parent), attr)
                    if key not in seen_parent_attr:
                        sites.append((parent, attr, attn))
                        seen_parent_attr.add(key)
    return sites


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class AttnInterceptorV2(nn.Module):
    def __init__(self, attn_module, layer_idx, mid_layer, alpha, beta, shared_cache, top_ratio=1.5, low_ratio=1):
        super().__init__()
        self.attn = attn_module            # 原始注意力模块（含 q/k/v 投影与 o_proj）
        self.layer_idx = int(layer_idx)
        self.mid_layer = int(mid_layer)
        self.start_layer = int(mid_layer)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.top_ratio = float(top_ratio)
        self.low_ratio = float(low_ratio)
        # 运行时状态（每个样本/批次前设置）
        self.active = True
        self.runtime_label_pos = None   # List[List[int]] or List[int]，按 batch 顺序
        self.runtime_spans = None       # List[Tuple[int,int]]，按 batch 顺序
        self.runtime_other_spans = None
        self.cache = shared_cache      # 多个层共享

    def set_runtime_state(self, label_pos_list, all_spans, corr_spans):
        # labels -> List[List[int]]
        if isinstance(label_pos_list, torch.Tensor):
            label_pos_list = label_pos_list.tolist()

        norm_labels = []
        for lp in label_pos_list:
            norm_labels.append(int(lp))

        if corr_spans is None:
            if isinstance(all_spans, torch.Tensor):
                all_spans = all_spans.detach().cpu().tolist()
            corr_spans=[] # 这里指的就是all_spans
            for spn in all_spans:
                b,e = spn
                corr_spans.append(list(range(b, e)))
            other_spns = None

        else:
            if isinstance(all_spans, torch.Tensor):
                all_spans = all_spans.detach().cpu().tolist()
            if isinstance(corr_spans, torch.Tensor):
                corr_spans= corr_spans.detach().cpu().tolist()
    
            other_spns = []
            for spn, cor_spn in zip(all_spans, corr_spans):
                runtime_indices = set()
                corr_indices = set()

                b,e = spn
                runtime_indices.update(range(b, e))
                corr_indices.update(cor_spn)

                final_indices = sorted(list(runtime_indices - corr_indices))
                other_spns.append(final_indices)
            if len(norm_labels) != len(corr_spans) or len(norm_labels) != len(other_spns) or len(other_spns)!= len(corr_spans):
                raise ValueError(f"len(label_pos_list)={len(norm_labels)} != len(spans)={len(other_spns)}") 
                       
        self.runtime_label_pos = norm_labels            # List[List[int]]
        self.runtime_other_spans = other_spns        #other img idx
        self.runtime_spans = corr_spans # correct area img idx or all_img_idx
        self.cache.clear()

    def enable(self, flag: bool = True):
        self.active = bool(flag)

    def forward(self, hidden_states, attention_mask=None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
            ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        if (not self.active) or (self.runtime_label_pos is None) or (self.runtime_spans is None):
            return self.attn(hidden_states, attention_mask = attention_mask, 
                                position_ids = position_ids, 
                                past_key_values = past_key_values,
                                cache_position = cache_position,
                                position_embeddings = position_embeddings,
                                **kwargs)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.attn.q_proj(hidden_states)
        key_states = self.attn.k_proj(hidden_states)
        value_states = self.attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1,self.attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.attn.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.attn.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states,self.attn.layer_idx, cache_kwargs)


        attn_weights, value_states = my_eager_attention_forward(self.attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.attn.training else self.attn.attention_dropout,
            scaling=self.attn.scaling,
            sliding_window=self.attn.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        label_pos_list = self.runtime_label_pos 
        spans = self.runtime_spans
        other_spans = self.runtime_other_spans

        if other_spans is None:
            if (self.layer_idx == self.mid_layer) and (self.cache.get('other_spans') is None)  and (self.cache.get('correct_spans') is None):
                correct_spans_list  = []
                other_spans_list = []
                for cur_spans, q_idx in zip(spans, label_pos_list):
                    P = attn_weights[0, :, q_idx, :] 
                    idxs = torch.as_tensor(cur_spans, device=attn_weights.device, dtype=torch.long)
                    idxs = torch.unique(idxs)
                    sub = P[:, idxs]                     # [H, M]

                    # 在 head 上取最大值 -> 得到每个 token 的代表值
                    token_vals = sub.mean(dim=0)# [M]
                    # 计算阈值
                    thr = token_vals.mean() * self.top_ratio
                    thr_low = token_vals.mean() * self.low_ratio
                    
                    # 其余的是 bottom
                    # 大于阈值的作为 top
                    top_local = torch.nonzero(token_vals > thr, as_tuple=False).squeeze(-1)
                    bot_local = torch.nonzero(token_vals <= thr_low, as_tuple=False).squeeze(-1)
                    # M = token_vals.numel()
                    # k_top = max(1, int(M * 0.02))

                    # 按从大到小排序
                    # _, order = torch.sort(token_vals, descending=True)

                    # top_local = order[:k_top]
                    # bot_local = order[k_top:]

                    top_token_idx = idxs[top_local]
                    bot_token_idx = idxs[bot_local]
                    correct_spans_list.append(top_token_idx)
                    other_spans_list.append(bot_token_idx)
                self.cache['other_spans'] = other_spans_list 
                self.cache['correct_spans'] = correct_spans_list
                
            if (self.layer_idx >= self.start_layer) and (self.cache.get('other_spans') is not None)  and (self.cache.get('correct_spans') is not None):
                est_spans = self.cache['correct_spans']
                est_other_spans = self.cache['other_spans']
                for cur_spans, other_spans, q_idx in zip(est_spans, est_other_spans, label_pos_list):
                    idxs = torch.as_tensor(cur_spans, device=attn_weights.device, dtype=torch.long)
                    idxs = torch.unique(idxs)

                    other_idxs = torch.as_tensor(other_spans, device=attn_weights.device, dtype=torch.long)
                    other_idxs = torch.unique(other_idxs)

                    if isinstance(q_idx, int):
                        q_idx_t = torch.tensor([q_idx], device=attn_weights.device, dtype=torch.long)
                        idxs = idxs[~torch.isin(idxs, q_idx_t)]

                    P = attn_weights[0, :, -1, :]          # [H, K]
                    H, K = P.shape

                    # 计算head
                    if idxs.numel() > 0:
                        # 每个 head 在目标区域 idxs 上的注意力总和或均值
                        head_scores = P[:, idxs].sum(dim=-1)      # [H] 也可以用 .mean(dim=-1)

                        top_h = 2
                        top_head_idx = torch.topk(head_scores, top_h, largest=True).indices   # [top_h]

                        h_idx = top_head_idx.unsqueeze(-1)        # [top_h, 1]
                        t_idx = idxs.unsqueeze(0)                 # [1, |idxs|]
                        P[h_idx, t_idx] = P[h_idx, t_idx] * self.alpha

                    if other_idxs.numel() > 0:
                        t_other_idx = other_idxs.unsqueeze(0)  
                        P[h_idx, t_other_idx] = P[h_idx, t_other_idx] * self.beta

                    if isinstance(q_idx, int) and 0 <= q_idx < K:
                        threshold = 0.005
                        head_scores = P[:, q_idx]  # [H] 也可以用 .mean(dim=-1)
                        top_h = 2
                        top_head_idx = torch.topk(head_scores, top_h, largest=True).indices   # [top_h]
                        h_idx = top_head_idx        # [top_h, 1]
                        P[h_idx, q_idx] =  P[h_idx, q_idx] * 1.2 #torch.where(P[:, q_idx] < threshold, P[:, q_idx] , P[:, q_idx])
     
     
                    P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    attn_weights[0, :, -1, :] = P 

        else:
            if self.layer_idx > self.mid_layer:
                for cur_spans, other_spans, q_idx in zip(spans, other_spans, label_pos_list):
                    idxs = torch.as_tensor(cur_spans, device=attn_weights.device, dtype=torch.long)
                    idxs = torch.unique(idxs)

                    other_idxs = torch.as_tensor(other_spans, device=attn_weights.device, dtype=torch.long)
                    other_idxs = torch.unique(other_idxs)

                    if isinstance(q_idx, int):
                        q_idx_t = torch.tensor([q_idx], device=attn_weights.device, dtype=torch.long)
                        idxs = idxs[~torch.isin(idxs, q_idx_t)]

                    # 2) 取出该 query（这里你用的是 -1 位置作为 predicted query）
                    P = attn_weights[0, :, -1, :]          # [H, K]
                    if idxs.numel() > 0:
                        P[:, idxs] = P[:, idxs] * self.alpha

                    if other_idxs.numel() > 0:
                        P[:, other_idxs] = P[:, other_idxs] * self.beta

                    K = P.size(-1)
                    if isinstance(q_idx, int) and 0 <= q_idx < K:
                        threshold = 0.005
                        P[:, q_idx] = torch.where(P[:, q_idx] < threshold, P[:, q_idx] * 1.2  ,  P[:, q_idx])
                    #     # print(P[:, q_idx])
                    P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    attn_weights[0, :, -1, :] = P

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous() # 到这里，等同于 attention_interface函数的输出，

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.attn.o_proj(attn_output)

        return attn_output, attn_weights


class AttnInterceptor(nn.Module):
    def __init__(self, attn_module, layer_idx, mid_layer, alpha,shared_cache):
        super().__init__()
        self.attn = attn_module            # 原始注意力模块（含 q/k/v 投影与 o_proj）
        self.layer_idx = int(layer_idx)
        self.mid_layer = int(mid_layer)
        self.alpha = float(alpha)

        # 运行时状态（每个样本/批次前设置）
        self.active = True
        self.runtime_label_pos = None   # List[List[int]] or List[int]，按 batch 顺序
        self.runtime_spans = None       # List[Tuple[int,int]]，按 batch 顺序
        self.cache = shared_cache      # 多个层共享

    def set_runtime_state(self, label_pos_list, spans, corr_spans=None):
        # labels -> List[List[int]]
        if isinstance(label_pos_list, torch.Tensor):
            label_pos_list = label_pos_list.tolist()

        norm_labels = []
        for lp in label_pos_list:
            norm_labels.append(int(lp))

        if isinstance(spans, torch.Tensor):
            spans = spans.detach().cpu().tolist()

        norm_spans = []
        for s in spans:
            b0, e0 = s
            sample_spans = (int(b0), int(e0))
            norm_spans.append(sample_spans)

        if len(norm_labels) != len(norm_spans):
            raise ValueError(f"len(label_pos_list)={len(norm_labels)} != len(spans)={len(norm_spans)}")

        self.runtime_label_pos = norm_labels            # List[List[int]]
        self.runtime_spans = norm_spans                 # List[List[Tuple[int,int]]]
        self.cache.clear()


    def enable(self, flag: bool = True):
        self.active = bool(flag)

    # 参考  class Qwen2_5_VLAttention(nn.Module):的实现 https://github.com/huggingface/transformers/blob/1f0b490a2c42eb129dccc69031ccb537058689c4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L619
    def forward(self, hidden_states, attention_mask=None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
            ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        if (not self.active) or (self.runtime_label_pos is None) or (self.runtime_spans is None):
            return self.attn(hidden_states, attention_mask = attention_mask, 
                                position_ids = position_ids, 
                                past_key_values = past_key_values,
                                cache_position = cache_position,
                                position_embeddings = position_embeddings,
                                **kwargs)

        bsz, q_len, _ = hidden_states.size()


        query_states = self.attn.q_proj(hidden_states)
        key_states = self.attn.k_proj(hidden_states)
        value_states = self.attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1,self.attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.attn.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.attn.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states,self.attn.layer_idx, cache_kwargs)


        attn_weights, value_states = my_eager_attention_forward(self.attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.attn.training else self.attn.attention_dropout,
            scaling=self.attn.scaling,
            sliding_window=self.attn.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        label_pos_list = self.runtime_label_pos 

        if self.layer_idx >= self.mid_layer:
            for q_idx in label_pos_list:
                row = attn_weights[0, :, q_idx, :]        # [H, T]
                H, T = row.shape

                # 随机扰动（随机+均匀）
                noise = torch.rand_like(row)              # [H, T]
                # 想控制随机幅度可以调 alpha，小一点更接近纯均匀
                alpha = 0.2
                row_new = (1 - alpha) * torch.ones_like(row) + alpha * noise

                row_new = torch.nan_to_num(row_new, nan=0.0, posinf=0.0, neginf=0.0)
                row_new = row_new / row_new.sum(dim=-1, keepdim=True).clamp(min=1e-12)

                attn_weights[0, :, q_idx, :] = row_new


        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous() # 到这里，等同于 attention_interface函数的输出，

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.attn.o_proj(attn_output)
        
        return attn_output, attn_weights


def wire_intervention(model, mid_layer=15, alpha=0.7, beta = 5, top_ratio=1.5, low_ratio=1, mode='flatten'):
    """
    对同一注意力对象（id 一样）的所有引用统一替换成同一个 AttnInterceptor 实例。
    这样无论 forward() 走哪个父模块/别名，都会进入拦截器。
    """
    sites = discover_attn_sites(model)  # [(parent, attr, attn)]
    # 分组：attn_id -> [(parent, attr, attn)]

    shared_cache = {}

    by_attn_id = {}
    for parent, attr, attn in sites:
        by_attn_id.setdefault(id(attn), []).append((parent, attr, attn))

    layer_idx = 0
    for _, group in by_attn_id.items():
        _, _, attn0 = group[0]
        if mode=='fallten':
            wrapper = AttnInterceptor(attn0, layer_idx, mid_layer, alpha, shared_cache)
            for p, a, _ in group:
                setattr(p, a, wrapper)

            alias_list = [(type(p).__name__, a) for p, a, _ in group]
            print(f"[wire] layer={layer_idx:02d} wrap {attn0.__class__.__name__} aliases={alias_list}")
            layer_idx += 1


        elif mode=="improve":
            wrapper = AttnInterceptorV2(attn0, layer_idx, mid_layer, alpha, beta, shared_cache, top_ratio=top_ratio, low_ratio=low_ratio)
            for p, a, _ in group:
                setattr(p, a, wrapper)

            alias_list = [(type(p).__name__, a) for p, a, _ in group]
            print(f"[wire] AttnInterceptorV2 layer={layer_idx:02d} wrap {attn0.__class__.__name__} aliases={alias_list}")
            layer_idx += 1
        elif mode=='already':
            wrapper = attn0
             # 更新内部参数
            wrapper.mid_layer = int(mid_layer)
            wrapper.alpha = float(alpha)
            wrapper.beta = float(beta)
            wrapper.mode = mode
            if hasattr(wrapper, "cache"):
                wrapper.cache = shared_cache
            for p, a, _ in group:
                setattr(p, a, wrapper)
            # print(f"[wire] layer={layer_idx:02d} already wrapped -> sync {[(type(p).__name__, a) for p,a,_ in group]}")
            layer_idx += 1
            continue
          
    return model

# 放在 copy_atten.py
def iter_interceptors(model):
    for m in model.modules():
        if isinstance(m, AttnInterceptor):
            yield m
        if isinstance(m, AttnInterceptorV2):
            yield m

def set_intervention_state(model, label_pos_list, all_spans, corr_spans=None):
    for intr in iter_interceptors(model):
        intr.set_runtime_state(label_pos_list, all_spans, corr_spans)

def enable_intervention(model, flag=True):
    for intr in iter_interceptors(model):
        intr.enable(flag)


def debug_attn_aliases(model, max_groups=8):
    sites = discover_attn_sites(model)
    by_attn_id = {}
    for parent, attr, attn in sites:
        by_attn_id.setdefault(id(attn), []).append((parent, attr, attn))

    print("=== debug_attn_aliases ===")
    for i, (aid, group) in enumerate(by_attn_id.items()):
        types = {attn.__class__.__name__ for _,_,attn in group}
        ptrs  = {hex(id(attn)) for _,_,attn in group}
        aliases = [(type(p).__name__, a) for p,a,_ in group]
        wrapped = ("AttnInterceptor" in types)
        print(f"group {i:02d} wrapped={wrapped} attn_ptrs={sorted(ptrs)} types={sorted(types)} aliases={aliases}")
        if i+1 >= max_groups:
            break


