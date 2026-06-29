# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

from tqdm import tqdm as tqdm
import torch
import copy
import numpy as np
from . import load_model_and_data as lmd
from PIL import Image
import os
from typing import  List, Dict, Any, Union, Optional
from dataclasses import dataclass
from .utils import _extract_image_grid_thw, locate_img_token_spans_expanded, _prepare_inputs_cpu, make_sentinels,  build_messages_with_sentinels, locate_text_only_spans, locate_all_spans, _reduce_heads, _reduce_tokens, _stack_rows_to_numpy, _select_positions, build_messages_with_label_signal, locate_demos_label_spans, locate_demos_question_spans, _build_messages_from_item
import torch.nn.functional as F


@dataclass
class Sample:
    messages: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Any]] = None
    text: Optional[str] = None

def _load_image(img_like) -> Image.Image:
    if isinstance(img_like, Image.Image):
        return img_like.convert("RGB")
    if isinstance(img_like, str):
        # 路径或 base64 之类，这里只实现路径
        if not os.path.exists(img_like):
            raise FileNotFoundError(f"Image path not found: {img_like}")
        return Image.open(img_like).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img_like)}")


@torch.no_grad()
def layerwise_entropy_from_inputs(
    model,
    inputs,
    position: int = -1,          # 评估哪个序列位置（-1=最后一个位置）
    temperature: float = 1.0,    # 可选：调和可视化的平滑，不影响排序
) -> torch.Tensor:
    """
    返回 shape [L+1] 的熵向量：embedding层 + 每一层 block 输出各一项。
    要求 batch_size=1（分析常用）。若要批量，自己把下方索引改成 [:, pos, :].
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    out = model(**inputs,
            return_dict=True,
            output_hidden_states=True,
            )
    hidden_states = out.hidden_states  # len = L+1, [B,T,D]
    # input_ids_cpu = inputs["input_ids"][0].detach().cpu()
    # print(input_ids_cpu.shape) # torch.Size([77])
    T = inputs["input_ids"].shape[1]
    pos = position if position >= 0 else (T + position)


    lm_head = model.lm_head
    final_norm = model.model.language_model.norm

    H_list: List[torch.Tensor] = []
    for h in hidden_states:
        h_pos = h[0, pos, :]  # [D]
        if final_norm is not None:
            h_pos = final_norm(h_pos)
        if hasattr(lm_head, "forward"):
            logits = lm_head(h_pos)  # [V]
        else:
            logits = F.linear(h_pos, lm_head.weight, getattr(lm_head, "bias", None))
        # 数值稳定 + 温度
        logits = (logits - logits.max()) / max(1e-6, temperature)
        p = F.softmax(logits, dim=-1)
        H = -(p * (p.clamp_min(1e-12)).log()).sum()
        H_list.append(H.detach().cpu())
    return torch.stack(H_list)  # [L+1]


@torch.no_grad()
def obtain_info_gain_by_layer(
    model,
    processor,
    few_shot_item,
    zero_shot_item,
    position: int = -1,
    temperature: float = 1.0,
) -> Dict[str, Any]:

    few_shot_messages = _build_messages_from_item(few_shot_item)
    zero_shot_messages = _build_messages_from_item(zero_shot_item)

    model.config._attn_implementation = "eager"
    import os; os.environ["PYTORCH_SDP_BACKEND"] = "math"

    model.config.output_attentions = True
    few_shot_inputs = processor.apply_chat_template(
        few_shot_messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    zero_shot_inputs = processor.apply_chat_template(
        zero_shot_messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

    """
    返回 [L+1]：每层的 InfoGain = H_zero - H_few
    """
    H0 = layerwise_entropy_from_inputs(model, zero_shot_inputs, position, temperature)
    HA = layerwise_entropy_from_inputs(model, few_shot_inputs,  position, temperature)
    return H0 - HA  # [L+1]


def compute_logit_lens(
    model,
    hidden_states,
    tokenizer,
    token_pos: int,
    target_id: int = None,           # ← 这里用 int（词表id），不是 list
    layer_indices=None,              # ← 改名更清晰：要分析哪些层
    gt_label = None,
    topk: int = 5
):
    """
    返回：dict[layer] = {
        "layer": int,
        "topk": [{"token": str, "prob": float, "id": int, "logit": float}, ...],
        "pred_token": str,
        "pred_prob": float,
        "pred_id": int,
        "pred_logit": float,
        "target_logit": float or None,
        "target_prob": float or None,
        "target_rank": int or None,
    }
    """
    lm_head = model.lm_head
    my_layer_norm = model.model.language_model.norm # Qwen2_5_VLForConditionalGeneration -> 
    L = len(hidden_states)
    if layer_indices is None:
        layer_indices = list(range(L))

    results = {}
    for layer in layer_indices:
        h = hidden_states[layer][0, token_pos]
        if layer == L-1: # 最后一层
            logits = lm_head(h) #已经有norm了，不需要再加了
        else:
            logits = lm_head(my_layer_norm(h)) ### 必须归一化一下
        probs  = torch.softmax(logits, dim=-1)

        # top-k
        topk_vals, topk_ids = torch.topk(probs, k=topk)
        topk_ids  = topk_ids.tolist()
        topk_vals = topk_vals.tolist()
        topk_tokens = [tokenizer.decode([i], skip_special_tokens=True) for i in topk_ids]
        topk_list = []
        for i_id, p in zip(topk_ids, topk_vals):
            topk_list.append({
                "token": tokenizer.decode([i_id], skip_special_tokens=True),
                "prob": float(p),
                "id": int(i_id),
                "logit": float(logits[i_id].item()),
            })
        pred_id = topk_ids[0]
        pred_prob = float(topk_vals[0])
        pred_token = topk_tokens[0]
        pred_logit = float(logits[pred_id].item())

        # 目标 token（如果给了 target_id）
        target_logit = None
        target_prob  = None
        target_rank  = None
        if target_id is not None:
            target_logit = float(logits[target_id].item())
            target_prob  = float(probs[target_id].item())
            target_rank  = int((logits > logits[target_id]).sum().item() + 1)

        results[layer] = {
            "layer": layer,
            "topk": topk_list,
            "pred_token": pred_token,
            "pred_prob": pred_prob,
            "pred_id": int(pred_id),
            "pred_logit": pred_logit,
            "target_logit": target_logit,
            "target_prob": target_prob,
            "target_rank": target_rank,
        }
        if gt_label:
            results[layer]['ground truth'] = gt_label
    return results


@torch.inference_mode()
def obtain_logit_lens(
    model,
    processor,
    item,
    label_list
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)
    # 1) 关闭 flash/sdpa，强制 eager/math
    model.config._attn_implementation = "eager"
    import os; os.environ["PYTORCH_SDP_BACKEND"] = "math"

    model.config.output_attentions = True

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states

    input_ids_cpu = inputs["input_ids"][0].detach().cpu()
    demos_question_span = locate_demos_question_spans(processor.tokenizer, input_ids_cpu, S, k)

    demos_ans_span = locate_demos_label_spans(processor.tokenizer, input_ids_cpu, S, k)
    
    logit_lens_ans_list = []
    for ques_pos, ans_pos, label in zip(demos_question_span, demos_ans_span, label_list[:-1]):
        target_id = int(input_ids_cpu[ans_pos].item())
        out_one = compute_logit_lens(
            model, hidden_states, processor,
            token_pos=ques_pos,
            target_id=target_id,
            layer_indices=None,
            topk=5,
            gt_label=label.split(" ")[-1].rstrip()
        )
        logit_lens_ans_list.append(out_one)

    # 用 processor 或 tokenizer 编码
    query_label = label_list[-1].split(" ")[-1].rstrip()
    tokenized = processor.tokenizer(query_label, add_special_tokens=False, return_tensors="pt")
    q_starget_id = tokenized["input_ids"][0][0].item()

    query_out = compute_logit_lens(
        model, hidden_states, processor,
        token_pos=-1,
        target_id=q_starget_id,
        layer_indices=None,
        topk=5,
        gt_label=query_label,
    )
    logit_lens_ans_list.append(query_out)
                                 
    return logit_lens_ans_list

@torch.inference_mode()
def obtain_prediction(
    model,
    processor,
    item,
    return_confi = False
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)
    # print("messages", messages)
    # print(1/0)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    # 1) 关闭 flash/sdpa，强制 eager/math
    model.config._attn_implementation = "eager"
    import os; os.environ["PYTORCH_SDP_BACKEND"] = "math"

    model.config.output_attentions = False

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model(
            **inputs,
            return_dict=True,
        )
        logits = out.logits[0, -1]
        next_token_id = logits.argmax(dim=-1)      # shape: ()

        decoded_output = processor.decode([next_token_id.item()], skip_special_tokens=True)
        if return_confi:
            probs = F.softmax(logits, dim=-1)         # 转成概率分布
            confidence = probs[next_token_id].item()  # 取被选中 token 的概率
            return decoded_output, confidence
        else:
            return decoded_output

@torch.inference_mode()
def obtain_vqa_prediction(
    model,
    processor,
    item,
    max_generation_length=5,
    min_generation_length=1,
    num_beams=1, # “模型每一步保留几个候选句子”。越大越聪明，但越慢
    length_penalty=1.0,
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)
    # print("messages", messages)
    # print(1/0)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    # 1) 关闭 flash/sdpa，强制 eager/math
    model.config._attn_implementation = "eager"
    import os; os.environ["PYTORCH_SDP_BACKEND"] = "math"

    model.config.output_attentions = False

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,                  # greedy，或你改 True 并加温度
            max_new_tokens=max_generation_length,
            min_new_tokens=min_generation_length,
            num_beams=num_beams,              # 你要 beam search 就写这个
            length_penalty=length_penalty,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    decoded_output = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
    return decoded_output
        
@torch.inference_mode()
def obtain_prediction_fw(
    model,
    processor,
    item,
    return_confi = False
) -> Dict[str, Any]:
    messages = _build_messages_from_item(item)


    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

        # 1) 关闭 flash/sdpa，强制 eager/math
    model.config._attn_implementation = "eager"
    import os; os.environ["PYTORCH_SDP_BACKEND"] = "math"

    model.config.output_attentions = True

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model(
            **inputs,
            return_dict=True,
        )
        logits = out.logits[0, -1]
        next_token_id = logits.argmax(dim=-1)      # shape: ()

        decoded_output = processor.decode([next_token_id.item()], skip_special_tokens=True)
        if return_confi:
            probs = F.softmax(logits, dim=-1)         # 转成概率分布
            confidence = probs[next_token_id].item()  # 取被选中 token 的概率
            return decoded_output, confidence
        else:
            return decoded_output



@torch.inference_mode()
def obtain_span_label_tok_idx(
    processor,
    item
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

    input_ids_cpu = inputs["input_ids"][0].detach().cpu()

    demos_label_span = locate_demos_label_spans(processor.tokenizer, input_ids_cpu, S, k)
    image_grid_thw = _extract_image_grid_thw(inputs)

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")

    spans, patch_grids = locate_img_token_spans_expanded(tok, inputs["input_ids"], k=len(image_grid_thw), image_grid_thw=image_grid_thw)

    return spans[:-1], patch_grids, demos_label_span # 忽略query的image

@torch.inference_mode()
def obtain_span_label_tok_idx_and_attn(
    model,
    processor,
    item
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)

    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if getattr(model.config, "use_flash_attn", None) is not None:
        model.config.use_flash_attn = False

    try:
        import torch
        torch.backends.cuda.sdp_kernel(enable_flash=False,
                                    enable_math=True,
                                    enable_mem_efficient=False)
    except Exception:
        pass

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            use_cache=False, # 关闭 cache，很多模型开 cache 时不返全层注意力
            # return_dict=True,
        )
        raw_atts = out.attentions
        L = len(raw_atts)

        # 存 head-avg 后的 attention
        atts = []

        for lay_id in range(L):
            att = raw_atts[lay_id].squeeze()  # [H, T, T]
            att_small = att.mean(dim=0)       # [T, T]
            atts.append(att_small)

        # 删除原始大张量
        del raw_atts
        del out.attentions
        torch.cuda.synchronize() 
        torch.cuda.empty_cache()

    input_ids_cpu = inputs["input_ids"][0].detach().cpu()
    demos_label_span = locate_demos_label_spans(processor.tokenizer, input_ids_cpu, S, k)
    image_grid_thw = _extract_image_grid_thw(inputs)

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")

    spans, patch_grids = locate_img_token_spans_expanded(tok, inputs["input_ids"], k=len(image_grid_thw), image_grid_thw=image_grid_thw)

    return spans[:-1], patch_grids, demos_label_span, atts


@torch.inference_mode()
def obtain_indemo_attention(
    model,
    processor,
    item
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)
        # ——尽量禁用会吞掉注意力权重的内核——
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if getattr(model.config, "use_flash_attn", None) is not None:
        model.config.use_flash_attn = False

    try:
        import torch
        torch.backends.cuda.sdp_kernel(enable_flash=False,
                                    enable_math=True,
                                    enable_mem_efficient=False)
    except Exception:
        pass


    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            use_cache=False, # 关闭 cache，很多模型开 cache 时不返全层注意力
            return_dict=True,
        )
        next_token_id = out.logits[0, -1].argmax(dim=-1)      # shape: ()
        decoded_output = processor.decode([next_token_id.item()], skip_special_tokens=True)


    input_ids_cpu = inputs["input_ids"][0].detach().cpu()


    demos_ans_span = locate_demos_label_spans(processor.tokenizer, input_ids_cpu, S, k) #(processor.tokenizer, input_ids_cpu, S, k)

    image_grid_thw = _extract_image_grid_thw(inputs)

    atts = out.attentions

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")
    label_token_idx= atts[0].shape[-1] -1
    demos_ans_span.append(label_token_idx)
    if hasattr(model, "config"):
        t_config =model.config
    else:
        t_config = None
    spans, patch_grids = locate_img_token_spans_expanded(tok, inputs["input_ids"], k=len(image_grid_thw), image_grid_thw=image_grid_thw,config = t_config)

    return atts, spans, patch_grids, demos_ans_span, decoded_output


@torch.inference_mode()
def obtain_each_demo_attention(
    model,
    processor,
    item
) -> Dict[str, Any]:

    k = len(item['question']) -1
    S = make_sentinels(k)

    messages = build_messages_with_label_signal(item, S)
        # ——尽量禁用会吞掉注意力权重的内核——
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if getattr(model.config, "use_flash_attn", None) is not None:
        model.config.use_flash_attn = False

    try:
        import torch
        torch.backends.cuda.sdp_kernel(enable_flash=False,
                                    enable_math=True,
                                    enable_mem_efficient=False)
    except Exception:
        pass


    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            use_cache=False, # 关闭 cache，很多模型开 cache 时不返全层注意力
            return_dict=True,
        )
        next_token_id = out.logits[0, -1].argmax(dim=-1)      # shape: ()
        decoded_output = processor.decode([next_token_id.item()], skip_special_tokens=True)


    input_ids_cpu = inputs["input_ids"][0].detach().cpu()

    demos_ques_span = locate_demos_question_spans(processor.tokenizer, input_ids_cpu, S, k) #(processor.tokenizer, input_ids_cpu, S, k)

    image_grid_thw = _extract_image_grid_thw(inputs)

    atts = out.attentions

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")
    label_token_idx= atts[0].shape[-1] -1
    demos_ques_span.append(label_token_idx)

    spans, patch_grids = locate_img_token_spans_expanded(tok, inputs["input_ids"], k=len(image_grid_thw), image_grid_thw=image_grid_thw)

    return atts, spans, patch_grids, demos_ques_span, decoded_output


@torch.inference_mode()
def obtain_attention(
    model,
    processor,
    item
) -> Dict[str, Any]:
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if getattr(model.config, "use_flash_attn", None) is not None:
        model.config.use_flash_attn = False

    try:
        import torch
        torch.backends.cuda.sdp_kernel(enable_flash=False,
                                    enable_math=True,
                                    enable_mem_efficient=False)
    except Exception:
        pass

    inputs, _ = _prepare_inputs_cpu(processor, item) 

    image_grid_thw = _extract_image_grid_thw(inputs)

    # 2) 前向（模型自行选择设备）；取注意力
    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            use_cache=False, # 关闭 cache，很多模型开 cache 时不返全层注意力
            return_dict=True,
        )
        next_token_id = out.logits[0, -1].argmax(dim=-1)      # shape: ()
        decoded_output = processor.decode([next_token_id.item()], skip_special_tokens=True)

    atts = out.attentions  # list[L], 每层 (B, H, T, T)

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")
 

    # 3) 定位 spans（展开后的区间）和 label_pos（同一份 inputs 上）
    spans, patch_grids = locate_img_token_spans_expanded(tok, inputs["input_ids"], k=len(image_grid_thw), image_grid_thw=image_grid_thw)

    label_token_idx = atts[0].shape[-1] -1


    return atts, spans, patch_grids, label_token_idx, decoded_output

def decode_tokens_by_pos(tok, input_ids_cpu, txt_pos):
    """
    tok: processor.tokenizer
    input_ids_cpu: 1D tensor on CPU (from inputs["input_ids"][0].cpu())
    txt_pos: either (st, ed) or list/np.array of indices
    return: (tokens, text)  # tokens是可视化token, text是解码后的可读文本
    """
    import torch, numpy as np

    if isinstance(txt_pos, tuple) and len(txt_pos) == 2:
        st, ed = txt_pos
        ids_slice = input_ids_cpu[st:ed]
    else:
        # 允许 list / np.array / torch.tensor 的离散索引
        if not torch.is_tensor(txt_pos):
            txt_pos = torch.tensor(np.array(txt_pos), dtype=torch.long)
        ids_slice = input_ids_cpu.index_select(0, txt_pos)

    ids_list = ids_slice.tolist()

    # 1) token 级别（会看到 Ġ 这类空格前缀标志）
    raw_tokens = tok.convert_ids_to_tokens(ids_list)
    vis_tokens = [t.replace("Ġ", " ") for t in raw_tokens]  # 仅为可读性

    # 2) 文本级别（还原为自然语言）
    # 重要：不要清理空格，否则位置对齐会变得不稳定
    text = tok.decode(ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return vis_tokens, text
@torch.inference_mode()
def probe_label_to_demos_positions_across_diff_head(
    model,
    processor,
    item,
    text_only=False,
    token_reduce: str = "mean",
):
    k = len(item["image"]) - 1
    S = make_sentinels(k)

    messages = build_messages_with_sentinels(item, S)

    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if getattr(model.config, "use_flash_attn", None) is not None:
        model.config.use_flash_attn = False

    try:
        import torch
        torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    except Exception:
        pass

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
        )
    torch.cuda.synchronize()  # 在 empty_cache 前加这个试试
    torch.cuda.empty_cache()

    atts = out.attentions
    L = len(atts)

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")

    input_ids_cpu = inputs["input_ids"][0].detach().cpu()

    if text_only:
        spans = locate_text_only_spans(tok, input_ids_cpu, S, k)
    else:
        t_config = getattr(model, "config", None)
        spans = locate_all_spans(tok, input_ids_cpu, S, k, t_config)

    label_pos = spans["label_pos"]

    per_head_img = []
    per_head_txt = []

    # demos
    for i in range(k):
        if text_only:
            img_pos = []
            txt_pos = spans["demos_txt_spans"][i]
        else:
            img_pos = spans["demos_img_spans"][i]
            txt_pos = spans["demos_txt_spans"][i]

        img_layer_rows = []
        txt_layer_rows = []

        for l in range(L):
            A = atts[l][0].detach().to("cpu")

            if len(img_pos) > 0:
                v_img = A[:, label_pos, img_pos]
                v_img = _reduce_tokens(v_img, token_reduce)
                img_layer_rows.append(v_img)
            else:
                img_layer_rows.append(torch.zeros(0, dtype=A.dtype))

            if len(txt_pos) > 0:
                v_txt = A[:, label_pos, txt_pos]
                v_txt = _reduce_tokens(v_txt, token_reduce)
                txt_layer_rows.append(v_txt)
            else:
                txt_layer_rows.append(torch.zeros(0, dtype=A.dtype))

            del A

        if len(img_pos) > 0:
            per_head_img.append(_stack_rows_to_numpy(img_layer_rows))
        else:
            per_head_img.append(
                torch.zeros((L, 0), dtype=torch.float16).numpy()
            )

        if len(txt_pos) > 0:
            per_head_txt.append(_stack_rows_to_numpy(txt_layer_rows))
        else:
            per_head_txt.append(
                torch.zeros((L, 0), dtype=torch.float16).numpy()
            )


    # -----------------------------------------
    # 合并所有 head 并取算术平均
    # -----------------------------------------
    def compute_average(per_head_list):
        # per_head_list: list of (L, H) numpy
        valid = [x for x in per_head_list if x.size > 0]
        if len(valid) == 0:
            return None
        stacked = np.stack(valid, axis=0)   # (N_heads, L, H)
        return [stacked.mean(axis=0)]        # (L, H)

    avg_img = compute_average(per_head_img)
    avg_txt = compute_average(per_head_txt)

    return {
        "avg": {
            "image": avg_img,
            "text": avg_txt,
        },
        "spans": spans,
        "S": S,
        "messages": messages,
    }

@torch.inference_mode()
def probe_label_to_demos_positions(
    model,
    processor,
    item,
    text_only= False,
    head_reduce: str = "mean",           # 先在多头上聚合
    per_demo_target_reduce: str = "mean",# 作为“按 demo 聚合曲线”的方式
) -> Dict[str, Any]:
    """
    返回：
      {
        "per_token": {
          "image": [ np(L, |img_i|) for i in 1..k ],
          "text":  [ np(L, |txt_i|) for i in 1..k ],
        },
        "per_demo_agg": {              # 每个 demo 一条曲线
          "image": [ np(L,), ... k ],
          "text":  [ np(L,), ... k ],
        },
        "spans": {...}, "S": {...}, "messages": ...
      }
    """
    k = len(item['image']) -1
    S = make_sentinels(k)

    messages = build_messages_with_sentinels(item, S)
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if getattr(model.config, "use_flash_attn", None) is not None:
        model.config.use_flash_attn = False

    try:
        import torch
        torch.backends.cuda.sdp_kernel(enable_flash=False,
                                    enable_math=True,
                                    enable_mem_efficient=False)
    except Exception:
        pass
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # 2) 前向（模型自行选择设备）；取注意力
    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            use_cache=False, # 关闭 cache，很多模型开 cache 时不返全层注意力
        )
    torch.cuda.synchronize() 
    torch.cuda.empty_cache()

    atts = out.attentions # list[L], 每层 (B, H, T, T)
    L = len(atts)

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor 缺少 tokenizer")
    
    input_ids_cpu = inputs["input_ids"][0].detach().cpu()
    if text_only:
        spans = locate_text_only_spans(tok, input_ids_cpu, S, k)
    else:
        if hasattr(model, "config"):
            t_config =model.config
        else:
            t_config = None
        spans = locate_all_spans(tok, input_ids_cpu, S, k, t_config)
    label_pos = spans["label_pos"]

    per_token_img: List[np.ndarray] = []
    per_token_txt: List[np.ndarray] = []

    # # txt_pos = spans["q_txt_span"]
    # img_pos =spans["demos_img_spans"][0]
    # print(img_pos)
    # vis_tokens, text = decode_tokens_by_pos(tok, input_ids_cpu, img_pos)
    # print("TOKENS:", vis_tokens)  # 横轴可直接用这些token做xticklabels
    # print("TEXT:", text)

    # img_pos =spans["demos_txt_spans"][0]
    # print(img_pos)
    # vis_tokens, text = decode_tokens_by_pos(tok, input_ids_cpu, img_pos)
    # print("TOKENS:", vis_tokens)  # 横轴可直接用这些token做xticklabels
    # print("TEXT:", text)

    # print(1/0)

    # vis_tokens, text = decode_tokens_by_pos(tok, input_ids_cpu, txt_pos)
    # print("TOKENS:", vis_tokens)  # 横轴可直接用这些token做xticklabels
    # print("TEXT:", text)
#     ATOKENS: [':', ' Identify', ' the', ' single', ' minority', ' (', 'either', ' color', ' or', ' shape', ')', ' in', ' the', ' image', '.', ' Output', ' with', ' one', ' lowercase', ' word', '.', ' ', ' Answer', ':']
#       TEXT: : Identify the single minority (either color or shape) in the image. Output with one lowercase word.  Answer:

    # print(1/0)
    # TOKENS: [':', ' Identify', ' the', ' single', ' minority', ' (', 'either', ' color', ' or', ' shape', ')', ' in', ' the', ' image', '.', ' Output', ' with', ' one', ' lowercase', '  word', '.', ' Answer', ':', ' star', 'Ċ']
    # TEXT: : Identify the single minority (either color or shape) in the image. Output with one lowercase word. Answer: star

    for i in range(k):
        if text_only:
            img_pos = []
            txt_pos = spans["demos_txt_spans"][i]
        else:
            img_pos = spans["demos_img_spans"][i]
            txt_pos = spans["demos_txt_spans"][i]


        img_layer_rows: List[torch.Tensor] = []
        txt_layer_rows: List[torch.Tensor] = []

        for l in range(L):
            # 仅当前层搬到 CPU，避免占用显存/内存峰值
            A = atts[l][0].detach().to("cpu")  # (H, T, T), batch=0

            # image span
            if len(img_pos) > 0:
                v_img = A[:, label_pos, img_pos]           # (H, |img|)
                v_img = _reduce_heads(v_img, head_reduce)
                img_layer_rows.append(v_img)
                # img_curve.append(float(_reduce_targets(v_img, per_demo_target_reduce)))
            else:
                img_layer_rows.append(torch.zeros(0, dtype=A.dtype))
                # img_curve.append(0.0)

            # text span
            if len(txt_pos) > 0:
                v_txt = A[:, label_pos, txt_pos]           # (H, |txt|)
                v_txt = _reduce_heads(v_txt, head_reduce)  # (|txt|,)
                # print(v_txt.shape) torch.Size([25])
                txt_layer_rows.append(v_txt)
                # txt_curve.append(float(_reduce_targets(v_txt, per_demo_target_reduce)))
            else:
                txt_layer_rows.append(torch.zeros(0, dtype=A.dtype))
                # txt_curve.append(0.0)

            # 及时释放当前层 CPU 张量引用
            del A

        if len(img_pos) > 0:
            H_img = _stack_rows_to_numpy(img_layer_rows)
            per_token_img.append(H_img)
        else:
            per_token_img.append(torch.zeros((L, 0), dtype=torch.float16).numpy())

        if len(txt_pos) > 0:
            # per_token_txt.append(torch.stack(txt_layer_rows, dim=0).numpy())
            H_txt = _stack_rows_to_numpy(txt_layer_rows)
            per_token_txt.append(H_txt)
        else:
            per_token_txt.append(torch.zeros((L, 0), dtype=torch.float16).numpy())


    if text_only:
        img_pos = []
        txt_pos = spans["q_txt_span"]
    else:
        img_pos = spans["q_img_span"]
        txt_pos = spans["q_txt_span"]

    img_layer_rows: List[torch.Tensor] = []
    txt_layer_rows: List[torch.Tensor] = []

    ## query
    for l in range(L):
        # 仅当前层搬到 CPU，避免占用显存/内存峰值
        A = atts[l][0].detach().to("cpu")  # (H, T, T), batch=0

        # image span
        if len(img_pos) > 0:
            v_img = A[:, label_pos, img_pos]           # (H, |img|)
            v_img = _reduce_heads(v_img, head_reduce)
            img_layer_rows.append(v_img)
        else:
            img_layer_rows.append(torch.zeros(0, dtype=A.dtype))
            # img_curve.append(0.0)

        # text span
        if len(txt_pos) > 0:
            v_txt = A[:, label_pos, txt_pos]           # (H, |txt|)
            v_txt = _reduce_heads(v_txt, head_reduce)
            txt_layer_rows.append(v_txt)
        else:
            txt_layer_rows.append(torch.zeros(0, dtype=A.dtype))


        # 及时释放当前层 CPU 张量引用
        del A

    if len(img_pos) > 0:
        H_img = _stack_rows_to_numpy(img_layer_rows)
        per_token_img.append(H_img)
    else:
        per_token_img.append(torch.zeros((L, 0), dtype=torch.float16).numpy())

    if len(txt_pos) > 0:
        H_txt = _stack_rows_to_numpy(txt_layer_rows)
        per_token_txt.append(H_txt)
    else:
        per_token_txt.append(torch.zeros((L, 0), dtype=torch.float16).numpy())


    return {
        "per_token": {
            "image": per_token_img,
            "text":  per_token_txt,
        },
        "spans": spans,
        "S": S,
        "messages": messages,
    }

@torch.inference_mode()
def ICL_inference_to_hidden_states(model, processor, prompts, pooling="last"):
    """
    返回:
      pooling="last"          -> [layer][prompt][hidden]
      pooling="per_pair_last" -> [layer][prompt][pair][hidden]
    """

    hidden_per_prompt = []

    for item in tqdm(prompts):
        torch.cuda.synchronize() 
        torch.cuda.empty_cache()

        # 1) inputs 与 ids_for_pos/attn 全部来自同一 enc，避免“tokens=0, features=xxx”
        inputs, msgs   = _prepare_inputs_cpu(processor, item)  # 全在 CPU
        ids_for_pos    = inputs.get("input_ids", None)
        attn_for_pos   = inputs.get("attention_mask", None)

        if ids_for_pos is None:
            # 极少数处理器可能不直接返回 input_ids；仍用 **tokenize=True** 的路径补齐
            # （而不是对纯文本再 tokenizer，从而避免丢失图像占位符）
            enc_fallback = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=(msgs[-1]["role"] != "assistant"),
                return_tensors="pt",
            )
            ids_for_pos  = enc_fallback["input_ids"]
            attn_for_pos = enc_fallback.get("attention_mask", None)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        positions = _select_positions(
            ids_for_pos,
            attention_mask=attn_for_pos,
        )

        # 4) 抽取 hidden
        per_layer = []

        pos = int(positions) if not isinstance(positions, list) else int(positions[0])
        for h in outputs.hidden_states:
            vec = h[0, pos].detach().float().cpu().numpy()
            per_layer.append(vec)

        hidden_per_prompt.append(per_layer)

    # 5) 转置到 [layer][prompt][...]
    num_layers = len(hidden_per_prompt[0])
    ret = []
    for L in range(num_layers):
        ret.append([hidden_per_prompt[P][L] for P in range(len(hidden_per_prompt))])
    return ret


@torch.inference_mode()
def ICL_generate_hidden_states(model, processor, prompts,selected_last_pos_id):
    """
    返回:
      pooling="last"          -> [layer][prompt][hidden]
      pooling="per_pair_last" -> [layer][prompt][pair][hidden]
    """

    hidden_per_prompt = []

    for item in tqdm(prompts):
        torch.cuda.synchronize() 
        torch.cuda.empty_cache()

        # 1) inputs 与 ids_for_pos/attn 全部来自同一 enc，避免“tokens=0, features=xxx”
        inputs, msgs   = _prepare_inputs_cpu(processor, item)  # 全在 CPU
        ids_for_pos    = inputs.get("input_ids", None)
        attn_for_pos   = inputs.get("attention_mask", None)

        if ids_for_pos is None:
            # 极少数处理器可能不直接返回 input_ids；仍用 **tokenize=True** 的路径补齐
            # （而不是对纯文本再 tokenizer，从而避免丢失图像占位符）
            enc_fallback = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=(msgs[-1]["role"] != "assistant"),
                return_tensors="pt",
            )
            ids_for_pos  = enc_fallback["input_ids"]
            attn_for_pos = enc_fallback.get("attention_mask", None)

        outputs = model(**inputs, output_hidden_states=True, use_cache=False,return_dict=True)

        positions = _select_positions(
            ids_for_pos,
            attention_mask=attn_for_pos,
            selected_last_pos_id=selected_last_pos_id
        )

        # 4) 抽取 hidden
        per_layer = []

        pos = int(positions) if not isinstance(positions, list) else int(positions[0])
        for h in outputs.hidden_states:
            vec = h[0, pos].detach().float().cpu().numpy()
            per_layer.append(vec)

        hidden_per_prompt.append(per_layer)
    # 5) 转置到 [layer][prompt][...]
    num_layers = len(hidden_per_prompt[0])
    ret = []
    for L in range(num_layers):
        ret.append([hidden_per_prompt[P][L] for P in range(len(hidden_per_prompt))])
    return ret

@torch.no_grad()
def ICL_inference_to_hidden_states_transposed(model, processor, prompts, pooling="last"):
    """
    返回:
      pooling="last"          -> [prompt][layer][hidden]
      pooling="per_pair_last" -> [prompt][layer][pair][hidden]
    """

    results = []  # 每个 prompt 的结果: [layer][hidden] 或 [layer][pair][hidden]

    for item in tqdm(prompts):
        torch.cuda.synchronize() 
        torch.cuda.empty_cache()

        # 1) inputs 与 ids_for_pos/attn 全部来自同一 enc（CPU），避免 tokens/positions 不一致
        inputs, msgs = _prepare_inputs_cpu(processor, item)   # inputs 在 CPU；模型前向可自动搬设备
        ids_for_pos  = inputs.get("input_ids", None)
        attn_for_pos = inputs.get("attention_mask", None)


        # 1.1 兜底：极少数处理器不直接给 input_ids，则用 tokenize=True 的同一路径补齐
        if ids_for_pos is None:
            enc_fallback = processor.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=(msgs[-1]["role"] != "assistant"),
                return_tensors="pt",
            )
            # enc_fallback 现在应是 BatchEncoding / dict
            ids_for_pos  = enc_fallback["input_ids"]
            attn_for_pos = enc_fallback.get("attention_mask", None)
            # 注意：不替换 inputs，保持与像素/占位符一致的那份 inputs 作为前向输入

        # 2) 前向
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # 3) 位置选择（仅索引，不做跨设备计算）
        positions = _select_positions(
            ids_for_pos,
            attention_mask=attn_for_pos,
        )

        # 4) 抽取当前 prompt 的每层 hidden
        per_prompt_layers = []

        # positions: int 或 单元素 list
        pos = int(positions[0]) if isinstance(positions, list) else int(positions)
        for h in outputs.hidden_states:
            vec = h[0, pos].detach().float().cpu().numpy()     # [D]
            per_prompt_layers.append(vec)

        results.append(per_prompt_layers)  # 追加一个 prompt 的结果

    # 不做转置，直接返回 [prompt][layer][hidden] 或 [prompt][layer][pair][hidden]
    return results


@torch.inference_mode()
def encoder_inference_to_feature(
    model,
    processor,
    queries: List[dict],             # 每个元素: {"text": str, "image": PIL.Image 或 路径}
    modality: str = "pair",          # 和原来保持一致；此处我们主要用 "pair"
    combine: str = "concat",         # 在 Qwen2.5-VL 分支下会当作 "mean" 处理
    l2_normalize: bool = False
):
    """
    对 Qwen2.5-VL:返回“最后一层 hidden state 的 masked mean”作为 embedding（float16 1D）。
    对 SigLIP 有 get_*_features 的双塔）：沿用原来的 text/image/pair 处理（若你保留了那部分）。
    """
    import os, torch, numpy as np
    from PIL import Image

    def _load_image(x):
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(x)}")

    image_token = getattr(processor, "image_token", "<image>")

    feats = []

    # —— 若需要：推测文本最大长度（对 Qwen 不关键；保留以兼容 SigLIP）——
    try:
        MAX_TXT = getattr(getattr(model.config, "text_config", object()), "max_position_embeddings", None)
        if MAX_TXT is None:
            MAX_TXT = getattr(getattr(processor, "tokenizer", object()), "model_max_length", 64)
    except Exception:
        MAX_TXT = 64

    for q in queries:
        if not isinstance(q, dict) or "image" not in q:
            raise TypeError('Each query must be a dict with keys {"text","image"} for modality="pair".')

        text = q.get("text", "")
        img  = _load_image(q["image"])

        # Qwen2.5-VL 需要文本里带上图像占位符
        if image_token not in text:
            text = f"{image_token}\n{text}".strip()
        #
        enc = processor(text=[text], images=[img], return_tensors="pt")
        # 注意：有些版本 processor 不返回 attention_mask，这里兜底
        attn = enc.get("attention_mask", None)
        if attn is None:
            # 找一个 [B, T] 的输入来推 T，常见是 input_ids
            T = None
            for k, v in enc.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    T = v.size(1)
                    break
            if T is None:
                raise RuntimeError("Cannot infer sequence length for attention mask.")
            attn = torch.ones((1, T), dtype=torch.long)

        # 送到设备并取最后一层 hidden states
        device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        for k, v in enc.items():
            if isinstance(v, torch.Tensor):
                enc[k] = v.to(device)
        attn = attn.to(device)

        outputs = model(**enc, return_dict=True, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]          # [1, T, H]
        mask = attn.to(last_hidden.dtype).unsqueeze(-1)  # [1, T, 1]

        # 统一按 masked mean 池化（concat 在 Qwen 路径下同样当 mean 处理，保证稳定）
        summed = (last_hidden * mask).sum(dim=1)         # [1, H]
        denom  = mask.sum(dim=1).clamp_min(1.0)          # [1, 1]
        pooled = (summed / denom)[0]                     # [H]

        if l2_normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)

        emb = pooled.to("cpu").float().numpy()

        feats.append(emb.astype(np.float16, copy=False))

    return feats

@torch.inference_mode()
def encoder_inference_to_feature_siglip(
    model,                          # transformers.models.siglip.SiglipModel 或兼容接口
    processor,                      # AutoProcessor (适配 SigLIP)
    queries: List[Union[str, Image.Image, Dict]],
    modality: str = "pair",         # "auto" | "text" | "image" | "pair"
    combine: str = "concat",        # "concat" | "mean" —— pair 模式下如何融合
    l2_normalize: bool = False      # 是否把输出做 L2 归一化
):
    """
    queries 中每个元素：
      - str: 作为纯文本（默认）。若想把字符串当作图片路径，设 _TREAT_STR_AS_IMAGE_PATH=True
      - PIL.Image 或 图像路径: 作为图像
      - dict: {"text": str, "image": PIL.Image or path} -> pair

    返回: List[np.ndarray]，每个元素是 1D float16 特征
    """
    feats: List[np.ndarray] = []
        # 读出 SigLIP 的文本最大长度（通常就是 64）
    try:
        MAX_TXT = getattr(getattr(model.config, "text_config", object()), "max_position_embeddings", None)
        if MAX_TXT is None:
            MAX_TXT = getattr(getattr(processor, "tokenizer", object()), "model_max_length", 64)
    except Exception:
        MAX_TXT = 64
        
    for q in queries:
        # ---------- 判定模态 ----------
        if modality == "auto":
            if isinstance(q, dict):
                mod = "pair"
            elif isinstance(q, Image.Image):
                mod = "image"
            elif isinstance(q, str):
                if os.path.exists(q):
                    mod = "image"
                else:
                    mod = "text"
            else:
                raise TypeError(f"Unsupported query type: {type(q)}")
        else:
            mod = modality

        # ---------- 文本 ----------
        if mod == "text":
            text = q if isinstance(q, str) else q["text"]
            enc = processor(text=[text], return_tensors="pt")       # 保持 CPU，让加速器自己搬
            txt = model.get_text_features(**enc)                    # [1, D]，可能在某个 GPU
            if l2_normalize:
                txt = torch.nn.functional.normalize(txt, dim=-1)
            emb = txt[0].to("cpu").float().numpy()                  # 统一到 CPU

        # ---------- 图像 ----------
        elif mod == "image":
            img = _load_image(q if not isinstance(q, dict) else q["image"])
            enc = processor(images=img, return_tensors="pt")        # 保持 CPU
            vis = model.get_image_features(**enc)                   # [1, D]，可能在另一个 GPU
            if l2_normalize:
                vis = torch.nn.functional.normalize(vis, dim=-1)
            emb = vis[0].to("cpu").float().numpy()

        # ---------- 文本 + 图像 ----------
        elif mod == "pair":
            if not isinstance(q, dict):
                raise TypeError('For "pair" modality, each query must be a dict with keys {"text","image"}')

            img = _load_image(q["image"])

            # 分别编码（保持在 CPU；模型/Accelerate 会把张量路由到对应设备）
            text_enc = processor(text=[q["text"]], return_tensors="pt", truncation=True, max_length=MAX_TXT,padding="max_length")
            img_enc  = processor(images=img,     return_tensors="pt")

            text_emb = model.get_text_features(**text_enc)[0]       # 可能在 cuda:a
            img_emb  = model.get_image_features(**img_enc)[0]       # 可能在 cuda:b

            if l2_normalize:
                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
                img_emb  = torch.nn.functional.normalize(img_emb, dim=-1)

            # 关键：先把两路搬到同一设备（这里统一到 CPU）再融合，避免 “not on the same device”
            text_cpu = text_emb.to("cpu")
            img_cpu  = img_emb.to("cpu")

            if combine == "concat":
                pair = torch.cat([text_cpu, img_cpu], dim=-1)       # [2D]
            elif combine == "mean":
                # 维度必须一致（SigLIP 文本/图像塔默认同维）
                pair = 0.5 * (text_cpu + img_cpu)
            else:
                raise ValueError('combine must be "concat" or "mean"')

            emb = pair.float().numpy()

        else:
            raise ValueError('modality must be "auto" | "text" | "image" | "pair"')

        feats.append(emb.astype(np.float16, copy=False))

    return feats

def get_ppl(model, tokenizer, queries):
    with torch.no_grad():
        ret = []
        for query in tqdm(queries):
            tknzd_data = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, labels = tknzd_data)
            ret.append(result.loss.detach().to(torch.float).cpu().numpy().item())
        return ret
    
def ICL_inference_to_multi_token_hidden_states(model, tokenizer, prompts, tokens): # [prompt] -> [layer][prompt][token][hidden_state]
    with torch.no_grad():
        ret = []
        hidden_states_in_layers = []
        for prompt in tqdm(prompts):
            torch.cuda.synchronize() 
            torch.cuda.empty_cache()
            hidden_states_in_layer = []
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            for layer in range(len(result.hidden_states)):
                hidden_states_in_layer.append(result.hidden_states[layer][-1][tokens].detach().to(torch.float).cpu().numpy())
            hidden_states_in_layers.append(hidden_states_in_layer)
        for layer in range(len(hidden_states_in_layers[0])):
            layer_hidden_states = []
            for prompt in hidden_states_in_layers:
                layer_hidden_states.append(prompt[layer])
            ret.append(layer_hidden_states)

    real_ret = [[] for _ in range(len(tokens))]
    for i in range(len(tokens)):
        for layer in range(len(ret)):
            temp_layer = []
            for sample in ret[layer]:
                temp_layer.append(sample[i])
            real_ret[i].append(temp_layer)

    return real_ret

def ICL_inference_to_natural_hidden_states_and_attention(model, tokenizer, prompts): # [prompt] -> [layer][prompt][hidden_state]
    with torch.no_grad():
        ret_hidden = []
        ret_attention = []
        for prompt in tqdm(prompts):
            torch.cuda.synchronize() 
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_hidden_layer = []
            res_attention_layer = []
            for layer in range(len(result.hidden_states)):
                res_hidden_layer.append(result.hidden_states[layer].detach().cpu())
            for layer in range(len(result.attentions)):
                res_attention_layer.append(result.attentions[layer].detach().cpu())
            ret_hidden.append(res_hidden_layer)
            ret_attention.append(res_attention_layer)
            del result
        return ret_hidden, ret_attention

def normlized_attention_score_for_single_sample(attention_score, hidden_state, query_key_values, w_s):
    ret = copy.deepcopy(attention_score)
    dimension = len(hidden_state[0][0])
    total_heads = len(attention_score[0][0])
    for layer in range(len(attention_score)):
        hidden_state_layer = hidden_state[layer][0]
        for head in range(len(attention_score[layer][0])):
            dimension_start = head * dimension // total_heads
            dimension_end = (head + 1) * dimension // total_heads
            for token in range(len(attention_score[layer][0][head])):
                v = query_key_values[layer](torch.tensor(hidden_state_layer[token]).to(torch.float).to(query_key_values[layer].device))[2//3*dimension][dimension_start:dimension_end].to(torch.float).cpu().numpy()
                v_before = np.array([0] * (dimension_start))
                v_after = np.array([0] * (dimension - dimension_end))
                v = np.concatenate([v_before, v, v_after])
                w = w_s[layer](torch.tensor(v).to(torch.float).to(w_s[layer].device))[0].to(torch.float).cpu().numpy()
                normw = np.linalg.norm(w)
                for qtoken in range(len(attention_score[layer][0][head])):
                    normlized_attention_score = attention_score[layer][0][head][qtoken][token] * normw
                    ret[layer][0][head][qtoken][token] = normlized_attention_score
    return ret

def copy_saliency_for_single_sample(hidden_states_numpy, model_modules):
    ret = []
    for layer in range(len(model_modules)):
        try:
            hidden_state = torch.tensor(hidden_states_numpy[layer]).to(model_modules[layer].device)
        except:
            hidden_state = torch.tensor(hidden_states_numpy[layer])
        hidden_state.requires_grad = True
        result = model_modules[layer](hidden_state)
        torch.sum(result[0][-1][-1]).backward()
        ret.append(hidden_state.grad.detach().cpu().numpy())
    return ret

def get_copy_magnitude_from_attention_for_single_sample(attention):
    ret = []
    for layer in range(len(attention)):
        ret.append(torch.mean(attention[layer][0], 0, keepdim = False))
    return ret

def get_copy_magnitude_from_attention_for_multi_sample(attention, Q, K):
    ret = [[] for _ in range(len(attention[0]))]
    for sample in attention:
        attention_magnitude = get_copy_magnitude_from_attention_for_single_sample(sample)
        for i in range(len(attention_magnitude)):
            ret[i].append(attention_magnitude[i][Q][K].detach().cpu().numpy().item())
    for i in range(len(ret)):
        ret[i] = np.mean(ret[i])
    return ret

def step3_get_fl_feature_and_lastftol_attention(model, tokenizer, prompts_with_label, experimentor, pythia = False): # ([sample][layer][tokens][hidden_state], [sample][layer][head][K])
    with torch.no_grad():
        ret_hidden = []
        ret_attention = []
        for prompt in tqdm(prompts_with_label):
            forerunner_loca, labels_loca = lmd.find_tokenized_label_word(tokenizer, experimentor, prompt, pythia)
            extract_loca = []
            for i in range(len(forerunner_loca)):
                extract_loca.append(forerunner_loca[i])
                extract_loca.append(labels_loca[i])
            torch.cuda.synchronize() 
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_hidden_layer = []
            res_attention_layer = []
            for layer in range(len(result.hidden_states)):
                res_hidden_layer.append(result.hidden_states[layer][0][extract_loca].detach().cpu())
            for layer in range(len(result.attentions)):
                res_attention_head = []
                for head in range(len(result.attentions[layer][0])):
                    res_attention_head.append(result.attentions[layer][0][head][-2][extract_loca].detach().cpu())
                res_attention_layer.append(res_attention_head)
            ret_hidden.append(res_hidden_layer)
            ret_attention.append(res_attention_layer)
            del result
        return ret_hidden, ret_attention

def step2_get_fl_feature_and_lastftol_attention(model, tokenizer, prompts_with_label): # ([sample][layer][tokens][hidden_state], [sample][layer][head][K])
    with torch.no_grad():
        ret_hidden = []
        ret_attention = []
        for prompt in tqdm(prompts_with_label):
            forerunner_loca, labels_loca = [-2], [-1]
            extract_loca = []
            for i in range(len(forerunner_loca)):
                extract_loca.append(forerunner_loca[i])
                extract_loca.append(labels_loca[i])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_hidden_layer = []
            res_attention_layer = []
            for layer in range(len(result.hidden_states)):
                res_hidden_layer.append(result.hidden_states[layer][0][extract_loca].detach().cpu())
            for layer in range(len(result.attentions)):
                res_attention_head = []
                for head in range(len(result.attentions[layer][0])):
                    res_attention_head.append(result.attentions[layer][0][head][-1][extract_loca].detach().cpu())
                res_attention_layer.append(res_attention_head)
            ret_hidden.append(res_hidden_layer)
            ret_attention.append(res_attention_layer)
            del result
        return ret_hidden, ret_attention
    
def step2_get_attention_of_different_location(model, tokenizer, prompts_with_label, experimentor, pythia = False): # ([sample][layer][tokens][hidden_state], [sample][layer][head][K])
    with torch.no_grad():
        normal_copy = []
        label_copy = []
        for prompt in tqdm(prompts_with_label):
            forerunner_loca, labels_loca = lmd.find_tokenized_label_word(tokenizer, experimentor, prompt, pythia)
            extract_loca = []
            for i in range(len(forerunner_loca)):
                extract_loca.append(labels_loca[i])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_attention_layer = []
            if len(normal_copy) == 0:
                for layer in range(len(result.attentions)):
                    normal_copy.append([])
                    label_copy.append([])
            for layer in range(len(result.attentions)):
                for head in range(len(result.attentions[layer][0])):
                    for index in range(labels_loca[-2] + 1, tknzd_data.shape[1]):
                        if index in extract_loca:
                            label_copy[layer].append(result.attentions[layer][0][head][index][index-1].detach().cpu().numpy().item() * tknzd_data.shape[1])
                        else:
                            normal_copy[layer].append(result.attentions[layer][0][head][index][index-1].detach().cpu().numpy().item() * tknzd_data.shape[1])
            del result
        return normal_copy, label_copy

def get_copy_magnitude_for_single_layer(ICL_attention, sample_index, layer):
    res = []
    for heads in ICL_attention[sample_index][layer]:
        res.append(heads[0].item())
    return res