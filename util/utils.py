
from typing import Tuple, List, Optional, Dict, Any
import unicodedata
import torch
import math


def _select_positions(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pad_token_id: Optional[int] = None,
    selected_last_pos_id: Optional[int] = 1,
) -> int:
    """
    返回序列中最后一个“非 padding”token 的位置。
    不跳过图像 token；只要不是 padding 就认为是有效 token。
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "仅支持 batch=1"
    ids = input_ids[0]

    if attention_mask is not None:
        # attention_mask 指示哪些是有效 token
        mask = attention_mask[0].to(torch.bool)
        valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(valid_positions) == 0:
            raise ValueError("attention_mask 全是 0，找不到有效位置")
        last_pos = int(valid_positions[-selected_last_pos_id].item())

    else:
        # 没有 mask 的话，就找非 pad token 的最后一个位置
        if pad_token_id is not None:
            non_pad = torch.nonzero(ids != pad_token_id, as_tuple=False).flatten()
            if len(non_pad) == 0:
                raise ValueError("全是 pad，找不到有效 token")
            last_pos = int(non_pad[-selected_last_pos_id].item())
        else:
            # 没有 pad_token_id，那就直接取最后第 selected_last_pos_id 个位置
            last_pos = ids.size(0) - selected_last_pos_id
    return last_pos


def _build_messages_from_item(item):
    """
    统一把各种输入形式变成 messages（仅规范结构，不做编码）。
    允许:
      - item = "纯文本"
      - item = {"messages": [...]}  # 已按 {"role","content":[{"type":"image"/"text",...}, ...]} 组织
      - item = {"images":[...], "text": "..."}  # 简化形式
    """
    if isinstance(item, str):
        return [{"role": "user", "content": [{"type": "text", "text": item}]}]

    if isinstance(item, dict):
        if "messages" in item and isinstance(item["messages"], list):
            for m in item["messages"]:
                if "content" in m and isinstance(m["content"], list):
                    for seg in m["content"]:
                        if seg.get("type") == "text" and isinstance(seg.get("text"), list):
                            seg["text"] = " ".join(map(str, seg["text"]))
            return item["messages"]

        imgs = item.get("image", None)
        txts  = item.get("text", None)
        if imgs is None or txts is None:
            raise ValueError("dict item 缺少 messages 或 (image, text)")
        content = []
        for t,im in zip(txts, imgs): # 先image然后再给text, +output
            if len(t)>0: # 如果是last image token, prompts中最后一个
                content.append({"type": "text", "text": t})
            if len(im)>0: # imgs的路径不是空字符串；； only_text, 这个len(im)==0
                content.append({"type": "image", "image": im})
        return [{"role": "user", "content": content}]

    raise TypeError(f"不支持的 item 类型: {type(item)}")


def _prepare_inputs_cpu(processor, item):
    """
    稳定、通用的图文编码流程：
      1) msgs -> templated_text (str)
      2) 提取 images（按 content 顺序）
      3) processor(text=templated_text, images=images, return_tensors='pt')
    全程保持在 CPU；后续由 accelerate/device_map 自动搬运。
    """
    msgs = _build_messages_from_item(item)
    add_gen = (msgs[-1]["role"] != "assistant")

    # 1) 生成“可读模板字符串”（包含图像占位符）
    templated_text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_gen, continue_final_message=False
    )


    images = []
    for m in msgs:
        if "content" not in m: 
            continue
        for seg in m["content"]:
            if seg.get("type") == "image":
                images.append(seg["image"])
    

    if len(images) == 0:
        inputs = processor(
            text=templated_text,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=templated_text,
            images=images,
            return_tensors="pt",
        )

    return inputs, msgs



def make_sentinels(k: int) -> Dict[str, str]:
    S = {
        "Q_IMG_B":  "[S_Q_IMG_B]",
        "Q_IMG_E":  "[S_Q_IMG_E]",
        "Q_TXT_B":  "[S_Q_TXT_B]",
        "Q_TXT_E":  "[S_Q_TXT_E]",
        "ANS":      "[S_ANS]",
    }
    for i in range(k):
        S[f"D{i}_IMG_B"] = f"[S_D{i}_IMG_B]"
        S[f"D{i}_IMG_E"] = f"[S_D{i}_IMG_E]"
        S[f"D{i}_TXT_B"] = f"[S_D{i}_TXT_B]"
        S[f"D{i}_TXT_E"] = f"[S_D{i}_TXT_E]"
        S[f"D{i}_ANS_B"] = f"[S_D{i}_ANS_B]"
        S[f"D{i}_ANS_E"] = f"[S_D{i}_ANS_E]"
    return S


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("\u00A0", " ")

def _find_all_indices(seq: List[int], pat: List[int]) -> List[int]:
    hits = []
    n, m = len(seq), len(pat)
    if m == 0 or n < m:
        return hits
    i = 0
    while i <= n - m:
        if seq[i:i+m] == pat:
            hits.append(i)
            i += m
        else:
            i += 1
    return hits

def _tok_ids(tokenizer, s: str, add_special_tokens=False) -> List[int]:
    return tokenizer(s, add_special_tokens=add_special_tokens).input_ids


def _all_pairs_vstart_vend(tokenizer, ids, config=None):
    """
    查找所有 <image_start> / <image_end> 区间的 index 对
    参数:
        tokenizer: transformers 的 tokenizer
        ids: 一维的 input_ids 列表或张量
        model: 可选，提供则尝试优先从 model.config 中提取 image_start/end token id
    返回:
        pairs: list of (start_idx, end_idx)
    """
    # 尝试从模型 config 获取 vision token id
    vs_id = ve_id = None
    if config is not None:
        vs_id = getattr(config, "image_start_token_id", None)
        ve_id = getattr(config, "image_end_token_id", None)

    # fallback: 尝试从字符串 token 中寻找
    if vs_id is None or ve_id is None:
        cand = [
            ("<|vision_start|>", "<|vision_end|>"),
            ("<vision_start>",   "<vision_end>"),
            ("<start_of_image>", "<end_of_image>"),
            ("<image_start>",    "<image_end>"),
            ("<image>",          "</image>"),
        ]
        for a, b in cand:
            try:
                x = tokenizer.convert_tokens_to_ids(a)
                y = tokenizer.convert_tokens_to_ids(b)
                if isinstance(x, int) and x != tokenizer.unk_token_id and isinstance(y, int) and y != tokenizer.unk_token_id:
                    vs_id, ve_id = x, y
                    break
            except Exception:
                continue

    if vs_id is None or ve_id is None:
        raise ValueError("找不到 vision_start/vision_end 的 token id。")

    # 查找所有的开始和结束位置
    starts = [i for i, t in enumerate(ids) if t == vs_id]
    ends   = [i for i, t in enumerate(ids) if t == ve_id]

    # 成对匹配
    pairs = []
    ei = 0
    for s in starts:
        while ei < len(ends) and ends[ei] <= s:
            ei += 1
        if ei < len(ends):
            pairs.append((s, ends[ei]))
            ei += 1
    return pairs

def find_nth_vision_block(tokenizer, input_ids, n: int, config=None) -> Optional[Tuple[int,int]]:
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    ids = input_ids.tolist()
    pairs = _all_pairs_vstart_vend(tokenizer, ids, config)
    return pairs[n]

def build_messages_with_label_signal(
    item: List[Dict[str, Any]],   # 每个元素: {"image": <path or PIL.Image>, "text": str}
    S,
) -> List[Dict[str, Any]]:

    num_demos = len(item["image"]) -1  #First, study the examples we provide. Then utilize what you have learned to answer the new question.
    content = []
    content.append({"type": "text",  "text": "Learn from the demos and give only the answer to the final question."})
    for i in range(num_demos):
        if len(item["image"][i]) !=0:
            content.append({"type": "image", "image": item["image"][i]})
            content.append({"type": "text",  "text":  item["question"][i] + S[f"D{i}_ANS_B"]+ " "+  item["label"][i]+ " "+ S[f"D{i}_ANS_E"]})
        else:
            content.append({"type": "text",  "text": item["question"][i] + S[f"D{i}_ANS_B"]+ " "+  item["label"][i]+ " "+ S[f"D{i}_ANS_E"]})
    # query 段
    if len(item["image"][-1]) != 0:
        content.append({"type": "image", "image": item["image"][-1]})

    if len(item["question"][-1]) != 0:
        content.append({"type": "text",  "text": item["question"][-1]+" Answer:"})

    return [{"role": "user", "content": content}]



def build_messages_with_sentinels(
    item: List[Dict[str, Any]],   # 每个元素: {"image": <path or PIL.Image>, "text": str}
    S,
) -> List[Dict[str, Any]]:
    """
    统一把各种输入形式变成 messages（仅规范结构，不做编码）。
      - item = {"image":[...], "text": "..."}  # 简化形式
      返回符合 processor.apply_chat_template 的 messages。
    demos 按顺序拼接，每个 demo: [D_i_IMG_B][image][D_i_IMG_E][D_i_TXT_B]text[D_i_TXT_E]
    query: [Q_IMG_B][image][Q_IMG_E][Q_TXT_B]text[Q_TXT_E][ANS]
    """
    task_instrcutuion = "Learn from the demos and give only the answer to the final question."
    # Question = "Question: Identify the single minority (either color or shape) in the sentence. Output with one lowercase word."
    # print("item",item)
    num_demos = len(item["image"]) -1 

    content = []
    for i in range(num_demos):
        if len(item["image"][i]) !=0:
            if i == 0:
                content.append({"type": "text",  "text": task_instrcutuion+ " " + S[f"D{i}_IMG_B"]}) # 最开始，要加一些icl的介绍

            content.append({"type": "image", "image": item["image"][i]})
            content.append({"type": "text",  "text": S[f"D{i}_IMG_E"] + " " + S[f"D{i}_TXT_B"] + " " +item["question"][i]+ item["label"][i] +" " + S[f"D{i}_TXT_E"]})
        else:
            if i == 0:
                content.append({"type": "text",  "text": task_instrcutuion}) # 最开始，要加一些icl的介绍

            content.append({"type": "text",  "text": S[f"D{i}_TXT_B"] + " " +item["question"][i]+ item["label"][i] + " " + S[f"D{i}_TXT_E"]})    

    # query 段
    if len(item["image"][-1]) == 0:
        content.append({"type": "text",  "text": S["Q_TXT_B"] + " " +item["question"][-1]+ " Answer: " + S["Q_TXT_E"]})

    elif len(item["question"][-1]) == 0: # text是空的. query只包括image的信息
        content.append({"type": "image", "image": item["image"][-1]})

    else:
        content.append({"type": "image", "image": item["image"][-1]})
        content.append({"type": "text",  "text": S["Q_TXT_B"] + " " + item["question"][-1]+ " Answer: " + S["Q_TXT_E"]})
    return [{"role": "user", "content": content}]


def locate_img_token_spans_expanded(tokenizer, input_ids, k, image_grid_thw, config=None):
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    ids = input_ids.tolist()

    pairs = _all_pairs_vstart_vend(tokenizer, ids,config)
    assert len(pairs) >= k, f"需要至少 {k} 个视觉块，只有 {len(pairs)}"

    spans, patch_grids_eff = [], []
    for i in range(k):
        s_pos, e_pos = pairs[i]
        t, H, W = image_grid_thw[i]   # 原始 thw（例如 t=1,H=W=36）
        actual_count = e_pos - s_pos - 1
        if actual_count <= 0:
            raise AssertionError(f"[{i}] 实际视觉 token 数={actual_count}，检查 processor。")

        # 每帧 token 数（合并后）
        per_frame = actual_count // max(t,1)
        # 推断合并倍率 m：H*W / per_frame ≈ m^2  ->  Qwen2/2.5 通常 m=2
        ratio = (H*W) / max(per_frame,1)
        m = int(round(math.sqrt(ratio)))
        H_eff, W_eff = H // m, W // m
        assert H_eff*W_eff*t == actual_count, \
            f"[{i}] H_eff*W_eff*t={H_eff*W_eff*t} != actual_count={actual_count} (m={m})"

        begin = s_pos + 1
        end   = begin + actual_count   # 不含 <vision_end>
        spans.append((begin, end))
        patch_grids_eff.append((H_eff, W_eff))
    return spans, patch_grids_eff


def locate_label_pos_by_marker(tokenizer, input_ids: torch.Tensor, S: dict) -> int:
    """
    优先用模板里的答案标记 S['ANS']；否则退化为找最后一个“像文本”的 token。
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    ids = input_ids.tolist()

    # 1) 用标记（推荐）
    if S and "ANS" in S:
        ans_id = tokenizer.convert_tokens_to_ids(S["ANS"])
        cand = [i for i,t in enumerate(ids) if t == ans_id]
        if cand:
            return cand[-1]

    # 2) 兜底：最后一个不是图像相关特殊符号的 token
    bad_tokens = set()
    for tok in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", ("<image_start_2>", "<image_end_2>"),  # ✅ GLM-4.1V-9B 专用标记
                "<vision_start>", "<vision_end>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int):
                bad_tokens.add(tid)
        except Exception:
            pass
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        bad_tokens.add(pad_id)

    for i in range(len(ids)-1, -1, -1):
        if ids[i] not in bad_tokens:
            return i
    raise ValueError("无法定位 label_pos；请提供明确的答案标记。")



def _find_subsequence(hay: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
    n, m = len(hay), len(needle)
    if m == 0 or n < m:
        return None
    for i in range(n - m + 1):
        if hay[i:i+m] == needle:
            return (i, i + m)
    return None

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("\u00A0", " ")

def _marker_indices(tokenizer, input_ids, marker: str) -> Tuple[int, int]:
    seq = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    marker = _norm(marker)

    variants = [marker, " " + marker, "\n" + marker, "\r\n" + marker]
    tried = set()
    encodings: List[List[int]] = []
    for v in variants:
        for add_spec in (False, True):
            ids = tokenizer(v, add_special_tokens=add_spec).input_ids
            key = (tuple(ids), add_spec)
            if ids and key not in tried:
                encodings.append(ids)
                tried.add(key)

    for ids in encodings:
        hit = _find_subsequence(seq, ids)
        if hit is not None:
            return hit

    # 2) 字符级回退：解码整段，再找变体
    decoded = tokenizer.decode(seq, skip_special_tokens=False)
    decoded_norm = _norm(decoded)

    # 优先找“独占一行/带空白”的版本
    for v in variants:
        v_norm = _norm(v)
        char_pos = decoded_norm.find(v_norm)
        if char_pos != -1:
            # 3) 用 offsets 把字符区间映射回 token 区间，再二次 token 子序列匹配确认
            enc = tokenizer(decoded_norm, add_special_tokens=False, return_offsets_mapping=True)
            offs = enc.offset_mapping
            ids_norm = enc.input_ids
            start_char = char_pos
            end_char = char_pos + len(v_norm)

            # 找到覆盖 [start_char, end_char) 的 token span
            start_tok = None
            end_tok = None
            for i, (a, b) in enumerate(offs):
                if start_tok is None and a <= start_char < b:
                    start_tok = i
                if end_tok is None and a < end_char <= b:
                    end_tok = i + 1
                    break
            if start_tok is not None and end_tok is not None and start_tok < end_tok:
                needle = ids_norm[start_tok:end_tok]
                hit = _find_subsequence(seq, needle)
                if hit is not None:
                    return hit

    snippet = decoded_norm[:500]
    raise ValueError(
        f"[MarkerNotFound] {marker}. 很可能在 chat 模板/分词时被吞/改写。\n"
        f"Decoded(前500):\n---\n{snippet}\n---\n"
        "修复建议：\n"
        "1) 改用方括号标记 [S_*] 且让标记独占一行；\n"
        "2) 或自己先渲染模板：rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)，\n"
        "   assert marker in rendered，再 tokenizer(rendered, add_special_tokens=False) 喂模型；\n"
        "3) 确保定位时用的 tokenizer 与生成 input_ids 的 tokenizer 是同一个实例。"
    )


def _span_between(l_exclusive: int, r_exclusive: int) -> List[int]:
    if r_exclusive - l_exclusive <= 1: return []
    return list(range(l_exclusive + 1, r_exclusive))

def _stack_rows_to_numpy(rows):
    """
    rows: List[Tensor], 每个元素形如 (W_i,) 或 (token_cnt,)
    返回: np.ndarray 形如 (L, W_i)，dtype=float32；rows 为空则返回 None
    """
    if not rows:
        return None
    H = torch.stack(rows, dim=0)  # (L, W_i)
    H = H.detach().to(device="cpu", dtype=torch.float32)  # 关键：float32
    return H.numpy()


def locate_all_spans(tokenizer, input_ids: torch.Tensor, S: Dict[str, str], k: int, config=None) -> Dict[str, Any]:
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    T = input_ids.shape[0]
    # 1) 最后一位
    label_pos = T - 1
    
    demos_img_spans, demos_txt_spans = [], []
    for i in range(k):
        img_span = find_nth_vision_block(tokenizer, input_ids, i, config=config)  # 第 i 个 demo 的视觉块
        if img_span is None:
            raise ValueError(f"Cannot locate vision block #{i}. "
                            f"Likely the chat template merged images; consider option A.")
        iB, iE = img_span

        tB = _marker_indices(tokenizer, input_ids, S[f"D{i}_TXT_B"])
        tE = _marker_indices(tokenizer, input_ids, S[f"D{i}_TXT_E"])
        # print("demos_img_spans",_span_between(iB, iE))
        # print("demos_txt_spans",_span_between(tB[-1], tE[0]))
        demos_img_spans.append(_span_between(iB, iE))
        demos_txt_spans.append(_span_between(tB[-1], tE[0]))

    img_span = find_nth_vision_block(tokenizer, input_ids, k, config=config)  # 第 i 个 demo 的视觉块
    q_iB, q_iE = img_span
    q_tB = _marker_indices(tokenizer, input_ids, S["Q_TXT_B"])
    q_tE = _marker_indices(tokenizer, input_ids, S["Q_TXT_E"])
    q_img_span = _span_between(q_iB, q_iE)
    q_txt_span = _span_between(q_tB[-1], q_tE[0])

    return {
        "label_pos": label_pos,
        "demos_img_spans": demos_img_spans,  # List[List[int]]，每个 demo 的 image span（逐 token）
        "demos_txt_spans": demos_txt_spans,  # List[List[int]]，每个 demo 的 text  span（逐 token）
        "q_img_span": q_img_span,
        "q_txt_span": q_txt_span,
    }


def locate_demos_label_spans(tokenizer, input_ids: torch.Tensor, S: Dict[str, str], k: int) -> Dict[str, Any]:
    demos_ans_spans = []
    for i in range(k):
        ans_B = _marker_indices(tokenizer, input_ids, S[f"D{i}_ANS_B"])
        ans_E = _marker_indices(tokenizer, input_ids, S[f"D{i}_ANS_E"])
        demos_ans_spans.append(_span_between(ans_B[-1], ans_E[0])[-2])
    return demos_ans_spans

def locate_demos_question_spans(tokenizer, input_ids: torch.Tensor, S: Dict[str, str], k: int) -> Dict[str, Any]:
    demos_question_spans = []
    for i in range(k):
        ans_B = _marker_indices(tokenizer, input_ids, S[f"D{i}_ANS_B"])
        ans_E = _marker_indices(tokenizer, input_ids, S[f"D{i}_ANS_E"])
        demos_question_spans.append(_span_between(ans_B[-1], ans_E[0])[-3]) # 这个是冒号，
    return demos_question_spans


def locate_img_token_spans(tokenizer, input_ids: torch.Tensor, S: Dict[str, str], k: int) -> Dict[str, Any]:
    # idx_ANS   = _marker_indices(tokenizer, input_ids, S["ANS"])
    # label_pos = idx_ANS[-1]
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    T = input_ids.shape[0]
    # 1) 最后一位
    label_token_idx = T - 1
    
    demos_img_spans = []
    for i in range(k):
        img_span = find_nth_vision_block(tokenizer, input_ids, i)  # 第 i 个 demo 的视觉块
        if img_span is None:
            raise ValueError(f"Cannot locate vision block #{i}. "
                            f"Likely the chat template merged images; consider option A.")
        iB, iE = img_span
        demos_img_spans.append((iB, iE))

    return demos_img_spans, label_token_idx

def _extract_image_grid_thw(inputs):
    """
    兼容 Qwen2 / Qwen2.5-VL:
      - 直接在顶层: inputs["image_grid_thw"]
      - 或嵌在 vision_info 里: inputs["vision_info"]["image_grid_thw"]
    """
    val = inputs.get("image_grid_thw", None)
    if val is None:
        vi = inputs.get("vision_info", None)
        if isinstance(vi, dict):
            val = vi.get("image_grid_thw", None)

    # Gemma-3 系列
    if val is None:
        # Gemma-3 通常没有 image_grid_thw 字段，可尝试直接读 token 数或用默认
        n_imgs = 1
        for k in ("images", "pixel_values"):
            if k in inputs:
                # try: #可以的
                n_imgs = len(inputs[k]) if hasattr(inputs[k], "__len__") else 1
                # except Exception:
                #     pass
                # break
        return [[1, 16, 16] for _ in range(n_imgs)]
    
    if val is None:
        # 这里不要瞎推断 patch 大小；直接报错提醒调用方必须用 processor(images=...) 合并视觉字段
        raise ValueError(
            "找不到 image_grid_thw。请确保在构造 inputs 时把图片也交给 processor，"
            "例如：processor(messages, images=..., return_tensors='pt')，"
            "或把 processor(images=...) 的返回与文本返回合并。"
        )
    return val



def locate_text_only_spans(tokenizer, input_ids: torch.Tensor, S: Dict[str, str], k: int) -> Dict[str, Any]:
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    T = input_ids.shape[0]
    # 1) 最后一位
    label_pos = T - 1

    demos_txt_spans = []
    for i in range(k):
        tB = _marker_indices(tokenizer, input_ids, S[f"D{i}_TXT_B"])
        tE = _marker_indices(tokenizer, input_ids, S[f"D{i}_TXT_E"])
        demos_txt_spans.append(_span_between(tB[-1], tE[0]))

    # query 部分可选：这里只研究 demos，就不强制计算；保留接口
    try:
        q_tB = _marker_indices(tokenizer, input_ids, S["Q_TXT_B"])
        q_tE = _marker_indices(tokenizer, input_ids, S["Q_TXT_E"])
        q_txt_span = _span_between(q_tB[-1], q_tE[0])
    except Exception:
        q_txt_span =  []

    return {
        "label_pos": label_pos,
        "demos_txt_spans": demos_txt_spans,  # List[List[int]]，每个 demo 的 text  span（逐 token）
        "q_txt_span": q_txt_span,
    }


def _reduce_heads(x: torch.Tensor, how: str = "mean") -> torch.Tensor:
    # x: (H, N) or (H,) → 返回 (N,) or ()
    if how == "mean": return x.mean(dim=0)
    elif how == "sum":  return x.sum(dim=0)
    elif how == "top5%":
        import math
        k = max(1, math.ceil(x.shape[0] * 0.05))
        topk, _ = torch.topk(x, k, dim=0)
        return topk.mean()
    elif how == "max":  return x.max(dim=0).values
    raise ValueError("head reduce must be mean/sum/max")
# _reduce

def _reduce_tokens(x: torch.Tensor, how: str = "mean") -> torch.Tensor:
    # x: (H, N) or (H,) → 返回 (H,) or ()
    if how == "mean": return x.mean(dim=1)
    elif how == "sum":  return x.sum(dim=1)
    elif how == "top5%":
        import math
        k = max(1, math.ceil(x.shape[0] * 0.05))
        topk, _ = torch.topk(x, k, dim=1)
        return topk.mean()
    elif how == "max":  return x.max(dim=1).values
    raise ValueError("head reduce must be mean/sum/max")

def _reduce_targets(x: torch.Tensor, how: str = "mean") -> torch.Tensor:
    # x: (N,) → 标量
    if x.numel() == 0: 
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    elif how == "mean": return x.mean()

    elif how == "top5%":
        import math
        n = x.numel()
        k = max(1, math.ceil(n * 0.05))
        topk, _ = torch.topk(x, k)
        return topk.mean()
    
    elif how == "sum":  return x.sum()
    elif how == "max":  return x.max()
    raise ValueError("target reduce must be mean/sum/max")

