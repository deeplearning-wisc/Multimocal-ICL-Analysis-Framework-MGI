
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]='1'#0,,3
project_root = "xxxx"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pathlib import Path
import numpy as np
import gc  # 引入垃圾回收模块
from util import load_mllm_model_and_data, mllm_inference, my_plots
import StaICC
import torch
import random
import json
import re
import copy_atten
import copy_atten_gemma

copy_att_setup = True
ICL_model_name_list      = ["google/gemma-3-12b-it","google/gemma-3-27b-it" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct"]
huggingface_token   = "Your_HuggingFace_Token"
k_list                  = [1,2,3]
dataset_index_list       = [0]
model_forced_reload = False

import numpy as np
from PIL import Image
from typing import Optional, Literal, Dict, Any, Tuple, List

def _integral_image_sum(img: np.ndarray) -> np.ndarray:
    # 简单积分图（含顶/左边界 0 填充）
    integ = np.pad(img, ((1,0),(1,0)), mode='constant')
    np.cumsum(integ, axis=0, out=integ)
    np.cumsum(integ, axis=1, out=integ)
    return integ

def _rect_sum(integ: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    # [y0,y1)×[x0,x1) 的和
    return float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])

def downsample_mask_to_grid(mask, H: int, W: int,
                            mode: Literal["area","majority"]="area",
                            thresh: float=0.5) -> np.ndarray:
    """把像素级 mask 下采样到 (H,W) 网格；返回 ∈[0,1] 的 float32 网格。"""
    if isinstance(mask, Image.Image):
        mask = mask.convert("L")
        mask = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)
    else:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

    Mh, Mw = mask.shape
    y_edges = np.linspace(0, Mh, H+1, dtype=int)
    x_edges = np.linspace(0, Mw, W+1, dtype=int)
    integ = _integral_image_sum(mask)
    cell_area = (y_edges[1:] - y_edges[:-1])[:, None] * (x_edges[1:] - x_edges[:-1])[None, :]

    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        y0, y1 = y_edges[i], y_edges[i+1]
        for j in range(W):
            x0, x1 = x_edges[j], x_edges[j+1]
            s = _rect_sum(integ, y0, x0, y1, x1)
            out[i, j] = s / max(1, cell_area[i, j])

    if mode == "majority":
        out = (out > thresh).astype(np.float32)
    return out

def mask_to_image_token_indices(
    mask,
    H: int,
    W: int,
    img_span: Optional[Tuple[int, int]] = None,   # 该图像在整体序列里的 [b, e)，若无就传 None
    mode: Literal["area","majority"] = "area",
    select: Literal["nonzero", "thresh"] = "nonzero",
    thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Returns:
      - local_indices: indices relative to a single frame (H*W); if t>1 and
        frame=None, the indices for each frame are concatenated
      - global_indices: indices offset by img_span when provided; otherwise None
      - weights: weights aligned with local_indices, derived from the downsampled
        occupancy grid
      - grid_bool: (H, W) boolean hit grid for a single frame
      - grid_weights: (H, W) occupancy weights for a single frame
    """
    grid = downsample_mask_to_grid(mask, H, W, mode=("area" if mode=="area" else "majority"), thresh=thresh)
    if select == "nonzero":
        hit = grid > 0
    else:
        hit = grid >= thresh

    flat_idx_1 = np.flatnonzero(hit)  # 相对单帧 0..H*W-1 的索引
    local_indices = flat_idx_1

    # 计算全局索引（若给了 [b,e)）
    if img_span is not None:
        b, e = img_span
        global_indices = (b + local_indices).astype(np.int64)
    else:
        global_indices = None

    return global_indices


def get_mask_path(img_path: str,
                        mask_subdir: str = "masks",
                        suffix: str = "_mask_ood_color") -> str:
    """
    Generate the corresponding OOD color-mask path from the original image path.
    Example: /home/.../img_00000.png
       -> /home/.../masks/img_00000_mask_ood_color.png
    """
    p = Path(img_path)
    mask_dir = p.parent / mask_subdir
    mask_name = f"{p.stem}{suffix}{p.suffix}"  # 保留原扩展名
    return str(mask_dir / mask_name)

def load_mask_images(img_path_list):
    color_mask_list = []
    shape_mask_list = []
    for image_path in img_path_list:
        try:
            color_mask_path = get_mask_path(image_path,suffix='_mask_ood_color')
            shape_mask_path = get_mask_path(image_path, suffix='_mask_ood_shape')

            color_mask = Image.open(color_mask_path)
            shape_mask = Image.open(shape_mask_path)

            color_mask_list.append(color_mask)
            shape_mask_list.append(shape_mask)

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return color_mask_list, shape_mask_list


def get_optimal_layer_idx(attention, q_idx_list, spans, metric="entropy", eps=1e-12):
    """
    attention: list/tuple, len = L
        attention[l] shape ~ [n_heads, seq_len, seq_len]
    q_idx_list: list[int]
    spans: list[tuple[int, int]]  (b, e)
    metric: "entropy" 或 "var"
    """
    L = len(attention)
    layer_scores = []

    for lay_id in range(L):
        attn = attention[lay_id]
        score = 0.0
        for q_idx, (b, e) in zip(q_idx_list, spans):
            P = attn[q_idx, b:e]          # shape [span_len]

            if metric == "entropy":
                probs = P / (P.sum() + eps)
                probs = probs.clamp_min(eps)
                entropy = -(probs * probs.log()).sum().item()
                score += entropy
            elif metric == "var":
                var = P.var(unbiased=False).item()
                score += var
            else:
                raise ValueError("metric must be 'entropy' or 'var'")
        layer_scores.append(score)

    best_layer = int(torch.tensor(layer_scores).argmin().item())
    return best_layer

def set_seed(seed: int = 42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✅ Random seed set to {seed}")

def parse_int(s):
    if s is None:
        return None
    m = re.search(r"-?\d+", str(s))
    return int(m.group()) if m else None

def main(ICL_model_name, copy_att_setup,layer_idx, my_alpha, my_beta=5, my_top_ratio=1.5, my_low_ratio=1, seed=42):
    print("[Init] Build benchmark ...")
    for k in k_list:
        benchmark = StaICC.MLLM_Normal(k)

        vars_dict = vars() if "ICL_model" in vars() else locals()
        if ("ICL_model" not in vars_dict) or model_forced_reload:
            print("[Load] Loading ICL and encoder models ...")
            ICL_model, ICL_tknz = load_mllm_model_and_data.load_ICL_model(
                ICL_model_name, hf_token=huggingface_token
            )
            if copy_att_setup:
                if "gemma" in ICL_model_name: # model先正常运算一次，然后计算start_layer_idx,开始设置warp_attn
                    ICL_model = copy_atten_gemma.wire_intervention(ICL_model, mid_layer=layer_idx, alpha=my_alpha, beta = my_beta, top_ratio= my_top_ratio, low_ratio= my_low_ratio, mode='improve')
                else:
                    ICL_model = copy_atten.wire_intervention(ICL_model, mid_layer=layer_idx, alpha=my_alpha, beta = my_beta, top_ratio= my_top_ratio, low_ratio= my_low_ratio, mode='improve')

        safe_name = my_plots._sanitize_name(ICL_model_name)
        # 保存每个样本的结果
        results_per_sample = []
        error_num = 0
        base_error_num = 0
        dataset_name = benchmark[dataset_index_list[0]].dataset_name
        for dataset_index in dataset_index_list:
            # dataset_name = benchmark[dataset_index].dataset_name
            prompts = load_mllm_model_and_data.load_data_from_indemo_attn_experimentor(benchmark[dataset_index])
            for selected_idx in range(len(prompts)):
                selected_sample = prompts[selected_idx]
                label = prompts[selected_idx]["label"][-1]
                
                spans, patch_grids, demos_label_token_idx, attns =  mllm_inference.obtain_span_label_tok_idx_and_attn(ICL_model, ICL_tknz, selected_sample)
                cur_start_layer= get_optimal_layer_idx(attns, demos_label_token_idx, spans, metric="entropy", eps=1e-12)
                if copy_att_setup:
                    if "gemma" in ICL_model_name: 
                        ICL_model = copy_atten_gemma.wire_intervention(ICL_model, mid_layer=cur_start_layer, alpha=my_alpha, beta = my_beta, top_ratio= my_top_ratio, low_ratio= my_low_ratio, mode='already')
                    else:
                        ICL_model = copy_atten.wire_intervention(ICL_model, mid_layer=cur_start_layer, alpha=my_alpha, beta = my_beta, top_ratio= my_top_ratio, low_ratio= my_low_ratio, mode='already')

  
                if "gemma" in ICL_model_name:
                    if copy_att_setup:
                        copy_atten_gemma.enable_intervention(ICL_model, True)
                        copy_atten_gemma.set_intervention_state(ICL_model, demos_label_token_idx, spans)
                        if "clock" in dataset_name or "operator" in dataset_name:
                            prediction = mllm_inference.obtain_vqa_prediction(ICL_model,ICL_tknz, selected_sample, max_generation_length=4)
                        else:
                            prediction = mllm_inference.obtain_prediction(ICL_model,ICL_tknz, selected_sample)

                    copy_atten_gemma.enable_intervention(ICL_model, False)
                    if "clock" in dataset_name or "operator" in dataset_name:
                        pred_base = mllm_inference.obtain_vqa_prediction(ICL_model, ICL_tknz, selected_sample, max_generation_length=4)
                    else:
                        pred_base = mllm_inference.obtain_prediction(ICL_model, ICL_tknz, selected_sample)
                    
                    copy_atten_gemma.enable_intervention(ICL_model, True)

                    if "clock" in dataset_name or "operator" in dataset_name:
                        label_number = parse_int(label)
                        prediction_number = parse_int(prediction)
                        pred_base_number = parse_int(pred_base)

                        if prediction_number is None:
                            is_correct_interv = False
                        else:
                            is_correct_interv =  (label_number == prediction_number)

                        if pred_base_number is None:
                            is_correct_base   =  False
                        else:
                            is_correct_base   =  (label_number ==  pred_base_number)
                    else:
                        is_correct_interv = prediction in label
                        is_correct_base   = pred_base in label
                else:
                    if copy_att_setup:
                        copy_atten.enable_intervention(ICL_model, True)
                        copy_atten.set_intervention_state(ICL_model, demos_label_token_idx, spans)
                        if "clock" in dataset_name or "operator" in dataset_name:
                            prediction = mllm_inference.obtain_vqa_prediction(ICL_model,ICL_tknz, selected_sample, max_generation_length=6)
                        else:
                            prediction = mllm_inference.obtain_prediction(ICL_model,ICL_tknz, selected_sample)
                            

                    copy_atten.enable_intervention(ICL_model, False)
                    if "clock" in dataset_name or "operator" in dataset_name:
                        pred_base = mllm_inference.obtain_vqa_prediction(ICL_model, ICL_tknz, selected_sample, max_generation_length=6)
                    else:
                        pred_base = mllm_inference.obtain_prediction(ICL_model, ICL_tknz, selected_sample)
                    copy_atten.enable_intervention(ICL_model, True)

                    if "clock" in dataset_name or "operator" in dataset_name:
                        label_number = parse_int(label)
                        prediction_number = parse_int(prediction)
                        pred_base_number = parse_int(pred_base)

                        if prediction_number is None:
                            is_correct_interv = False
                        else:
                            is_correct_interv =  (label_number == prediction_number)

                        if pred_base_number is None:
                            is_correct_base   =  False
                        else:
                            is_correct_base   =  (label_number ==  pred_base_number)
                            
                    else:
                        is_correct_interv = prediction in label
                        is_correct_base   = pred_base in label

                if is_correct_interv and (not is_correct_base):
                    print("11111")

                if (not is_correct_interv) and is_correct_base:
                    print("not good")
                if not is_correct_interv:
                    error_num += 1
                if not is_correct_base:
                    base_error_num += 1
                
                results_per_sample.append({
                    "dataset": dataset_name,
                    "n-shot": k,
                    "index": selected_idx,
                    "start_layer_idx":cur_start_layer,
                    "label": selected_sample["label"],
                    "image" : selected_sample['image'],
                    "prediction_intervene": prediction,
                    "prediction_baseline": pred_base,
                    "correct_intervene": bool(is_correct_interv),
                    "correct_baseline": bool(is_correct_base)
                })
            print("error_num", error_num)
            print("base_error_num", base_error_num)
            
        total = len(results_per_sample)
        acc_interv = 1 - error_num / total
        acc_base   = 1 - base_error_num / total

        summary = {
            "model": safe_name,
            "total_samples": total,
            "acc_intervene": acc_interv,
            "acc_baseline": acc_base,
            "error_intervene": error_num,
            "error_baseline": base_error_num,
            "dataset_indices": dataset_index_list,
            "details": results_per_sample
        }

        save_dir = Path("results/ours_v1_rebuttal")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{k}shot-{dataset_name}-{safe_name}-start{layer_idx}layer-alpha{my_alpha}_beta{my_beta}_top{my_top_ratio}_low{my_low_ratio}_intervention_result.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Results saved to: {save_path}")
        print(f"Baseline acc: {acc_base:.5f} | Intervention acc: {acc_interv:.5f}")




if __name__ == "__main__":
    # top_times =  [1.5, 1.5, 1.5, 1.2] 
    # low_times = [0.6, 0.6, 0.001, 0.6]
    # top_ratio = [1.5,  1.5, 1.5, 2.0]
    # low_ratio =  [1.5, 1.5, 1.5, 1.5]
    top_times = [2]
    for cur_seed in [42]:
        set_seed(cur_seed)
        for idx in range(len(top_times)):
            for slayer_id in [17]:
                for icl_model_name in ICL_model_name_list:
                    safe_name = my_plots._sanitize_name(icl_model_name)
                    print("model_name", safe_name)
                    main(icl_model_name, copy_att_setup, slayer_id, top_times[idx], 0.1, my_top_ratio=1.5, my_low_ratio=1.5, seed=cur_seed)
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"Finished seed {cur_seed} for {safe_name}. Memory cleared.")

