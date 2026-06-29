
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
project_root = "/u/y/u/yuwang/yuwang/ICL_Circuit"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pathlib import Path
import numpy as np
import math
from util import load_mllm_model_and_data, mllm_inference, my_plots
import StaICC
import torch
import random
import json

import copy_atten
import copy_atten_gemma

copy_att_setup = True

ICL_model_name_list      = ["google/gemma-3-12b-it","google/gemma-3-27b-it" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct"]
huggingface_token   = "Your_HuggingFace_Token"
k                   = 4
dataset_index_list       = [1,0]
model_forced_reload = False 

import numpy as np
from PIL import Image
from typing import Optional, Literal, Dict, Any, Tuple, List

def _integral_image_sum(img: np.ndarray) -> np.ndarray:
    integ = np.pad(img, ((1,0),(1,0)), mode='constant')
    np.cumsum(integ, axis=0, out=integ)
    np.cumsum(integ, axis=1, out=integ)
    return integ

def _rect_sum(integ: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    return float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])

def downsample_mask_to_grid(mask, H: int, W: int,
                            mode: Literal["area","majority"]="area",
                            thresh: float=0.5) -> np.ndarray:
    """Downsample a pixel-level mask to an (H, W) float32 grid in [0, 1]."""
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
    img_span: Optional[Tuple[int, int]] = None,   # The image's [b, e) span in the full sequence, or None
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

    flat_idx_1 = np.flatnonzero(hit)  # Indices relative to one frame: 0..H*W-1
    local_indices = flat_idx_1

    # Compute global indices when [b, e) is provided
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
    mask_name = f"{p.stem}{suffix}{p.suffix}"  # Preserve the original extension
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

def set_seed(seed: int = 42):
    # Python built-in RNG
    random.seed(seed)
    
    # NumPy RNG
    np.random.seed(seed)
    
    # PyTorch CPU RNG
    torch.manual_seed(seed)
    
    # PyTorch GPU RNGs (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cuDNN deterministic (potentially at a small performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable used by some libraries (such as DataLoader)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"✅ Random seed set to {seed}")


def main(ICL_model_name, copy_att_setup,layer_idx, my_alpha, my_beta=5):
    print("[Init] Build benchmark ...")
    benchmark = StaICC.MLLM_Normal(k)

    vars_dict = vars() if "ICL_model" in vars() else locals()
    if ("ICL_model" not in vars_dict) or model_forced_reload:
        print("[Load] Loading ICL and encoder models ...")
        ICL_model, ICL_tknz = load_mllm_model_and_data.load_ICL_model(
            ICL_model_name, hf_token=huggingface_token
        )

        if copy_att_setup:
            if "gemma" in ICL_model_name:
                ICL_model = copy_atten_gemma.wire_intervention(ICL_model, mid_layer=layer_idx, alpha=my_alpha, beta = my_beta, mode='improve')
            else:
                ICL_model = copy_atten.wire_intervention(ICL_model, mid_layer=layer_idx, alpha=my_alpha, beta = my_beta, mode='improve')

    safe_name = my_plots._sanitize_name(ICL_model_name)
    # Store the result for each sample
    results_per_sample = []
    error_num = 0
    base_error_num = 0
    for dataset_index in dataset_index_list:
        prompts = load_mllm_model_and_data.load_data_from_indemo_attn_experimentor( benchmark[dataset_index])
        dataset_name = benchmark[dataset_index].dataset_name
        selected_idx = 0
        for selected_idx in range(len(prompts)):
            selected_sample = prompts[selected_idx]
            label = prompts[selected_idx]["label"][-1]
            spans, patch_grids, demos_label_token_idx =  mllm_inference.obtain_span_label_tok_idx(ICL_tknz, selected_sample)
            correct_regin_spans=[]
            # Obtain the correct-region indices from the masks
            color_mask_list, shape_mask_list = load_mask_images(selected_sample['image'])
            if "color" in dataset_name:
                correct_mask_list = color_mask_list
            else:
                correct_mask_list = shape_mask_list

            for i, (span, mask_img) in enumerate(zip(spans, correct_mask_list)):
                b, e = span
                span_len = e - b

                if patch_grids is not None and i < len(patch_grids) and patch_grids[i] is not None:
                    H, W = patch_grids[i]

                else:
                    s = int(math.sqrt(span_len))
                    assert s * s == span_len, f"span_len={span_len} is not a perfect square; without a grid, H, W, or t>1 cannot be inferred."
                    H = W = s

                res = mask_to_image_token_indices(mask=mask_img, H=H, W=W, img_span=(b, e))
                correct_regin_spans.append(res)
                
            if "gemma" in ICL_model_name:
                if copy_att_setup:
                    copy_atten_gemma.enable_intervention(ICL_model, True)
                    copy_atten_gemma.set_intervention_state(ICL_model, demos_label_token_idx, spans, correct_regin_spans)
                    prediction = mllm_inference.obtain_prediction(ICL_model,ICL_tknz, selected_sample)

                # Run the baseline
                copy_atten_gemma.enable_intervention(ICL_model, False)
                pred_base = mllm_inference.obtain_prediction(ICL_model, ICL_tknz, selected_sample)
                copy_atten_gemma.enable_intervention(ICL_model, True)

                is_correct_interv = prediction in label
                is_correct_base   = pred_base in label
            else:
                if copy_att_setup:
                    copy_atten.enable_intervention(ICL_model, True)
                    copy_atten.set_intervention_state(ICL_model, demos_label_token_idx, spans, correct_regin_spans)
                    prediction = mllm_inference.obtain_prediction(ICL_model,ICL_tknz,selected_sample)
                
                # Run the baseline
                copy_atten.enable_intervention(ICL_model, False)
                pred_base = mllm_inference.obtain_prediction(ICL_model, ICL_tknz, selected_sample)
                copy_atten.enable_intervention(ICL_model, True)

                is_correct_interv = prediction in label
                is_correct_base   = pred_base in label
            # print("pred_ours:", prediction)
            # print("pred_base", pred_base)
            # print("label",label)
            # print("================================")


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
                "index": selected_idx,
                "label": label,
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

    # Output file path
    save_dir = Path("results/attention_intervention")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{safe_name}-layer{layer_idx}-alpha{my_alpha}_beta{my_beta}_intervention_result.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {save_path}")
    print(f"Baseline acc: {acc_base:.3f} | Intervention acc: {acc_interv:.3f}")


if __name__ == "__main__":
    set_seed(42)
    for icl_model_name, slayer_id in zip(ICL_model_name_list, start_layer):
        safe_name = my_plots._sanitize_name(icl_model_name)
        print("model_name" , safe_name) # beta<1, alpha>10
        main(icl_model_name, copy_att_setup, slayer_id, 30, 0.001)
