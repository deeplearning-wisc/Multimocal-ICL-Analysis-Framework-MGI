import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import matplotlib.pyplot as plt
from collections import defaultdict
project_root = "/u/y/u/yuwang/yuwang/ICL_Circuit"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pathlib import Path
import numpy as np

from util import load_model_and_data, inference, my_plots
import StaICC
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random
import json
from scipy.signal import savgol_filter


ICL_model_name_list      = ["google/gemma-3-12b-it","google/gemma-3-27b-it" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct"]
huggingface_token   = "Your_HuggingFace_Token"
k                   = 4
dataset_index_list       = [0,1]
model_forced_reload = False   



import json
def process_examples(json_data):
    processed_dict = {}
    
    for ex_id, content in json_data.items():
        raw_answer = content.get("answer", "")

        if "<OR>" in raw_answer:
            clean_answer = raw_answer.split("<OR>")[0].strip()
        elif "<AND>" in raw_answer:
            parts = raw_answer.split("<AND>")
            clean_answer = " and ".join([p.strip() for p in parts])
        else:
            clean_answer = raw_answer.strip()

        processed_dict[ex_id] = {
            "question": content.get("question", ""),
            "image": content.get("imagename", ""),
            "text": content.get("text", ""),
            "answer": clean_answer,
            "support_idx": [s.strip() for s in content.get("suppoert", "").split(",") if s.strip()]
        }
    
    return processed_dict

def load_mmvt_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    all_dataset = process_examples(json_data)
    prompts = []
    for ex_id, item in all_dataset.items():
        support_idx = item["support_idx"]
        cur_text_list = []
        cur_img_list = []
        cur_label_list = []
        for sup_key in support_idx:
            demo = all_dataset[sup_key]
            text_prompt = "Question: " + demo['question'] + " Answer:"
            cur_text_list.append(text_prompt)
            cur_img_list.append(demo["image"])
            cur_label_list.append("Answer:" + demo['answer']+ "\n")
        cur_text_list.append("Question: " + item['question']) # query
        cur_img_list.append(item["image"])
        cur_label_list.append(item['answer'])
        cur_sample = {"question": cur_text_list, "image":cur_img_list, 'label': cur_label_list}
        prompts.append(cur_sample)
    return prompts

def compute_mask_attention(heat: np.ndarray,
                           correct_mask_img,
                           white_threshold: int = 200):

    correct_mask = np.array(correct_mask_img) > white_threshold

    h = heat.astype(np.float64)

    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    correct_sum = h[correct_mask].sum() /correct_mask.sum()
    out_mask = ~(correct_mask)
    out_sum = h[out_mask].sum() / out_mask.sum()


    return {
        "correct_sum": float(correct_sum),
        "out_sum": float(out_sum),

    }


def map_image_to_mask(image_path):
    img_p = Path(image_path)

    mask_dir = img_p.parent.parent / "gt_mask"

    mask_path = mask_dir / img_p.name
    
    return str(mask_path)



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


def load_images(img_path_list):
    img_list = []
    for image_path in img_path_list:
        try:
            # Open the image file
            img = Image.open(image_path)
            img_list.append(img)

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return img_list


def get_mask_path(img_path: str,
                        mask_subdir: str = "masks",
                        suffix: str = "_mask_ood_color") -> str:
    """
    Given an image path, returns the corresponding mask path.
    such as: /home/.../img_00000.png
       -> /home/.../masks/img_00000_mask_ood_color.png
    """
    p = Path(img_path)
    mask_dir = p.parent / mask_subdir
    mask_name = f"{p.stem}{suffix}{p.suffix}"
    return str(mask_dir / mask_name)


def load_mask_images(img_path_list):
    mask_list = []
    for image_path in img_path_list:
        try:
            mask_path = map_image_to_mask(image_path)

            mask = Image.open(mask_path)
            mask_list.append(mask)

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return  mask_list


def _restore_heatmap(v, H_eff, W_eff, t=1, reduce_mode="mean"):
    n = v.shape[0]
    hw = H_eff * W_eff
    if n == hw and t == 1:
        return v.reshape(H_eff, W_eff)
    if n != t * hw:
        raise ValueError(f"len(v)={n} is not equal to t*H_eff*W_eff={t*hw} ")
    v3 = v.reshape(t, H_eff, W_eff)
    if reduce_mode == "mean":
        return v3.mean(axis=0)
    elif reduce_mode == "max":
        return v3.max(axis=0)
    else:
        return v3.sum(axis=0)


def cal_atten_ratio_on_diff_area(
    attention, 
    correct_mask_list, # List[PIL.Image]
    label_pos_list,  
    spans_per_image, 
    patch_grids=None, 
    direction="label_to_image",  # or "image_to_label"
):
    attn_i = attention.mean(dim=0)


    results_attn_ratio = { "correct_sum": 0,  "out_sum": 0}
    for i, _ in enumerate(correct_mask_list):
        correct_mask = correct_mask_list[i]
    
        b, e = spans_per_image[i]
        label_pos = label_pos_list[i]
        if direction == "label_to_image":
            vec = attn_i[label_pos, b:e]           # (num_vis_tokens,)
        elif direction == "image_to_label":
            vec = attn_i[b:e, label_pos]           # (num_vis_tokens,)
        else:
            raise ValueError("direction must be 'label_to_image' or 'image_to_label'")

        v = vec.to(torch.float32).detach().contiguous().cpu().numpy()
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        if patch_grids is not None and patch_grids[i] is not None:
            H, W = patch_grids[i]
            n = (e - b)
            assert n == H * W or n % (H * W) == 0, \
                f"span_len={n} is not equal to H*W={H*W} "

            heat = _restore_heatmap(v, H, W, reduce_mode="mean")
        else:
            side = int(np.sqrt(len(v)))
            if side * side != len(v):
                raise ValueError("No grid information and cannot form a square matrix.")
            heat = v.reshape(side, side)

        heat = heat.astype(np.float32)

        def mask_size_xy(mask):
            from PIL import Image as _Image
            if isinstance(mask, _Image.Image):
                return mask.size  # (W, H)
            else:
                # ndarray: shape = (H, W) 或 (H, W, C)
                h, w = mask.shape[:2]
                return (w, h)

        target_size = mask_size_xy(correct_mask)

        heat = np.array(
            Image.fromarray(heat).resize(target_size, resample=Image.BILINEAR),
            dtype=np.float32
        )

        ans = compute_mask_attention(heat, correct_mask)
        results_attn_ratio['correct_sum'] += ans['correct_sum']
        results_attn_ratio['out_sum'] += ans['out_sum']

    return results_attn_ratio['correct_sum'], results_attn_ratio['out_sum']

def main(ICL_model_name):
    vars_dict = vars() if "ICL_model" in vars() else locals()
    if ("ICL_model" not in vars_dict) or model_forced_reload:
        print("[Load] Loading ICL and encoder models ...")
        ICL_model, ICL_tknz = load_model_and_data.load_ICL_model(
            ICL_model_name, hf_token=huggingface_token
        )
    else:
        print("[Load] Reusing already loaded models from the current session.")

    try:
        plt.style.use('default')
        plt.rc('font', family='Cambria Math')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif']  = ['Cambria Math'] + plt.rcParams['font.serif']
    except Exception as e:
        print(f"[Warn] Font setup failed, fallback to default. Reason: {e}")

    safe_name = my_plots._sanitize_name(ICL_model_name)

    total_correct_ratio_per_layer = defaultdict(list)
    total_other_ratio_per_layer   = defaultdict(list)
    

    prompts = load_mmvt_dataset("./data/mm-vet/mm-vet-text.json")
    dataset_name = "mmvt"

    correct_ratio_per_layer = defaultdict(list)
    other_ratio_per_layer   = defaultdict(list)
    selected_idx = 0
    num_sam = 0
    for selected_idx in range(len(prompts)):
        selected_sample = prompts[selected_idx]

        attention, spans, patch_grids, demos_label_token_idx, prediction =  inference.obtain_indemo_attention(ICL_model, ICL_tknz, selected_sample)
        mask_list = load_mask_images(selected_sample['image'])
        L = len(attention)
        for lay_id in range(L):
            attns = attention[lay_id]
            correct_ratio, other_ratio = cal_atten_ratio_on_diff_area(
                attention = attns[0],
                correct_mask_list = mask_list[-1:],
                label_pos_list = demos_label_token_idx[-1:],
                spans_per_image= spans[-1:],
                patch_grids= patch_grids,
            )

            correct_ratio_per_layer[lay_id].append(float(correct_ratio))
            other_ratio_per_layer[lay_id].append(float(other_ratio))

            total_correct_ratio_per_layer[lay_id].append(float(correct_ratio))
            total_other_ratio_per_layer[lay_id].append(float(other_ratio))


        layers = sorted(correct_ratio_per_layer.keys())
        def mean_or_zero(v): 
            return float(np.mean(v)) if len(v) > 0 else 0.0

        mean_correct = [mean_or_zero(correct_ratio_per_layer[l]) for l in layers]
        mean_other   = [mean_or_zero(other_ratio_per_layer[l])   for l in layers]

        plt.figure(figsize=(7.5, 4.5))
        plt.plot(layers, mean_correct ,color='green', linestyle='-', label='Correct area ratio (avg)')
        plt.plot(layers, mean_other , color='blue', linestyle='-', label='Other area ratio (avg)')
        plt.xlabel('Layer ID')
        plt.ylabel('Average attention ratio across correct samples')
        plt.title('Per-layer attention ratios on correct samples')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"CORR_{safe_name}_{dataset_name}_query_attention_ratios_per_layer.png", dpi=800)
        

        per_layer_table = [
            {"layer": l, "correct_avg": c, "other_avg": o}
            for l, c, o in zip(layers, mean_correct, mean_other)
        ]
        with open(f"CORR_{safe_name}_{dataset_name}_query_attention_ratios_per_layer.json", "w") as f:
            json.dump(per_layer_table, f, indent=2)

if __name__ == "__main__":
    set_seed(42)
    for icl_model_name in ICL_model_name_list:
        safe_name = my_plots._sanitize_name(icl_model_name)
        print("model_name" , safe_name)
        main(icl_model_name)