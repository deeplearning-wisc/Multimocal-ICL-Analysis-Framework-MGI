
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
project_root = "xxx"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pathlib import Path
import numpy as np

from util import load_model_and_data, inference, my_plots
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



def main(ICL_model_name, copy_att_setup,layer_idx, my_alpha):
    print("[Init] Build benchmark ...")
    benchmark = StaICC.MLLM_Normal(k)

    vars_dict = vars() if "ICL_model" in vars() else locals()
    if ("ICL_model" not in vars_dict) or model_forced_reload:
        print("[Load] Loading ICL and encoder models ...")
        ICL_model, ICL_tknz = load_model_and_data.load_ICL_model(
            ICL_model_name, hf_token=huggingface_token
        )

        if copy_att_setup:
            if "gemma" in ICL_model_name:
                ICL_model = copy_atten_gemma.wire_intervention(ICL_model, mid_layer=layer_idx, alpha=my_alpha)
            else:
                ICL_model = copy_atten.wire_intervention(ICL_model, mid_layer=layer_idx, alpha=my_alpha)                

    safe_name = my_plots._sanitize_name(ICL_model_name)

    results_per_sample = []
    error_num = 0
    base_error_num = 0
    for dataset_index in dataset_index_list:
        prompts = load_model_and_data.load_data_from_indemo_attn_experimentor( benchmark[dataset_index])
        dataset_name = benchmark[dataset_index].dataset_name
        selected_idx = 0
        for selected_idx in range(len(prompts)):
            selected_sample = prompts[selected_idx]
            label = prompts[selected_idx]["label"][-1]
            
            spans, patch_grids, demos_label_token_idx =  inference.obtain_span_label_tok_idx(ICL_tknz, selected_sample)

            if "gemma" in ICL_model_name:
                if copy_att_setup:
                    copy_atten_gemma.enable_intervention(ICL_model, True)
                    copy_atten_gemma.set_intervention_state(ICL_model, demos_label_token_idx, spans)
                
                prediction = inference.obtain_prediction(ICL_model,ICL_tknz,selected_sample)

                # baseline
                copy_atten_gemma.enable_intervention(ICL_model, False)
                pred_base = inference.obtain_prediction(ICL_model, ICL_tknz, selected_sample)
                copy_atten_gemma.enable_intervention(ICL_model, True)

                is_correct_interv = prediction in label
                is_correct_base   = pred_base in label
            else:
                # use the spans & label_pos_list
                if copy_att_setup:
                    copy_atten.enable_intervention(ICL_model, True)
                    copy_atten.set_intervention_state(ICL_model, demos_label_token_idx, spans)
                prediction = inference.obtain_prediction(ICL_model,ICL_tknz,selected_sample)
                # baseline
                copy_atten.enable_intervention(ICL_model, False)
                pred_base = inference.obtain_prediction(ICL_model, ICL_tknz, selected_sample)
                copy_atten.enable_intervention(ICL_model, True)

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


    save_dir = Path("results/attention_intervention")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{safe_name}-layer{layer_idx}-alpha{my_alpha}_intervention_result.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {save_path}")
    print(f"Baseline acc: {acc_base:.3f} | Intervention acc: {acc_interv:.3f}")


if __name__ == "__main__":
    set_seed(42)
    for icl_model_name in ICL_model_name_list:
        safe_name = my_plots._sanitize_name(icl_model_name)
        print("model_name" , safe_name)
        main(icl_model_name, copy_att_setup,1,0.5)