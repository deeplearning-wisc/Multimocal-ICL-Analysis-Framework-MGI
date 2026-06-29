#!/bin/bash
# 40 17
# "Qwen/Qwen2.5-VL-32B-Instruct" "google/gemma-3-12b-it"

# ICL_model_name_list=("Qwen/Qwen2.5-VL-7B-Instruct" "google/gemma-3-27b-it" "google/gemma-3-12b-it" "Qwen/Qwen2.5-VL-32B-Instruct")
# start_layer=(16 37 17 40) # 
# export CUDA_VISIBLE_DEVICES="4,5"
# ############ ours
# export PYTHONPATH=/nobackup2/yuwang/ICL_Circuit:$PYTHONPATH
# for i in "${!ICL_model_name_list[@]}"; do
#       model_name="${ICL_model_name_list[$i]}"
#       slayer_id="${start_layer[$i]}"
#       echo "running model: $model_name layer: $slayer_id"
#       python vqa_eval/evaluate.py \
#         --rices \
#         --model_name "$model_name" \
#         --save_dir "results_vqa" \
#         --layer_idx "$slayer_id" \
#         --my_alpha 2 \
#         --my_beta 0.01 \
#         --top_ratio 1.7 \
#         --low_ratio 0.8 \
#         --vision_encoder_pretrained openai \
#         --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
#         --precision amp_bf16 \
#         --batch_size 1 \
#         --num_samples 2048 \
#         --shots 4 \
#         --eval_vqav2 \
#         --vqav2_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/train2014" \
#         --vqav2_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_train2014_annotations.json" \
#         --vqav2_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
#         --vqav2_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/val2014" \
#         --vqav2_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_val2014_annotations.json" \
#         --vqav2_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"

# done



ICL_model_name_list=("google/gemma-3-12b-it" "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct")
start_layer=(17 40 16) # 
export CUDA_VISIBLE_DEVICES="4,5"
############ ours
export PYTHONPATH=/nobackup2/yuwang/ICL_Circuit:$PYTHONPATH
for i in "${!ICL_model_name_list[@]}"; do
      model_name="${ICL_model_name_list[$i]}"
      slayer_id="${start_layer[$i]}"
      echo "running model: $model_name layer: $slayer_id"
      python vqa_eval/evaluate.py \
        --rices \
        --model_name "$model_name" \
        --save_dir "results_vqa" \
        --layer_idx "$slayer_id" \
        --copy_att_setup True \
        --my_alpha 6 \
        --my_beta 0.01 \
        --top_ratio 1.7 \
        --low_ratio 0.8 \
        --vision_encoder_pretrained openai \
        --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
        --precision amp_bf16 \
        --batch_size 1 \
        --num_samples 2048 \
        --shots 4 \
        --eval_vqav2 \
        --vqav2_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/train2014" \
        --vqav2_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_train2014_annotations.json" \
        --vqav2_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
        --vqav2_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/val2014" \
        --vqav2_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_val2014_annotations.json" \
        --vqav2_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"

done



ICL_model_name_list=("google/gemma-3-27b-it")
start_layer=(37) # 
export CUDA_VISIBLE_DEVICES="4,5"
############ ours
export PYTHONPATH=/nobackup2/yuwang/ICL_Circuit:$PYTHONPATH
for i in "${!ICL_model_name_list[@]}"; do
      model_name="${ICL_model_name_list[$i]}"
      slayer_id="${start_layer[$i]}"
      echo "running model: $model_name layer: $slayer_id"
      python vqa_eval/evaluate.py \
        --rices \
        --model_name "$model_name" \
        --save_dir "results_vqa" \
        --layer_idx "$slayer_id" \
        --copy_att_setup True \
        --my_alpha 2 \
        --my_beta 0.01 \
        --top_ratio 1.7 \
        --low_ratio 0.8 \
        --vision_encoder_pretrained openai \
        --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
        --precision amp_bf16 \
        --batch_size 1 \
        --num_samples 2048 \
        --shots 4 \
        --eval_vqav2 \
        --vqav2_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/train2014" \
        --vqav2_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_train2014_annotations.json" \
        --vqav2_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
        --vqav2_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/val2014" \
        --vqav2_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_val2014_annotations.json" \
        --vqav2_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"

done




#  "Qwen/Qwen2.5-VL-32B-Instruct" "google/gemma-3-12b-it" "google/gemma-3-27b-it"
#  40 17 37
# ICL_model_name_list=("Qwen/Qwen2.5-VL-7B-Instruct")
# start_layer=(16)
# export C
# export PYTHONPATH=/nobackup2/yuwang/ICL_Circuit:$PYTHONPATH
# ############ baseline
# for i in "${!ICL_model_name_list[@]}"; do
#       model_name="${ICL_model_name_list[$i]}"
#       slayer_id="${start_layer[$i]}"
#       echo "running model: $model_name layer: $slayer_id"
#       python vqa_eval/evaluate.py \
#         --rices \
#         --model_name "$model_name" \
#         --save_dir "results_vqa" \
#         --layer_idx "$slayer_id" \
#         --my_alpha 6 \
#         --my_beta 0.01 \
#         --vision_encoder_pretrained openai \
#         --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
#         --precision amp_bf16 \
#         --batch_size 1 \
#         --num_samples 2048 \
#         --shots 4 \
#         --eval_ok_vqa \
#         --ok_vqa_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/train2014" \
#         --ok_vqa_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_train2014_annotations.json" \
#         --ok_vqa_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
#         --ok_vqa_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/val2014" \
#         --ok_vqa_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_val2014_annotations.json" \
#         --ok_vqa_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_val2014_questions.json" \

# done



# # "Qwen/Qwen2.5-VL-32B-Instruct"  40
# ICL_model_name_list=("Qwen/Qwen2.5-VL-7B-Instruct" "google/gemma-3-12b-it" "google/gemma-3-27b-it")
# start_layer=(16 17 37)
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export PYTHONPATH=/nobackup2/yuwang/ICL_Circuit:$PYTHONPATH
# ############ baseline
# for i in "${!ICL_model_name_list[@]}"; do
#       model_name="${ICL_model_name_list[$i]}"
#       slayer_id="${start_layer[$i]}"
#       echo "running model: $model_name layer: $slayer_id"
#       python vqa_eval/evaluate.py \
#         --rices \
#         --model_name "$model_name" \
#         --save_dir "results_vqa" \
#         --layer_idx "$slayer_id" \
#         --my_alpha 6 \
#         --my_beta 0.01 \
#         --vision_encoder_pretrained openai \
#         --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
#         --precision amp_bf16 \
#         --batch_size 1 \
#         --num_samples 2048 \
#         --shots 4 \
#         --eval_vqav2 \
#         --vqav2_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/train2014" \
#         --vqav2_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_train2014_annotations.json" \
#         --vqav2_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
#         --vqav2_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/val2014" \
#         --vqav2_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_val2014_annotations.json" \
#         --vqav2_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json" \

# done




# #"Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct" 
# ICL_model_name_list=("google/gemma-3-12b-it" "google/gemma-3-27b-it")
# start_layer=(17 37) #16 40 
# export CUDA_LAUNCH_BLOCKING=1
# ############ ours
# export PYTHONPATH=/nobackup2/yuwang/ICL_Circuit:$PYTHONPATH
# for i in "${!ICL_model_name_list[@]}"; do
#       model_name="${ICL_model_name_list[$i]}"
#       slayer_id="${start_layer[$i]}"
#       echo "running model: $model_name layer: $slayer_id"
#       python vqa_eval/evaluate.py \
#         --rices \
#         --model_name "$model_name" \
#         --save_dir "results_vqa" \
#         --layer_idx "$slayer_id" \
#         --copy_att_setup True \
#         --my_alpha 6 \
#         --my_beta 0.01 \
#         --vision_encoder_pretrained openai \
#         --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
#         --precision amp_bf16 \
#         --batch_size 1 \
#         --num_samples 2048 \
#         --shots 4 \
#         --eval_ok_vqa \
#         --ok_vqa_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/train2014" \
#         --ok_vqa_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_train2014_annotations.json" \
#         --ok_vqa_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
#         --ok_vqa_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/val2014" \
#         --ok_vqa_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_val2014_annotations.json" \
#         --ok_vqa_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_val2014_questions.json" \
#         --eval_vqav2 \
#         --vqav2_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/train2014" \
#         --vqav2_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_train2014_annotations.json" \
#         --vqav2_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
#         --vqav2_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/val2014" \
#         --vqav2_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_val2014_annotations.json" \
#         --vqav2_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"

# done
#######################  #
###########################
        # --eval_ok_vqa \
        # --ok_vqa_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/train2014" \
        # --ok_vqa_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_train2014_annotations.json" \
        # --ok_vqa_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
        # --ok_vqa_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/val2014" \
        # --ok_vqa_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_val2014_annotations.json" \
        # --ok_vqa_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_val2014_questions.json" \
        # --eval_vqav2 \
        # --vqav2_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/train2014" \
        # --vqav2_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_train2014_annotations.json" \
        # --vqav2_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
        # --vqav2_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/val2014" \
        # --vqav2_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_mscoco_val2014_annotations.json" \
        # --vqav2_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json" \
#2048 
        # 
        # 
        # --eval_textvqa \
        # --textvqa_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/textvqa/train_images/" \
        # --textvqa_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/textvqa/train_questions_vqa_format.json" \
        # --textvqa_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/textvqa/train_annotations_vqa_format.json" \
        # --textvqa_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/textvqa/val_questions_vqa_format.json" \
        # --textvqa_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/textvqa/val_annotations_vqa_format.json" \
        # --eval_ok_vqa \
        # --ok_vqa_train_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/train2014" \
        # --ok_vqa_train_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_train2014_annotations.json" \
        # --ok_vqa_train_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
        # --ok_vqa_test_image_dir_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/val2014" \
        # --ok_vqa_test_annotations_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/mscoco_val2014_annotations.json" \
        # --ok_vqa_test_questions_json_path "/nobackup2/yuwang/ICL_Circuit/data/okvqa/OpenEnded_mscoco_val2014_questions.json" \