# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp


import torch
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, AutoModel
from packaging import version
import transformers
from typing import  List, Dict

def load_ICL_model(
    model_name: str,
    device: str = "auto",                     # ← 默认多卡/auto（路线B）
    hf_token: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    推荐 model_name 例如：
      - "Qwen/Qwen2-VL-2.5-7B-Instruct"
      - "Qwen/Qwen2-VL-7B-Instruct"
      - "Qwen/Qwen2-VL-2B-Instruct"
    说明：
      - device="auto"  -> 多卡拆分（accelerate hooks），禁止再手动 model.to(...)
      - device="cuda" / "cuda:0" 等 -> 单卡，整模放该设备
    """
    if version.parse(transformers.__version__) < version.parse("4.48.0"):
        raise RuntimeError(
            f"transformers=={transformers.__version__} 太旧，建议升级到 >= 4.48（最好 4.49+）。"
        )

    # 根据 device 决定是否单卡
    single_card = (device != "auto")

    kwargs = {
        "torch_dtype": dtype,                  # ← 正确 key
        "device_map": {"": 0} if single_card else "auto",
        "low_cpu_mem_usage": True,
    }
    if hf_token is not None:
        kwargs["token"] = hf_token


    model = None
    processor = None
    last_exc = None
    model_type = model_name.lower()
    if any(t in model_type for t in ["qwen2-vl"]):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
            processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{model_name}'. Last error: {e}\n"
                f"(Earlier error: {last_exc})"
            ) from e

    elif any(t in model_type for t in ["qwen2.5-vl"]):
        try:
            
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
            processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
            
            print("qwen2.5-vl")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{model_name}'. Last error: {e}\n"
                f"(Earlier error: {last_exc})"
            ) from e

    elif any(t in model_type for t in ["gemma-3"]):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, **kwargs).eval()

        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        print(model_name)

    if single_card:
        model.to(device)

    # Qwen3-VL (including MoE: model_type often "qwen3_vl" or "qwen3_vl_moe")
    elif any(t in model_type for t in ["qwen3-vl"]):
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_name, **kwargs)
            processor = AutoProcessor.from_pretrained(model_name, **kwargs)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{model_name}'. Last error: {e}\n"
                f"(Earlier error: {last_exc})"
            ) from e

    elif any(t in model_type for t in ["glm-4.1v"]):
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        # MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("THUDM/GLM-4.1V-9B-Thinking")

    elif any(t in model_type for t in ["llava"]):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        # MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, load_in_4bit=True, **kwargs)
        print("THUDM/GLM-4.1V-9B-Thinking")

    elif any(t in model_type for t in ["internvl"]):
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained(model_name,  trust_remote_code=True, **kwargs)
        processor = AutoTokenizer.from_pretrained(model_name, **kwargs)


    model.eval()
    return model, processor

def load_siglip(
    name: str,
    device: str = "auto",
    hf_token: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    加载 SigLIP / CLIP / 其他 encoder。
    - device="auto"  -> 多卡（不再手动 .to）
    - device="cuda" / "cuda:0" -> 单卡（整模上该卡）
    """
    single_card = (device != "auto")

    kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": 0} if single_card else "auto",
        "low_cpu_mem_usage": True,
    }
    if hf_token is not None:
        kwargs["token"] = hf_token

    model = AutoModel.from_pretrained(name, **kwargs)
    processor = AutoProcessor.from_pretrained(name, token=hf_token)

    if single_card:
        model.to(device)

    model.eval()
    return model, processor

def load_data_from_StaICC_experimentor_v2(experimentor, prompt_cut = "none", demos_type="all", target_label_correction = True):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()[:len(_queries)] #和_queries的长度对齐,其中每个元素这个应该是字典, {"text": [str, ...], "image": [PIL/路径, ...]}

    # prompts : demos+querys,dict,他这里根据不同的cut type选择drop，prompts里的部分信息
    queries = [] # list of test
    for i in range(len(_queries)):
        queries.append({"text": [_queries[i][0][0]],"image":[_queries[i][1]]})

    cut_amount = -1
    if prompt_cut == "label_words": #加上label的信息
        cut_amount = -1 # 因为label的长度是1；； negative->1; positive->2
        for j in range(len(queries)):
            queries[j]['text'][-1] = queries[j]['text'][-1] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[j][2])] + ' '

        for i in range(len(prompts)):
            if target_label_correction: # experimentor.prompt_former._label_space, 用这个可以修改_label_space。。。
                prompts[i]['text'][-1] = prompts[i]['text'][-1] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][2])] + ' '
            else: # 给一个错误的标签 # -1只是针对，query的问题，然后加上他对应的answer
                prompts[i]['text'][-1] = prompts[i]['text'][-1] + experimentor.prompt_former._label_space[(_queries._label_space.index(_queries[i][2]) + 1) % len(_queries._label_space)] + ' '

    elif prompt_cut == "last_sentence_token":#没有label信息
        # input text prefixes: ['sentence: ']
        label_prefix_length = len(experimentor.prompt_former._label_prefix)
        cut_amount = -label_prefix_length - 1

    if  prompt_cut != "last_image_token":
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = prompts[i]['text'][-1][:cut_amount]

        for i in range(len(queries)):
            queries[i]['text'][-1] = queries[i]['text'][-1][:cut_amount]
    else:
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = "" # 是个空的sentence,只有image的信息
        for i in range(len(queries)):
            queries[i]['text'][-1] = ""


    if demos_type == "text-only":
        for p in prompts:
            p['image'][:-1] = [[] for _ in p['image'][:-1]]

    elif demos_type == "image-only":
        for p in prompts:
            p['text'][:-1] = [[] for _ in p['text'][:-1]]

    return prompts, queries


def load_data_from_StaICC_experimentor(experimentor, prompt_cut = "none", demos_type="all", target_label_correction = True):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()[:len(_queries)] #和_queries的长度对齐,其中每个元素这个应该是字典, {"text": [str, ...], "image": [PIL/路径, ...]}

    # prompts : demos+querys,dict,他这里根据不同的cut type选择drop，prompts里的部分信息
    queries = [] # list of test
    for i in range(len(_queries)):
        queries.append({"text": _queries[i][0][0],"image":_queries[i][1]})

    cut_amount = -1
    if prompt_cut == "none":
        cut_amount = -1
    
    elif prompt_cut == "label_words": #加上label的信息
        cut_amount = -1 # 因为label的长度是1；； negative->1; positive->2
        for i in range(len(prompts)):
            if target_label_correction: # experimentor.prompt_former._label_space, 用这个可以修改_label_space。。。
                prompts[i]['text'][-1] = prompts[i]['text'][-1] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][2])] + ' '
            else: # 给一个错误的标签 # -1只是针对，query的问题，然后加上他对应的answer
                prompts[i]['text'][-1] = prompts[i]['text'][-1] + experimentor.prompt_former._label_space[(_queries._label_space.index(_queries[i][2]) + 1) % len(_queries._label_space)] + ' '

    elif prompt_cut == "last_sentence_token":#没有label信息
        # input text prefixes: ['sentence: ']
        label_prefix_length = len(experimentor.prompt_former._label_prefix)
        cut_amount = -label_prefix_length - 1

    if  prompt_cut != "last_image_token":
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = prompts[i]['text'][-1][:cut_amount]
    else:
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = "" # 是个空的sentence,只有image的信息

    if demos_type == "text-only":
        for p in prompts:
            p['image'][:-1] = [[] for _ in p['image'][:-1]]

    elif demos_type == "image-only":
        for p in prompts:
            p['text'][:-1] = [[] for _ in p['text'][:-1]]

    return prompts, queries


def load_data_from_StaICC_experimentor_v3(experimentor, prompt_cut = "none", demos_type="all", target_label_correction = True):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()[:len(_queries)] #和_queries的长度对齐,其中每个元素这个应该是字典, {"text": [str, ...], "image": [PIL/路径, ...]}

    # prompts : demos+querys,dict,他这里根据不同的cut type选择drop，prompts里的部分信息
    queries = [] # list of test
    for i in range(len(_queries)):
        queries.append({"text": [_queries[i][0][0]],"image":[_queries[i][1]]})

    cut_amount = -1
    if prompt_cut == "none":
        cut_amount = -1
    
    elif prompt_cut == "label_words": #加上label的信息
        cut_amount = -1 # 因为label的长度是1；； negative->1; positive->2

        for j in range(len(queries)):
            queries[j]['text'][-1] = queries[j]['text'][-1] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[j][2])] + ' '

        for i in range(len(prompts)):
            if target_label_correction: # experimentor.prompt_former._label_space, 用这个可以修改_label_space。。。
                prompts[i]['text'][-1] = prompts[i]['text'][-1]  + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][2])] + ' '
            else: # 给一个错误的标签 # -1只是针对，query的问题，然后加上他对应的answer
                prompts[i]['text'][-1] = prompts[i]['text'][-1] + experimentor.prompt_former._label_space[(_queries._label_space.index(_queries[i][2]) + 1) % len(_queries._label_space)] + ' '

    elif prompt_cut == "last_sentence_token":#没有label信息
        # input text prefixes: ['sentence: ']
        label_prefix_length = len(experimentor.prompt_former._label_prefix)
        cut_amount = -label_prefix_length - 1

    if  prompt_cut != "last_image_token":
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = prompts[i]['text'][-1][:cut_amount]
    else:
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = "" # 是个空的sentence,只有image的信息

    if demos_type == "text-only":
        for p in prompts:
            p['image'][:-1] = [[] for _ in p['image'][:-1]]

    elif demos_type == "image-only":
        for p in prompts:
            p['text'][:-1] = [[] for _ in p['text'][:-1]]
    return prompts, queries


def load_data_from_StaICC_experimentor_v4(experimentor, prompt_cut = "none", demos_type="all", target_label_correction = True):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()[:len(_queries)] #和_queries的长度对齐,其中每个元素这个应该是字典, {"text": [str, ...], "image": [PIL/路径, ...]}
    queries = [] # list of test
    labels = []
    for i in range(len(_queries)):
        queries.append({"text": [_queries[i][0][0]],"image":[_queries[i][1]]})
        labels.append(_queries[i][2])

    cut_amount = -1
    if prompt_cut == "none":
        cut_amount = -1
    
    elif prompt_cut == "label_words": #加上label的信息
        cut_amount = -1 # 因为label的长度是1；； negative->1; positive->2

        for j in range(len(queries)):
            queries[j]['text'][-1] = queries[j]['text'][-1] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[j][2])] + ' '

        for i in range(len(prompts)):
            if target_label_correction: # experimentor.prompt_former._label_space, 用这个可以修改_label_space。。。
                prompts[i]['text'][-1] = prompts[i]['text'][-1]  + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][2])] + ' '
            else: # 给一个错误的标签 # -1只是针对，query的问题，然后加上他对应的answer
                prompts[i]['text'][-1] = prompts[i]['text'][-1] + experimentor.prompt_former._label_space[(_queries._label_space.index(_queries[i][2]) + 1) % len(_queries._label_space)] + ' '
    elif prompt_cut == "last_sentence_token":#没有label信息
        # input text prefixes: ['sentence: ']
        label_prefix_length = len(experimentor.prompt_former._label_prefix)
        cut_amount = -label_prefix_length - 1

    if  prompt_cut != "last_image_token":
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = prompts[i]['text'][-1][:cut_amount]
    else:
        for i in range(len(prompts)):
            prompts[i]['text'][-1] = "" # 是个空的sentence,只有image的信息

    if demos_type == "text-only":
        for p in prompts:
            p['image'][:-1] = [[] for _ in p['image'][:-1]]

    elif demos_type == "image-only":
        for p in prompts:
            p['text'][:-1] = [[] for _ in p['text'][:-1]]
    return prompts, queries, labels


def load_data_from_indemo_attn_experimentor(experimentor):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_dir()[:len(_queries)] #和_queries的长度对齐,其中每个元
    #{"question": text_input_list, "image":image_input_list , "label": label_list}
    for i in range(len(prompts)):
        query_label = experimentor.prompt_former._label_prefix+ experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][2])] + experimentor.prompt_former._label_affix # "\n"
        prompts[i]["label"].append(query_label) 
    return prompts


def set_abstract_label_space(experimentor):
    new_label_space = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(experimentor.prompt_former._label_space)]
    experimentor.prompt_former.change_label_space(new_label_space)

def find_tokenized_label_word(tokenizer, experimentor, prompt, pythia = False):
    tokenized_prompt_input_ids = tokenizer(prompt)['input_ids']
    fore_runner_loca = []
    label_words_loca = []
    if pythia:
        divider = experimentor.prompt_former._label_prefix[:-1]
    else:
        divider = ' ' + experimentor.prompt_former._label_prefix[:-1]
    tokenized_divider = tokenizer(divider)['input_ids']
    tokenized_divider = tokenized_divider[-2:]
    for i in range(len(tokenized_prompt_input_ids)):
        if tokenized_prompt_input_ids[i:i + len(tokenized_divider)] == tokenized_divider:
            fore_runner_loca.append(i + 1)
            label_words_loca.append(i + 2)
    return fore_runner_loca, label_words_loca

def load_demonstrations_and_labels(experimentor):
    demo_inputs = []
    demo_labels = []
    for i in range(len(experimentor.demonstration_sampler)):
        temp_demo_inputs = []
        temp_demo_labels = []
        demonstration_indexs = experimentor.demonstration_sampler[i]
        for index in demonstration_indexs:
            temp_demo_inputs.append(experimentor.triplet_dataset.demonstration.get_input_text(index)[0])
            temp_demo_labels.append(experimentor.triplet_dataset.demonstration.get_label(index))
        demo_inputs.append(temp_demo_inputs)
        demo_labels.append(temp_demo_labels)

    queries = []
    for i in range(len(experimentor.test_set())):
        queries.append(experimentor.test_set().get_input_text(i)[0])
    
    return demo_inputs, demo_labels, queries