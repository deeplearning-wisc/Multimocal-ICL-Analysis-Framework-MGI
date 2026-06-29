
import sys
import os
## Change the working directory
project_root = "/home/yuw/ICL_Circuit"
os.chdir(project_root)

# 把项目根目录加入 Python 模块搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)


## Import libraries
from util import load_mllm_model_and_data,  mllm_inference
import StaICC
import matplotlib.pyplot as plt

## Some definations for the plots.
plt.style.use('default')
plt.rc('font',family='Cambria Math')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']

ICL_model_name = "THUDM/GLM-4.1V-9B-Thinking"  # OpenGVLab/InternVL2-8B # "OpenGVLab/InternVL3-8B-hf"
encoder_model_name = "google/siglip-so400m-patch14-384"
huggingface_token   = os.environ.get("HF_TOKEN")
ICL_selected_token_type = "label_words" 
k = 4 
dataset_index = 0
pesudo_dataset_index =  0 
model_forced_reload = False

benchmark = StaICC.MLLM_Normal(k)

vars_dict = vars() if "ICL_model" in vars() else locals()
if "ICL_model" not in vars_dict or model_forced_reload:
    ICL_model, ICL_tknz = load_mllm_model_and_data.load_ICL_model(ICL_model_name, hf_token = huggingface_token)


prompts, queries = load_mllm_model_and_data.load_data_from_StaICC_experimentor(benchmark[dataset_index], ICL_selected_token_type)
ICL_hidden_states = mllm_inference.ICL_inference_to_hidden_states(ICL_model, ICL_tknz, prompts)

