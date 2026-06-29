import torch
from ..util import experimentor
from ..util import functional
from ..util import configs
from ..util import stable_random
from . import model_kernel
import itertools
import numpy as np

def PPL_ICL(
    model: callable,
    tokenizer: callable,
    instruction_set = None,
    experimentor: experimentor.single_experimentor = None,
    cache_empty: callable = torch.cuda.empty_cache, # GPU cache empty function. Can be torch.cuda.empty_cache.
):
    # https://aclanthology.org/2023.findings-emnlp.679
    if instruction_set is None:
        instruction_set = configs.PPL_ICL_INSTRUCTION_SETTINGS[experimentor.triplet_dataset.get_dataset_name()]
    with torch.no_grad():
        LM_losses = []
        for instruction in instruction_set:
            cache_empty()
            tknzd_data = tokenizer(instruction, return_tensors="pt").input_ids.to(model.device)
            LM_losses.append(model(tknzd_data, labels = tknzd_data).loss.detach().to(torch.float).cpu().item())
        min_index = functional.argmin(LM_losses)
        if experimentor is not None:
            experimentor.prompt_former.change_instruction(instruction_set[min_index])
        return instruction_set[min_index]


class SA_ICL():
    # https://aclanthology.org/2023.acl-long.79
    # Only usable on single-sentence classification tasks.
    def __init__(
        self,
        model: callable,
        tokenizer: callable,
        experimentor: experimentor.single_experimentor = None,
        cache_empty: callable = torch.cuda.empty_cache, # GPU cache empty function. Can be torch.cuda.empty_cache.
        calibrate_function: callable = None,
        demonstration_set_cut = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.experimentor = experimentor
        self.cache_empty = cache_empty
        self.demonstration_set = experimentor.demonstration_set()
        if demonstration_set_cut is not None:
            self.demonstration_set = self.demonstration_set.cut_by_index(demonstration_set_cut)
        self.test_set = experimentor.test_set()
        self.label_space = experimentor.get_label_space()
        self.TopK_anchors = []
        self.calibrate_function = calibrate_function
        self.demonstration_set_cut = demonstration_set_cut

        self._encode_demonstrations()
        
        fake_demonstration_sampler = [[0] * self.experimentor.get_k() for _ in range(len(self.test_set))]
        self.experimentor.set_demonstration_sampler(
            fake_demonstration_sampler
        )
    
    def _encode_demonstrations(self):
        count = 0
        for demo in self.demonstration_set:
            if len(demo[0]) == 0:
                continue
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((count + 1) / len(self.demonstration_set) * 100), 
                (count + 1), 
                len(self.demonstration_set)
            ), ">>" * int((count + 1) / len(self.demonstration_set) * 32), end="")
            try:
                count += 1
                self.TopK_anchors.append(
                    model_kernel.standard_ICL_inference_with_torch_Causal_LM(
                        prompt = demo[0], 
                        model = self.model, 
                        tokenizer = self.tokenizer, 
                        label_space = self.label_space, 
                        cache_empty = self.cache_empty,
                        calibration_function = None,
                        return_hidden_state = True
                    )[-1]
                )
            except:
                continue
        self.TopK_anchors = np.array(self.TopK_anchors)
    
    def _get_top_k_indexes(self, input, k):
        distance = []
        input_encoded = model_kernel.standard_ICL_inference_with_torch_Causal_LM(
            prompt = input, 
            model = self.model, 
            tokenizer = self.tokenizer, 
            label_space = self.label_space, 
            cache_empty = self.cache_empty,
            calibration_function = None,
            return_hidden_state = True
        )[-1]
        for anchor in self.TopK_anchors:
            distance.append(np.linalg.norm(input_encoded - np.array(anchor)))
        ret = []
        for _ in range(k):
            ret.append(functional.argmin(distance))
            distance[functional.argmin(distance)] = 1e10
        return ret

    def set_TopK_to_demonstration(self, k):
        # https://arxiv.org/abs/2101.06804
        demonstration_sampler = []
        for i in range(len(self.test_set)):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / len(self.test_set) * 100), 
                (i + 1), 
                len(self.test_set)
            ), ">>" * int((i + 1) / len(self.test_set) * 32), end="")
            demonstration_sampler.append(self._get_top_k_indexes(self.test_set[i][0], k))
        self.experimentor.set_demonstration_sampler(demonstration_sampler)
    
    def _enum_orders_given_indexes_set(self, indexes_set, each_length, return_length):
        ret = list(itertools.permutations(indexes_set, each_length))
        random = stable_random.stable_random()
        random.shuffle_list(ret)
        return ret[:return_length]
    
    def _inference(self, query_index, nearest_k = 5, inference_demos_number = 4, return_length = 10):
        query = self.test_set[query_index][0]
        top_k_indexes = self._get_top_k_indexes(query, nearest_k)
        orders = self._enum_orders_given_indexes_set(top_k_indexes, inference_demos_number, return_length)
        result_candidates = []
        entropies = []
        count = 0
        for order in orders:
            demo_data_lines = []
            for index in order:
                demo_data_lines.append(self.demonstration_set[index])
            prompt = self.experimentor.prompt_former.write_prompt_from_dataline(demo_data_lines, query)
            result_candidates.append(model_kernel.standard_ICL_inference_with_torch_Causal_LM(
                prompt = prompt, 
                model = self.model, 
                tokenizer = self.tokenizer, 
                label_space = self.label_space, 
                cache_empty = self.cache_empty,
                calibration_function = self.calibrate_function
            ))
            entropies.append(functional.entropy(result_candidates[-1]))
        return result_candidates[functional.argmin(entropies)]
    
    def inference_interface(self, prompt, label_space, nearest_k = 5, inference_demos_number = 4, return_length = 10):
        # should be used in the batched mode in the experimentor.
        ret = []
        for i in range(len(prompt)):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / len(self.test_set) * 100), 
                (i + 1), 
                len(self.test_set)
            ), ">>" * int((i + 1) / len(self.test_set) * 32), end="")
            ret.append(self._inference(i, nearest_k, inference_demos_number, return_length))
        return ret