from . import normal
from .util import configs, functional, experimentor
import copy
import warnings

class Triplet_bias():
    def __init__(self):
        self.contextual = Contextual_bias()
        self.domain = Domain_bias()
        self.post = Post_bias()
    
    def __call__(
        self, 
        list_of_forward_inference: list[callable], # for each dataset, you should give a forward_inference function. If you just give one, we will expand it to the length of the benchmark.
        return_divided_results = True,
        batched_inference = False
    ):
        return self.auto_run(list_of_forward_inference, return_divided_results, batched_inference)
    
    def auto_run(self, list_of_forward_inference, return_divided_results, batched_inference):
        return {
            "contextual": self.contextual.auto_run(list_of_forward_inference, return_divided_results, batched_inference),
            "domain": self.domain.auto_run(list_of_forward_inference, return_divided_results, batched_inference),
            "post": self.post.auto_run(list_of_forward_inference, return_divided_results, batched_inference)
        }

class Contextual_bias(normal.Normal):
    def __init__(
        self, 
        k = 4,
        noisy_channel = False,
        metrics: dict = {
            "entropy": functional.bias_mean_entropy_metric,
            "distribution": functional.bias_mean_metric,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        self._load_data()
        self.metrics = metrics
        self.noisy_channel = noisy_channel

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)

    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False): 
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "contextual"
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "contextual"
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel, bias_type = "contextual")
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")


class Domain_bias(normal.Normal):
    def __init__(
        self, 
        k = 4,
        noisy_channel = False,
        metrics: dict = {
            "entropy": functional.bias_mean_entropy_metric,
            "distribution": functional.bias_mean_metric,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
        domain_query_length = 128
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        self._load_data()
        self.metrics = metrics
        self.noisy_channel = noisy_channel
        self.domain_query_length = domain_query_length

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)

    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False): 
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "domain",
                        domain_query_length = self.domain_query_length
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "domain",
                        domain_query_length = self.domain_query_length
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel, bias_type = "domain", domain_query_length = self.domain_query_length)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")


class Post_bias(normal.Normal):
    def __init__(
        self, 
        k = 4,
        noisy_channel = False,
        metrics: dict = {
            "DL div.": functional.post_bias_dl_metric,
            "distribution": functional.post_bias_dis_metric,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
        domain_query_length = 128
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        self._load_data()
        self.metrics = metrics
        self.noisy_channel = noisy_channel
        self.domain_query_length = domain_query_length

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)

    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False): 
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.post_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.post_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.post_bias_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")


class GLER(normal.Normal):
    def __init__(
        self, 
        k=4,  
        metrics: dict = {
            "accuracy": functional.accuracy,
            "averaged_truelabel_likelihood": functional.averaged_truelabel_likelihood,
            "macro_F1": functional.macro_F1,
            "expected_calibration_error_1": functional.expected_calibration_error_1
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
        interpolations = 5
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        self._load_data()
        self.metrics = metrics
        self.noisy_channel = False
        self.interpolations = interpolations

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)
    
    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False):
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.GLER_experimentor(
                        original_dataset = data, 
                        k=k, 
                        sensitivity_test = self.interpolations,
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.GLER_experimentor(
                        original_dataset = data, 
                        k=k, 
                        sensitivity_test = self.interpolations,
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.GLER_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel, sensitivity_test = self.interpolations)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")
    
    def auto_run(
        self, 
        list_of_forward_inference: list[callable], # for each dataset, you should give a forward_inference function. If you just give one, we will expand it to the length of the benchmark.
        return_divided_results = True,
        batched_inference = False
    ):
        count = 0
        if type(list_of_forward_inference) != list:
            list_of_forward_inference = [list_of_forward_inference] * len(self.experimentor)
        else:
            if len(list_of_forward_inference) != len(self.experimentor):
                if len(list_of_forward_inference) == 1:
                    list_of_forward_inference = list_of_forward_inference * len(self.experimentor)
                else:
                    raise ValueError("The length of list_of_forward_inference must be the same as the number of datasets in the benchmark. You can use the get_experiment_data method to get the datasets and their order.")
        ret_divided = {}
        ret_sum = {}
        for name, metric in self.metrics.items():
            ret_sum[name] = 0
        for i, single_experimentor in enumerate(self.experimentor):
            count += 1
            print("\n\nExperiment {} in {}".format(count, len(self.experimentor)))
            temp_res, success = single_experimentor(forward_inference = list_of_forward_inference[i], batched_inference = batched_inference)
            ret_divided[single_experimentor.triplet_dataset.dataset_name] = temp_res
            if not success:
                warnings.warn("The experimentor on the dataset " + single_experimentor.triplet_dataset.get_dataset_name() + " failed.")
                continue

        for name, res in ret_divided.items():
            for metric_name, metric in res['sensitivity'].items():
                ret_sum[metric_name] += metric[0]
        for name, metric in ret_sum.items():
            ret_sum[name] = metric / len(ret_divided)

        if return_divided_results:
            return {"Divided results": ret_divided, "Averaged results": ret_sum}
        else:
            return {"Averaged results": ret_sum} 
        

class Template_sens(normal.Normal):
    def __init__(
        self, 
        k=4,  
        metrics: dict = {
            "consistency": functional.consistency,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        self._load_data()
        self.metrics = metrics

        self.re_initialize(k = k)
    
    def re_initialize(self, k: int = 4, keep_prompter = False):
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.template_sensitivity_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], 100],
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.template_sensitivity_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], 100],
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.template_sensitivity_experimentor(original_dataset = data, k=k, metrics=self.metrics)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")
    
    def auto_run(
        self, 
        list_of_forward_inference: list[callable], # for each dataset, you should give a forward_inference function. If you just give one, we will expand it to the length of the benchmark.
        return_divided_results = True,
        batched_inference = False
    ):
        count = 0
        if type(list_of_forward_inference) != list:
            list_of_forward_inference = [list_of_forward_inference] * len(self.experimentor)
        else:
            if len(list_of_forward_inference) != len(self.experimentor):
                if len(list_of_forward_inference) == 1:
                    list_of_forward_inference = list_of_forward_inference * len(self.experimentor)
                else:
                    raise ValueError("The length of list_of_forward_inference must be the same as the number of datasets in the benchmark. You can use the get_experiment_data method to get the datasets and their order.")
        ret_divided = {}
        ret_sum = {}
        for name, metric in self.metrics.items():
            ret_sum[name] = 0
        for i, single_experimentor in enumerate(self.experimentor):
            count += 1
            print("\n\nExperiment {} in {}".format(count, len(self.experimentor)))
            temp_res, success = single_experimentor(forward_inference = list_of_forward_inference[i], batched_inference = batched_inference)
            ret_divided[single_experimentor.triplet_dataset.dataset_name] = temp_res
            if not success:
                warnings.warn("The experimentor on the dataset " + single_experimentor.triplet_dataset.get_dataset_name() + " failed.")
                continue

        for name, res in ret_divided.items():
            for metric_name, metric in res['sensitivity'].items():
                ret_sum[metric_name] += metric
        for name, metric in ret_sum.items():
            ret_sum[name] = metric / len(ret_divided)

        if return_divided_results:
            return {"Divided results": ret_divided, "Averaged results": ret_sum}
        else:
            return {"Averaged results": ret_sum} 
        

class Demo_sens(normal.Normal):
    def __init__(
        self, 
        k=4,  
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        self._load_data()

        self.re_initialize(k = k)
    
    def re_initialize(self, k: int = 4, keep_prompter = False):
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.demonstration_sensitivity_experimentor(
                        original_dataset = data, 
                        k=k, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.demonstration_sensitivity_experimentor(
                        original_dataset = data, 
                        k=k, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.demonstration_sensitivity_experimentor(original_dataset = data, k=k)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")
    
    def auto_run(
        self, 
        list_of_forward_inference: list[callable], # for each dataset, you should give a forward_inference function. If you just give one, we will expand it to the length of the benchmark.
        return_divided_results = True,
        batched_inference = False
    ):
        count = 0
        if type(list_of_forward_inference) != list:
            list_of_forward_inference = [list_of_forward_inference] * len(self.experimentor)
        else:
            if len(list_of_forward_inference) != len(self.experimentor):
                if len(list_of_forward_inference) == 1:
                    list_of_forward_inference = list_of_forward_inference * len(self.experimentor)
                else:
                    raise ValueError("The length of list_of_forward_inference must be the same as the number of datasets in the benchmark. You can use the get_experiment_data method to get the datasets and their order.")
        ret_divided = {}
        ret_sum = {}
        for name, metric in self[0].metrics.items():
            ret_sum[name] = 0
        for i, single_experimentor in enumerate(self.experimentor):
            count += 1
            print("\n\nExperiment {} in {}".format(count, len(self.experimentor)))
            temp_res, success = single_experimentor(forward_inference = list_of_forward_inference[i], batched_inference = batched_inference)
            ret_divided[single_experimentor.triplet_dataset.dataset_name] = temp_res
            if not success:
                warnings.warn("The experimentor on the dataset " + single_experimentor.triplet_dataset.get_dataset_name() + " failed.")
                continue

        for name, res in ret_divided.items():
            for metric_name, metric in res['sensitivity'].items():
                ret_sum[metric_name] += metric
        for name, metric in ret_sum.items():
            ret_sum[name] = metric / len(ret_divided)

        if return_divided_results:
            return {"Divided results": ret_divided, "Averaged results": ret_sum}
        else:
            return {"Averaged results": ret_sum} 