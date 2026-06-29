from . import configs, functional, stable_random, mllm_dataset_interface
import copy
import warnings
import functools

class single_experimentor():
    """
        The main experimentor for this toolkit.
        The single_experimentor is designed to test the forward inference function on the triplet dataset.
        The forward inference function should be a callable that takes a prompt and returns a list of label logits or a label index.
        Main members:
            _triplet_dataset: mllm_dataset_interface.triplet_dataset; The dataset for the experiment.
                - You should'n access this member directly. Use the methods to access the dataset.
            prompt_former: mllm_dataset_interface.prompt_writter; The prompt former for the experiment.
                - You can use the methods in this member to design the prompt templates.
            demonstration_sampler: mllm_dataset_interface.demonstration_sampler; The demonstration sampler for the experiment.
                - You can reload it into a list-shaped list of integers to sample the demonstrations for each test sample.
                - The reloading list shape: [test_sample_index][demonstration_sample_index (k)]. e.g. [[1, 2, 3], [4, 7, 5], [7, 1, 2]] for k=3, test_sample_number=3.
            _k: The number of demonstrations for each test sample.
                - Should only be set in initialization.
                - You shouldn't set this member after initialization.
            _repeat_times: The number of times the test samples will be tested.
            metrics: The metrics for the experiment.
                - You can add or remove metrics by changing this member.
                - The metrics should be a dictionary with the format: {metric_name: metric_function}.
        Main methods:
            __init__: The initialization method.
                - triplet_dataset: mllm_dataset_interface.triplet_dataset; The dataset for the experiment.
                - original_dataset: mllm_dataset_interface.original_dataset; The original dataset for the experiment.
              >> You should provide at least and only one dataset.
                - k: int; The number of demonstrations for each test sample.
                - metrics: dict; The metrics for the experiment.
                - repeat_times: int; The number of times the test samples will be tested. Default: 2.
                - dividing: list[int]; The dividing list for the triplet dataset. 
                - noisy_channel: bool; If True, the prompt will be given as a noisy channel form (label, input; label, input; ...).
            __call__: The callable method for the experiment.
                - forward_inference: callable; The forward inference function.
                - batched_inference: bool; If True, the forward_inference function should be a function that takes a list of prompts and returns a list of logits for each label.
            reset_demonstration_sampler: Reset the demonstration sampler to the default state. The anti-operation of set_demonstration_sampler.
            get_prompt_writter_from_dataline: Get the function prompt_former.write_prompt_from_dataline.
            get_label_space: Get the label space of the dataset.
            set_demonstration_sampler: Set the demonstration sampler to a fixed list-shaped sampler.
                - sampler: list[list[int]]; The sampler for the demonstrations.
            auto_run: The main method for the experiment. == __call__.
                - forward_inference: callable; The forward inference function defined by user. See the prefabricate_inference library as examples. 
                    (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. OR (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>.
                - batched_inference: bool; If True, the forward_inference function should be a function that takes a list of prompts and returns a list of logits for each label.
            calibration_set: Get the calibration set of the dataset.
            demonstration_set: Get the demonstration set of the dataset.
            test_set: Get the test set of the dataset.
    """
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        metrics: dict = {
            "accuracy": functional.accuracy,
            "averaged_truelabel_likelihood": functional.averaged_truelabel_likelihood,
            "macro_F1": functional.macro_F1,
            "expected_calibration_error_1": functional.expected_calibration_error_1
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = configs.STANDARD_SETTINGS["test_times_for_each_test_sample"],
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]], # A list of integers that divides the test samples into 3 splits. The first split will be used for calibration, the second split will be used for demonstration, and the third split will be used for testing. Only can be used when original_dataset is given.
        noisy_channel = False # If True, the demonstration set will be generated by a noisy channel model.
    ):
        if k < 0:
            raise ValueError("k should be a positive integer.")
        if repeat_times < 0:
            raise ValueError("repeat_times should be a positive integer.")

        if repeat_times != configs.STANDARD_SETTINGS["test_times_for_each_test_sample"]:
            warnings.warn(configs.WARNING_SETTINGS["tampering"])
        
        if triplet_dataset is None and original_dataset is None:
            raise ValueError("You should provide at least one dataset.")
        if triplet_dataset is not None and original_dataset is not None:
            raise ValueError("You should provide only one dataset.")
        
        self._k = k
        if triplet_dataset is not None:
            self.triplet_dataset = triplet_dataset
        if original_dataset is not None:
            self.triplet_dataset = mllm_dataset_interface.triplet_dataset(original_dataset, dividing[0], dividing[1], dividing[2])
        self.dataset_name = original_dataset.dataset_name
        self.prompt_former = mllm_dataset_interface.prompt_writter(self.triplet_dataset, noisy_channel)
        demo_dataset = self.triplet_dataset.demonstration
        test_dataset = self.triplet_dataset.test

        if  len(demo_dataset[0]) > 3:
            has_task_type =  True
        else:
            has_task_type= False
        if has_task_type:
            demo_tasks = [ex[-1] for ex in demo_dataset]
            query_tasks = [
                ex[-1]
                for _ in range(repeat_times)
                for ex in test_dataset
            ]
            self.demonstration_sampler = mllm_dataset_interface.demonstration_sampler_v2(self._k, demonstration_set_size=len(self.triplet_dataset.demonstration), query_numbers=repeat_times * len(self.triplet_dataset.test), query_tasks= query_tasks, demo_tasks=demo_tasks)
        else:
            self.demonstration_sampler = mllm_dataset_interface.demonstration_sampler(self._k, len(self.triplet_dataset.demonstration),repeat_times * len(self.triplet_dataset.test))
        
        self._default_demonstration_sampler = copy.deepcopy(self.demonstration_sampler)
        self._default_repeat_times = repeat_times
        self._repeat_times = repeat_times
        self.metrics = metrics
        self.label_dis = [0] * len(self.triplet_dataset.get_label_space())
        self.predictions = []
        self.bias_type = "None"
    
    def __len__(self):
        return len(self.triplet_dataset.test) * self._repeat_times

    def __call__(self, forward_inference: callable = None, input_prediction = None, batched_inference = False, return_outputs = False):
        return self.auto_run(forward_inference, preentered_prediction = input_prediction, batched_inference = batched_inference, return_outputs = return_outputs)
    
    def __str__(self) -> str:
        ret = ("--- single experimentor ---\n" +
        "\nTriplet Dataset: " + str(self.triplet_dataset) +
        "\nPrompt Former: " + str(self.prompt_former) +
        "\nDemonstration Sampler: " + str(self.demonstration_sampler) +
        "\nK: " + str(self._k) +
        "\nMetrics: " + str(self.metrics) +
        "\nSamples for each test sample: " + str(self._repeat_times))
        return ret

    def __repr__(self) -> str:
        return self.__str__()

    def _get_prompts_for_test_sample(self, test_sample_index: int, repeat_time: int):
        # repeat_time_from_0
        index = test_sample_index + repeat_time * len(self.triplet_dataset.test)
        demos_indexes = self.demonstration_sampler[index]
        if len(demos_indexes) != self._k:
            warnings.warn("The length of the demonstration indexes should be equal to k, in test index: " + str(index))
        return self.prompt_former.write_prompt(demos_indexes, test_sample_index) ## 返回的是字典


    def _get_prompts_dir_for_test_sample(self, test_sample_index: int, repeat_time: int):
        # repeat_time_from_0
        index = test_sample_index + repeat_time * len(self.triplet_dataset.test)
        demos_indexes = self.demonstration_sampler[index]
        if len(demos_indexes) != self._k:
            warnings.warn("The length of the demonstration indexes should be equal to k, in test index: " + str(index))
        return self.prompt_former.get_prompt_dir(demos_indexes, test_sample_index) ## 返回的是字典


    def add_metric(self, metric_name: str, metric_function: callable):
        if metric_name in self.metrics:
            warnings.warn("The metric name already exists. Overwriting the metric function.")
        self.metrics[metric_name] = metric_function

    def set_k(self, k: int):
        self.reset_demonstration_sampler()
        self._k = k
        self.demonstration_sampler = mllm_dataset_interface.demonstration_sampler(self._k, len(self.triplet_dataset.demonstration), self._repeat_times * len(self.triplet_dataset.test))
        self._default_demonstration_sampler = copy.deepcopy(self.demonstration_sampler)

    def get_k(self):
        return self._k
    
    def get_repeat_times(self):
        return self._repeat_times
    
    def set_out_of_domain_mode(self):
        self.reset_demonstration_sampler()
        # wash the demonstration sampler
        wash_list = []
        for i in range(len(self.triplet_dataset.test) * self._repeat_times):
            label = self.triplet_dataset.get_default_ground_truth_label_index(i % len(self.triplet_dataset.test))
            demonstration_indexes = self.demonstration_sampler[i]
            for index in demonstration_indexes:
                if self.triplet_dataset.demonstration.find_index_from_label(self.triplet_dataset.demonstration[index][1]) == label:
                    wash_list.append(i)
                    break

        for i in wash_list:
            label = self.triplet_dataset.get_default_ground_truth_label_index(i % len(self.triplet_dataset.test))
            success = False
            while True:
                success = True
                new_sample = self.demonstration_sampler._get_next_sample()
                for index in new_sample:
                    if self.triplet_dataset.demonstration.find_index_from_label(self.triplet_dataset.demonstration[index][1]) == label:
                        success = False
                        break
                if success:
                    break
            self.demonstration_sampler._set_sample(i, new_sample)

    def set_in_domain_mode(self):
        self.reset_demonstration_sampler()
        # wash the demonstration sampler
        wash_list = []
        for i in range(len(self.triplet_dataset.test) * self._repeat_times):
            label = self.triplet_dataset.get_default_ground_truth_label_index(i % len(self.triplet_dataset.test))
            demonstration_indexes = self.demonstration_sampler[i]
            label_exist = False
            for index in demonstration_indexes:
                if self.triplet_dataset.demonstration.find_index_from_label(self.triplet_dataset.demonstration[index][1]) == label:
                    label_exist = True
                    break
            if not label_exist:
                wash_list.append(i)

        for i in wash_list:
            label = self.triplet_dataset.get_default_ground_truth_label_index(i % len(self.triplet_dataset.test))
            success = False
            while True:
                success = False
                new_sample = self.demonstration_sampler._get_next_sample()
                for index in new_sample:
                    if self.triplet_dataset.demonstration.find_index_from_label(self.triplet_dataset.demonstration[index][1]) == label:
                        success = True
                        break
                if success:
                    break
            self.demonstration_sampler._set_sample(i, new_sample)

    def reset_demonstration_sampler(self):
        self.demonstration_sampler = copy.deepcopy(self._default_demonstration_sampler)
        self._repeat_times = self._default_repeat_times

    def get_prompt_writter_from_dataline(self):
        return self.prompt_former.write_prompt_from_dataline

    def get_label_space(self):
        return copy.deepcopy(self.prompt_former.get_label_space())
    
    def set_demonstration_sampler(self, sampler):
        # The sampler can be a list-shaped list of integers. 
        # The self._default_repeat_times will be set to 1 since no repeat time is needed with a fixed sampler.
        # The demonstrations will be sampled as: sampler[test_sample_index].
        # For example: when the sampler is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], the demonstrations for the first test sample will be the [1, 2, 3]-th samples in the demonstration set and so on.
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if len(sampler) != len(self.triplet_dataset.test):
            raise ValueError("The length of the sampler should be equal to the number of the test samples.")
        if all([len(x) != self._k for x in sampler]):
            raise ValueError("The length of each sample in the sampler should be equal to k.")
        self.demonstration_sampler = sampler
        self._repeat_times = 1
    
    def prompt_set(self):
        ret = []
        for time in range(self._repeat_times):
            for index in range(len(self.triplet_dataset.test)):
                prompt = self._get_prompts_for_test_sample(index, time)
                ret.append(prompt)
        return ret

    def prompt_dir(self):
        ret = []
        for time in range(self._repeat_times):
            for index in range(len(self.triplet_dataset.test)):
                prompt = self._get_prompts_dir_for_test_sample(index, time)
                ret.append(prompt)
        return ret


    def auto_run(
        self, 
        forward_inference: callable = None, 
            # When batched_inference is disabled, forward_inference: (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. The inputted parameter signs are fixed to prompt and label_space.
            # When batched_inference is enabled, forward_inference: (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>.
        preentered_prediction = None, 
            # The prediction for the test set (list[int]). If None, the prediction will be calculated by the forward_inference.
        batched_inference = False, 
            # for batched inference like BatchCalibration.
            # If enabled, we will input all the prompts into the forward_inference; and if disabled, we will input the prompt into the forward_inference one by one
        return_outputs = False, 
            # If True, the outputs will be returned.
        _previous_prediction = None 
            # If you need to connect multiple inference results, please set it to the previous prediction.
    ):
        # The forward_inference function should be a callable that takes a prompt and returns a list of label logits or a label index.
        # We encourage the forward_inference function to be a function that takes a prompt and returns a list of logits for each label, so that we can calculate more metrics.
        # >> If you use a function that returns a label index, the metrics that require logits will be calculated as if the logits are one-hot encoded.
        success = False
        
        if self._k > len(self.triplet_dataset.demonstration):
            warnings.warn("The k value is larger than the length of the demonstration dataset. Return all-0 results.")
            ret = {}
            for metric_name, metric_function in self.metrics.items():
                ret[metric_name] = 0
            return ret, success
        ground_truth = []
        prediction = []
        total_samples = len(self.triplet_dataset.test) * self._repeat_times

        # INFERENCE
        if preentered_prediction is None and forward_inference is not None:
            print("\nStart testing the forward inference function " + str(forward_inference) + " on the dataset: " + str(self.triplet_dataset.test.dataset_name) + " with bias type: " + self.bias_type + ".\n")
            if not batched_inference:
                # Iterative inference: forward_inference: (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. Inferring one by one.
                for time in range(self._repeat_times):
                    for index in range(len(self.triplet_dataset.test)):
                        prompt = self._get_prompts_for_test_sample(index, time)
                        result = forward_inference(prompt = prompt, label_space = self.prompt_former.get_label_space()) # The inputted parameter signs are fixed to prompt and label_space.
                        ground_truth.append(self.triplet_dataset.get_default_ground_truth_label_index(index))
                        self.label_dis[ground_truth[-1]] += 1
                        prediction.append(result)
                        print("\r", end="")
                        print("Process: {}%, {} in {}".format(
                            int((index + time * len(self.triplet_dataset.test) + 1) / total_samples * 100), 
                            (index + time * len(self.triplet_dataset.test) + 1), 
                            total_samples
                        ), ">>" * int((index + time * len(self.triplet_dataset.test)) / total_samples * 32), end="")
            else:
                # Batched inference: forward_inference: (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>. Inferring all at once
                prompts = []
                for time in range(self._repeat_times):
                    for index in range(len(self.triplet_dataset.test)):
                        prompts.append(self._get_prompts_for_test_sample(index, time))
                        ground_truth.append(self.triplet_dataset.get_default_ground_truth_label_index(index))
                        self.label_dis[ground_truth[-1]] += 1
                prediction = forward_inference(prompt = prompts, label_space = self.prompt_former.get_label_space())
        elif preentered_prediction is not None:
            for time in range(self._repeat_times):
                for index in range(len(self.triplet_dataset.test)):
                    ground_truth.append(self.triplet_dataset.get_default_ground_truth_label_index(index))
                    self.label_dis[ground_truth[-1]] += 1
            prediction = preentered_prediction
        else:
            raise ValueError("You should provide either the forward_inference function or the input_prediction.")

        # TEST
        ret = {}
        for metric_name, metric_function in self.metrics.items():
            self.predictions = functional.extend_onehot_prediction_to_logits(prediction)
            if _previous_prediction is not None:
                print("\nPrevious prediction is given. Connecting the previous prediction with length " + str(len(_previous_prediction)) + ".\n")
                self.predictions = _previous_prediction + self.predictions
  
            ret[metric_name] = metric_function(ground_truth, self.predictions)
        success = True
        if return_outputs:
            return ret, success, {"groundtruth": ground_truth, "predicted": functional.compress_logits_prediction_to_onehot(prediction), "prob.": self.predictions}
        return ret, success
    
    def calibration_set(self):
        return self.triplet_dataset.calibration
    
    def demonstration_set(self):
        return self.triplet_dataset.demonstration
    
    def test_set(self):
        # print(self.triplet_dataset.test)
        # Elements: (['A man and a girl are climbing a tree.'], '/home/yuw/ICL_Circuit/data/snli/images/417839679.jpg', 'contradiction')
        return self.triplet_dataset.test


class prior_bias_experimentor(single_experimentor):
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        bias_type: str = "contextual", # "contextual" or "domain"
        domain_query_length: int = 128,
        metrics: dict = {
            "entropy": functional.bias_mean_entropy_metric,
            "distribution": functional.bias_mean_metric,
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = configs.STANDARD_SETTINGS["test_times_for_each_test_sample"],
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]], # A list of integers that divides the test samples into 3 splits. The first split will be used for calibration, the second split will be used for demonstration, and the third split will be used for testing. Only can be used when original_dataset is given.
        noisy_channel = False # If True, the demonstration set will be generated by a noisy channel model.
    ):
        if bias_type not in ["contextual", "domain"]:
            raise ValueError("bias_type should be 'contextual' or 'domain'.")
        super().__init__(
            triplet_dataset = triplet_dataset, 
            original_dataset = original_dataset, 
            k = k, 
            metrics = metrics, 
            repeat_times = repeat_times, 
            dividing = dividing, 
            noisy_channel = noisy_channel
        )
        self.bias_type = bias_type
        if bias_type == "contextual":
            self.meanless_query = self._gen_empty_query()
        elif bias_type == "domain":
            self.meanless_query = self._gen_domain_query(self.triplet_dataset.demonstration, domain_query_length)
        self.prompt_former = mllm_dataset_interface.prompt_writter(
            triplet_dataset = self.triplet_dataset, 
            use_noisy_channel = noisy_channel, 
            pseudo_query_generater = self.meanless_query
        )

    def _gen_empty_query(self):
        while True:
            yield ["" for _ in range(len(self.triplet_dataset.calibration[0][0]))]
    
    def _gen_domain_query(self, sample_set, sample_length):
        my_random = stable_random.stable_random()
        while True:
            ret = []
            for i in range(len(sample_set[0][0])):
                output = []
                while len(output) < sample_length:
                    random_sample = sample_set[my_random.get_int_from_range(0, len(sample_set) - 1)][0][i]
                    random_sample = random_sample.split(' ')
                    random_index = my_random.get_int_from_range(0, len(random_sample) - 1)
                    output.append(random_sample[random_index])
                output = ' '.join(output)
                ret.append(output)
            yield ret


class post_bias_experimentor(single_experimentor):
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        metrics: dict = {
            "DL div.": functional.post_bias_dl_metric,
            "distribution": functional.post_bias_dis_metric,
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = configs.STANDARD_SETTINGS["test_times_for_each_test_sample"],
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]], # A list of integers that divides the test samples into 3 splits. The first split will be used for calibration, the second split will be used for demonstration, and the third split will be used for testing. Only can be used when original_dataset is given.
        noisy_channel = False # If True, the demonstration set will be generated by a noisy channel model.
    ):
        super().__init__(triplet_dataset, original_dataset, k, metrics, repeat_times, dividing, noisy_channel)
        self.bias_type = "post"


class sensitivity_experimentor(single_experimentor):
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        sensitivity_test = 5,
        metrics: dict = {
            "accuracy": functional.accuracy,
            "averaged_truelabel_likelihood": functional.averaged_truelabel_likelihood,
            "macro_F1": functional.macro_F1,
            "expected_calibration_error_1": functional.expected_calibration_error_1
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = configs.STANDARD_SETTINGS["test_times_for_each_test_sample"],
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]], # A list of integers that divides the test samples into 4 splits. The first split will be used for calibration, the second split will be used for demonstration, the third split will be used for testing, and the fourth split will be used for sensitivity test. Only can be used when original_dataset is given.
        noisy_channel = False # If True, the demonstration set will be generated by a noisy channel model.
    ):
        super().__init__(
            triplet_dataset = triplet_dataset, 
            original_dataset = original_dataset, 
            k = k, 
            metrics = metrics, 
            repeat_times = repeat_times, 
            dividing = dividing, 
            noisy_channel = noisy_channel
        )
        self.test_times = sensitivity_test
    
    def _sensitivity_init(self):
        pass

    def _sensitivity_step(self):
        pass

    def inference_run(self, forward_inference: callable, batched_inference=False, _previous_prediction = False):
        result_dicts = []
        self._sensitivity_init()
        for i in range(self.test_times):
            if _previous_prediction:
                result_dicts.append(super().auto_run(forward_inference = forward_inference, batched_inference = batched_inference, _previous_prediction = copy.deepcopy(self.predictions))[0])
            else:
                result_dicts.append(super().auto_run(forward_inference = forward_inference, batched_inference = batched_inference, _previous_prediction = None)[0])
            if i != self.test_times - 1:
                self._sensitivity_step()
        return result_dicts


class GLER_experimentor(sensitivity_experimentor):
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        sensitivity_test = 5,
        metrics: dict = {
            "accuracy": functional.accuracy,
            "averaged_truelabel_likelihood": functional.averaged_truelabel_likelihood,
            "macro_F1": functional.macro_F1,
            "expected_calibration_error_1": functional.expected_calibration_error_1
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = configs.STANDARD_SETTINGS["test_times_for_each_test_sample"],
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]], # A list of integers that divides the test samples into 4 splits. The first split will be used for calibration, the second split will be used for demonstration, the third split will be used for testing, and the fourth split will be used for sensitivity test. Only can be used when original_dataset is given.
        noisy_channel = False # If True, the demonstration set will be generated by a noisy channel model.
    ):
        super().__init__(
            triplet_dataset = triplet_dataset, 
            original_dataset = original_dataset, 
            k = k, 
            sensitivity_test = sensitivity_test, 
            metrics = metrics, 
            repeat_times = repeat_times, 
            dividing = dividing, 
            noisy_channel = noisy_channel
        )
        if sensitivity_test < 2:
            raise ValueError("The sensitivity test should be larger than 1.")
        self.current_error_rate = 0

    def _sensitivity_init(self):
        self.current_error_rate = 0
        self.prompt_former.set_label_wrong_rate(self.current_error_rate)

    def _sensitivity_step(self):
        self.current_error_rate += 1 / (self.test_times - 1)
        self.prompt_former.set_label_wrong_rate(self.current_error_rate)
    
    def auto_run(
        self, 
        forward_inference: callable, 
            # When batched_inference is disabled, forward_inference: (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. The inputted parameter signs are fixed to prompt and label_space.
            # When batched_inference is enabled, forward_inference: (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>.
        input_prediction = None, # Unused
        batched_inference = False, # for batched inference like BatchCalibration.
            # If enabled, we will input all the prompts into the forward_inference; and if disabled, we will input the prompt into the forward_inference one by one
        return_outputs = False, # Unused
        preentered_prediction = None, # Unused
        _previous_prediction = None # Unused
    ):
        result_dicts = {}
        sensitivity_dict = {}
        intermidiate_results = self.inference_run(forward_inference = forward_inference, batched_inference = batched_inference)
        arguments = [1]
        for i in range(1, self.test_times):
            arguments.append(arguments[-1] - i / (self.test_times - 1))
        for i in range(self.test_times):
            result_dicts[str(i)] = intermidiate_results[i]
        for metric_name, metric_function in self.metrics.items():
            temp_results = []
            for i in range(len(intermidiate_results)):
                res = intermidiate_results[i]
                temp_results.append(res[metric_name])
            sensitivity_dict[metric_name] = functional.linear_regression(arguments, temp_results)
        result_dicts["sensitivity"] = sensitivity_dict
        return result_dicts, True
    

class template_sensitivity_experimentor(sensitivity_experimentor):
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        metrics: dict = {
            "consistency": functional.consistency,
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = 1,
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], 100], # A list of integers that divides the test samples into 4 splits. The first split will be used for calibration, the second split will be used for demonstration, the third split will be used for testing, and the fourth split will be used for sensitivity test. Only can be used when original_dataset is given.
        orthogonal_table = configs.STANDARD_SETTINGS["L9(3,4)_orthogonal_table"]
    ):
        super().__init__(
            triplet_dataset = triplet_dataset, 
            original_dataset=original_dataset, 
            k=k, metrics=metrics, 
            repeat_times=repeat_times, 
            dividing=dividing, 
            sensitivity_test=len(orthogonal_table)
        )
        self.current_try = 0
        self.orthogonal_table = orthogonal_table
        self.alternate_template = self.triplet_dataset.get_alternate_template()

    def _set_template(self, try_index):
        setting_index = self.orthogonal_table[try_index]
        dictionary = {}
        for i, key in enumerate(self.alternate_template.keys()):
            dictionary[key] = self.alternate_template[key][setting_index[i]]
        self.prompt_former.set_config_dict(dictionary)

    def _sensitivity_init(self):
        self.current_try = 0
        self._set_template(self.current_try)
    
    def _sensitivity_step(self):
        self.current_try += 1
        self._set_template(self.current_try)
    
    def auto_run(self, 
        forward_inference: callable, 
            # When batched_inference is disabled, forward_inference: (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. The inputted parameter signs are fixed to prompt and label_space.
            # When batched_inference is enabled, forward_inference: (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>.
        input_prediction = None, # Unused
        batched_inference = False, # for batched inference like BatchCalibration.
            # If enabled, we will input all the prompts into the forward_inference; and if disabled, we will input the prompt into the forward_inference one by one
        return_outputs = False, # Unused
        preentered_prediction = None, # Unused
        _previous_prediction = None # Unused
    ):
        result_dicts = {}
        intermidiate_results = self.inference_run(forward_inference, batched_inference, _previous_prediction=True)
        result_dicts["sensitivity"] = intermidiate_results[-1]
        return result_dicts, True
    

class demonstration_sensitivity_experimentor(sensitivity_experimentor):
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        metrics: dict = {
            "consistency": functools.partial(functional.consistency, loop_length = 128),
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = 1,
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]], # A list of integers that divides the test samples into 4 splits. The first split will be used for calibration, the second split will be used for demonstration, the third split will be used for testing, and the fourth split will be used for sensitivity test. Only can be used when original_dataset is given.
        sensitivity_test = 8
    ):
        inner_repeat_times = 2
        super().__init__(triplet_dataset = triplet_dataset, original_dataset=original_dataset, k=k, metrics=metrics, repeat_times=inner_repeat_times, dividing=dividing, sensitivity_test=1)
        self.triplet_dataset.test.cut_by_index(len(self.triplet_dataset.test) * inner_repeat_times // sensitivity_test)
        self._repeat_times = sensitivity_test

    def _sensitivity_init(self):
        return
    
    def _sensitivity_step(self):
        return
    
    def auto_run(self, 
        forward_inference: callable, 
            # When batched_inference is disabled, forward_inference: (prompt: str, label_space: list[str]) -> list[float] <logits> or int <label>. The inputted parameter signs are fixed to prompt and label_space.
            # When batched_inference is enabled, forward_inference: (prompts: list[str], label_space: list[str]) -> list[list[float]] <logits> or list[int] <label>.
        input_prediction = None, # Unused
        batched_inference = False, # for batched inference like BatchCalibration.
            # If enabled, we will input all the prompts into the forward_inference; and if disabled, we will input the prompt into the forward_inference one by one
        return_outputs = False, # Unused
        preentered_prediction = None, # Unused
        _previous_prediction = None # Unused
    ):
        result_dicts = {}
        intermidiate_results = self.inference_run(forward_inference, batched_inference, _previous_prediction=False)
        result_dicts["sensitivity"] = intermidiate_results[-1]
        return result_dicts, True