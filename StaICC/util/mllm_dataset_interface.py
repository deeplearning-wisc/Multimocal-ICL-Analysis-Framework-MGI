# Adapt basic_datasets_loader into ICL usable shape.
from . import stable_random
from . import mllm_dataset_loader
from . import configs
import warnings
import copy

class triplet_dataset():
    """
        Split the dataset into three parts: calibration, demonstration, and test.
        Notice: this class should not be accessed by the user for inference. Use the methods in the `single_experimentor` class instead.
        Main members:
            triplet_dataset.calibration: mllm_dataset_loader.basic_datasets_loader; the calibration dataset loader.
            triplet_dataset.demonstration: mllm_dataset_loader.basic_datasets_loader; the demonstration dataset loader.
            triplet_dataset.test: mllm_dataset_loader.basic_datasets_loader; the test dataset loader.
            triplet_dataset.dataset_name: str; the name of the dataset.
        Main methods:
            __init__:
                - original_dataset_loader: mllm_dataset_loader.basic_datasets_loader; the dataset loader.
                - calibration_number: int; the number of calibration samples.
                - demonstration_number: int; the number of demonstration samples.
                - test_number: int; the number of test samples.
                - random_seed: int; the random seed for split the original_dataset_loader.
            - get:
                triplet_dataset.get_dataset_name(): str; get the name of the dataset.
                triplet_dataset.get_label_space(): list[str]; get the label space of the demonstration set.
                triplet_dataset.get_default_ground_truth_label(index: int): str; get the default ground truth label word of the `index`-th examples defined by the `mllm_dataset_loader.basic_datasets_loader` of the test set.
                triplet_dataset.get_default_ground_truth_label_index(index: int): int; get the index of the default ground truth label index of the `index`-th examples defined by the `mllm_dataset_loader.basic_datasets_loader` of the test set.
            - change: We don't recommend to use these methods. Use the same methods in the `prompt_writter` class instead.
                triplet_dataset.change_label_space_triple(label_space: list[str]): None; change the label space of the calibration, demonstration, and test set into the `label_space`.
                triplet_dataset.change_instruction_triple(instruction: str): None; change the instruction of the calibration, demonstration, and test set into the `instruction`.
                triplet_dataset.change_input_text_prefixes_triple(input_text_prefixes: list[str]): None; change the input text prefixes of the calibration, demonstration, and test set into the `input_text_prefixes`.
                triplet_dataset.change_input_text_affixes_triple(input_text_affixes: list[str]): None; change the input text affixes of the calibration, demonstration, and test set into the `input_text_affixes`.
                triplet_dataset.change_label_prefix_triple(label_prefix: str): None; change the label prefix of the calibration, demonstration, and test set into the `label_prefix`.
                triplet_dataset.change_label_affix_triple(label_affix: str): None; change the label affix of the calibration, demonstration, and test set into the `label_affix`.
                triplet_dataset.change_query_prefix_triple(query_prefix: str): None; change the query prefix of the calibration, demonstration, and test set into the `query_prefix`.
                triplet_dataset.change_label_space_triple(label_space: list[str]): None; change the label space of the calibration, demonstration, and test set into the `label_space`.
    """

    def __init__(self, 
        original_dataset_loader: mllm_dataset_loader.basic_datasets_loader, 
        calibration_number = configs.STANDARD_SETTINGS["calibration_number"], 
        demonstration_number = configs.STANDARD_SETTINGS["demonstration_number"], 
        test_number = configs.STANDARD_SETTINGS["test_number"],
        random_seed = configs.STANDARD_SETTINGS["random_seed"]
    ):
        if random_seed != configs.STANDARD_SETTINGS["random_seed"]:
            warnings.warn(configs.WARNING_SETTINGS["tampering"])
        self._split_number_check(original_dataset_loader.get_dataset_name(), calibration_number, demonstration_number, test_number)
        unsplited_dataset = original_dataset_loader
        if len(unsplited_dataset) < calibration_number + demonstration_number + test_number:
            raise ValueError("The dataset {} is too small ({}) to split ({}).".format(unsplited_dataset.get_dataset_name(), len(unsplited_dataset), calibration_number + demonstration_number + test_number))
        my_random = stable_random.stable_random(random_seed)
        indexes = my_random.sample_index_set(calibration_number + demonstration_number + test_number, len(unsplited_dataset))
        self.calibration, self.demonstration, self.test = original_dataset_loader.split([indexes[:calibration_number], indexes[calibration_number:calibration_number+demonstration_number], indexes[calibration_number+demonstration_number:]])
        self.dataset_name = original_dataset_loader.get_dataset_name()
        self.calibration.rename_dataset(original_dataset_loader.get_dataset_name()+"-calibration")
        self.demonstration.rename_dataset(original_dataset_loader.get_dataset_name()+"-demonstration")
        self.test.rename_dataset(original_dataset_loader.get_dataset_name()+"-test")
        self.alternate_template = original_dataset_loader.get_alternate_template()
    
    def __str__(self) -> str:
        return ("Calibration set: \n" + self.calibration.__str__() + "\nDemonstration set: \n" + self.demonstration.__str__() + "\nTest set: \n" + self.test.__str__())
    
    def __repr__(self) -> str:
        ret = "--- Triplet Dataset ---" + "\n"
        ret += "Calibration set: \n" + self.calibration.__repr__() + "\nDemonstration set: \n" + self.demonstration.__repr__() + "\nTest set: \n" + self.test.__repr__()
        return ret
    
    def _split_number_check(self, dataset_name, calibration_number, demonstration_number, test_number):
        # Check if the split numbers are the default numbers.
        if dataset_name == "financial_phrasebank":
            if calibration_number != configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"] or demonstration_number != configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"] or test_number != configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]:
                warnings.warn(configs.WARNING_SETTINGS["tampering"])
        elif dataset_name == "tweet_eval_emotion":
            if calibration_number != configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"] or demonstration_number != configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"] or test_number != configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]:
                warnings.warn(configs.WARNING_SETTINGS["tampering"])
        else:
            if calibration_number != configs.STANDARD_SETTINGS["calibration_number"] or demonstration_number != configs.STANDARD_SETTINGS["demonstration_number"] or test_number != configs.STANDARD_SETTINGS["test_number"]:
                warnings.warn(configs.WARNING_SETTINGS["tampering"])
    
    def get_alternate_template(self):
        return copy.deepcopy(self.alternate_template)

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_label_space(self):
        # Return a deep copy of the label space.
        return self.demonstration.get_label_space()

    def get_default_ground_truth_label(self, index) -> str:
        if index < 0 or index >= len(self.test):
            raise ValueError("Index out of range.")
        return self.test.get_label(index)
    
    def get_default_ground_truth_label_index(self, index) -> int:
        if index < 0 or index >= len(self.test):
            raise ValueError("Index out of range.")
        return self.test.find_index_from_label(self.get_default_ground_truth_label(index))

    def change_label_space_triple(self, label_space: list[str]):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_label_space(label_space)
        self.demonstration.change_label_space(label_space)
        self.test.change_label_space(label_space)

    def change_instruction_triple(self, instruction: str):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_instruction(instruction)
        self.demonstration.change_instruction(instruction)
        self.test.change_instruction(instruction)
    
    def change_input_text_prefixes_triple(self, input_text_prefixes: list[str]):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_input_text_prefixes(input_text_prefixes)
        self.demonstration.change_input_text_prefixes(input_text_prefixes)
        self.test.change_input_text_prefixes(input_text_prefixes)
    
    def change_input_text_affixes_triple(self, input_text_affixes: list[str]):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_input_text_affixes(input_text_affixes)
        self.demonstration.change_input_text_affixes(input_text_affixes)
        self.test.change_input_text_affixes(input_text_affixes)

    def change_label_prefix_triple(self, label_prefix: str):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_label_prefix(label_prefix)
        self.demonstration.change_label_prefix(label_prefix)
        self.test.change_label_prefix(label_prefix)
    
    def change_label_affix_triple(self, label_affix: str):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_label_affix(label_affix)
        self.demonstration.change_label_affix(label_affix)
        self.test.change_label_affix(label_affix)

    def change_query_prefix_triple(self, query_prefix: str):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_query_prefix(query_prefix)
        self.demonstration.change_query_prefix(query_prefix)
        self.test.change_query_prefix(query_prefix)

    def change_label_space_triple(self, label_space: list[str]):
        warnings.warn("We don't recommend to use these methods. Use the same methods in the prompt_writter class instead.")
        self.calibration.change_label_space(label_space)
        self.demonstration.change_label_space(label_space)
        self.test.change_label_space(label_space)


class demonstration_sampler_v2():
    """
        Help the `experimentor` to sample the demonstration indexes for each query.
        Notice: if you want to define the demonstration indexes for each query, you should change the `demonstration_sampler` in the `experimentor` class into a list[list[int]] shaped variable.
        `demonstration_sampler` acts as a list[list[int]] shaped class.
        Main members:
            You shouldn't directly access all the members in this class. Use the methods in the `experimentor` class instead.
        Main methods:
            __init__:
                - k: int; the number of demonstrations for each query.
                - demonstration_set_size: int; the size of the demonstration set. The width of the sampled list (len(self[0])).
                - query_numbers: int; the number of queries. The length of the sampled list (len(self)).
                - seed: int; the random seed for sampling.
    """

    def __init__(self, k: int, demonstration_set_size: int, query_numbers: int, query_tasks=None, demo_tasks=None, seed = configs.STANDARD_SETTINGS["random_seed"]):
        self._k = k
        self._demonstration_set_size = demonstration_set_size
        self._query_numbers = query_numbers
        self._random = stable_random.stable_random(seed=seed)
        self._sampled_indexes = []
        if query_tasks != None and len(query_tasks) != query_numbers:
            raise ValueError(
                f"len(query_tasks) ({len(query_tasks)}) != query_numbers ({query_numbers})"
            )
        if demo_tasks != None and len(demo_tasks) != demonstration_set_size:
            raise ValueError(
                f"len(demo_tasks) ({len(demo_tasks)}) != demonstration_set_size ({demonstration_set_size})"
            )

        if query_tasks!= None and demo_tasks != None:
            self._query_tasks = list(query_tasks)
            self._demo_tasks = list(demo_tasks)
            self._demo_indices_by_task = {}
            for idx, t in enumerate(self._demo_tasks):
                if t not in self._demo_indices_by_task:
                    self._demo_indices_by_task[t] = []
                self._demo_indices_by_task[t].append(idx)

        self._complie()
    
    # def _get_next_sample(self):
    #     if self._k > self._demonstration_set_size:
    #         return self._random.sample_index_set(self._k, self._demonstration_set_size, True)
    #     else:
    #         return self._random.sample_index_set(self._k, self._demonstration_set_size, False)

    def _get_next_sample(self, task_id):
        """
        为给定 task_id 的 query 采样 k 个 demo:
        只从相同 task_id 的 demo 索引中采样
        """
        if task_id not in self._demo_indices_by_task:
            raise ValueError(f"No demonstrations for task {task_id!r}.")

        candidate_indices = self._demo_indices_by_task[task_id]
        n = len(candidate_indices)
        if n == 0:
            raise ValueError(f"No demonstrations for task {task_id!r}.")

        # 在候选子集的位置上采样,再映射回真实 demo index
        if self._k > n:
            # 候选数量不够,就允许重复采样
            sampled_pos = self._random.sample_index_set(self._k, n, True)
        else:
            sampled_pos = self._random.sample_index_set(self._k, n, False)

        return [candidate_indices[p] for p in sampled_pos]
        
    def _complie(self):
        self._sampled_indexes = []
        for i in range(self._query_numbers):
            task_id = self._query_tasks[i]
            self._sampled_indexes.append(self._get_next_sample(task_id))

        # for i in range(self._query_numbers):
        #     self._sampled_indexes.append(self._get_next_sample())

    def __len__(self) -> int:
        return len(self._sampled_indexes)
    
    def __getitem__(self, index: int) -> list[int]:
        return self.get_sampled_indexes(index)
    
    def __str__(self) -> str:
        return (
            "--- demonstration index sampler ---" + 
            "\n\tk: " + str(self._k) +
            "\n\tdemonstration set size: " + str(self._demonstration_set_size) +
            "\n\ttotal sample numbers: " + str(self._query_numbers)
        )

    def __repr__(self):
        return self.__str__()
    
    def _pop(self, index: int) -> list[int]:
        if index < 0 or index >= self._query_numbers:
            raise ValueError("Index out of range.")
        ret = self._sampled_indexes[index]
        self._sampled_indexes.pop(index)
        return ret
    
    def _insert(self, index: int, value: list[int]) -> None:
        if index < 0 or index > self._query_numbers:
            raise ValueError("Index out of range.")
        if len(value) != self._k:
            raise ValueError("The length of the value should be equal to k.")
        self._sampled_indexes.insert(index, value)
    
    def _append(self, value: list[int]) -> None:
        if len(value) != self._k:
            raise ValueError("The length of the value should be equal to k.")
        self._sampled_indexes.append(value)

    def _set_sample(self, index, value: list[int]) -> None:
        if index < 0 or index >= self._query_numbers:
            raise ValueError("Index out of range.")
        if len(value) != self._k:
            raise ValueError("The length of the value should be equal to k.")
        self._sampled_indexes[index] = value
    
    def get_sampled_indexes(self, index) -> list[int]:
        if index < 0 or index >= self._query_numbers:
            raise ValueError("Index out of range.")
        return self._sampled_indexes[index]


class demonstration_sampler():
    """
        Help the `experimentor` to sample the demonstration indexes for each query.
        Notice: if you want to define the demonstration indexes for each query, you should change the `demonstration_sampler` in the `experimentor` class into a list[list[int]] shaped variable.
        `demonstration_sampler` acts as a list[list[int]] shaped class.
        Main members:
            You shouldn't directly access all the members in this class. Use the methods in the `experimentor` class instead.
        Main methods:
            __init__:
                - k: int; the number of demonstrations for each query.
                - demonstration_set_size: int; the size of the demonstration set. The width of the sampled list (len(self[0])).
                - query_numbers: int; the number of queries. The length of the sampled list (len(self)).
                - seed: int; the random seed for sampling.
    """

    def __init__(self, k: int, demonstration_set_size: int, query_numbers: int, seed = configs.STANDARD_SETTINGS["random_seed"]):
        self._k = k
        self._demonstration_set_size = demonstration_set_size
        self._query_numbers = query_numbers
        self._random = stable_random.stable_random(seed=seed)
        self._sampled_indexes = []
        self._complie()
    
    def _get_next_sample(self):
        if self._k > self._demonstration_set_size:
            return self._random.sample_index_set(self._k, self._demonstration_set_size, True)
        else:
            return self._random.sample_index_set(self._k, self._demonstration_set_size, False)
    
    def _complie(self):
        for i in range(self._query_numbers):
            self._sampled_indexes.append(self._get_next_sample())

    def __len__(self) -> int:
        return len(self._sampled_indexes)
    
    def __getitem__(self, index: int) -> list[int]:
        return self.get_sampled_indexes(index)
    
    def __str__(self) -> str:
        return (
            "--- demonstration index sampler ---" + 
            "\n\tk: " + str(self._k) +
            "\n\tdemonstration set size: " + str(self._demonstration_set_size) +
            "\n\ttotal sample numbers: " + str(self._query_numbers)
        )

    def __repr__(self):
        return self.__str__()
    
    def _pop(self, index: int) -> list[int]:
        if index < 0 or index >= self._query_numbers:
            raise ValueError("Index out of range.")
        ret = self._sampled_indexes[index]
        self._sampled_indexes.pop(index)
        return ret
    
    def _insert(self, index: int, value: list[int]) -> None:
        if index < 0 or index > self._query_numbers:
            raise ValueError("Index out of range.")
        if len(value) != self._k:
            raise ValueError("The length of the value should be equal to k.")
        self._sampled_indexes.insert(index, value)
    
    def _append(self, value: list[int]) -> None:
        if len(value) != self._k:
            raise ValueError("The length of the value should be equal to k.")
        self._sampled_indexes.append(value)

    def _set_sample(self, index, value: list[int]) -> None:
        if index < 0 or index >= self._query_numbers:
            raise ValueError("Index out of range.")
        if len(value) != self._k:
            raise ValueError("The length of the value should be equal to k.")
        self._sampled_indexes[index] = value
    
    def get_sampled_indexes(self, index) -> list[int]:
        if index < 0 or index >= self._query_numbers:
            raise ValueError("Index out of range.")
        return self._sampled_indexes[index]


class prompt_writter():
    """
        Help the `experimentor` to write the prompt for inference or calibration.
        Notice: if you want to define the prompt template, you should use methods in the `prompt_former` in the `experimentor` class listed as below.
        Main members:
            You shouldn't directly access all the members in this class. Use the methods in the `experimentor` class instead.
        Main methods:
            __init__:
                - triplet_dataset: triplet_dataset; the triplet dataset.
                - use_noisy_channel: bool; whether to use the noisy channel mode (with a prompt like <label><text><label><text>...).
                - pseudo_query_generater: None or callable; the pseudo query generater (with next() available) for some special usage (such as Contextual Calibration). If None, the pseudo query will not be used.
            prompt_writter.reset(): None; set the prompt writter to the default template defined by the original dataset (`triplet_dataset`).
            prompt_writter.use_noisy_channel(new_label_affix = " ", new_last_input_affix = "\n"): None; set the prompt writter to the noisy channel mode. The label affix and the last_input_affix can be changed.
            prompt_writter.get_label_of_test_samples(query_index: int): str; get the label word of the `index`-th examples defined by the `mllm_dataset_loader.basic_datasets_loader` of the test set.
            change: 
                prompt_writter.change_instruction(instruction: str): None; change the instruction of the prompt writter into the `instruction`.
                prompt_writter.change_input_text_prefixes(input_text_prefixes: list[str]): None; change the input text prefixes of the prompt writter into the `input_text_prefixes`.
                prompt_writter.change_input_text_affixes(input_text_affixes: list[str]): None; change the input text affixes of the prompt writter into the `input_text_affixes`.
                prompt_writter.change_label_prefix(label_prefix: str): None; change the label prefix of the prompt writter into the `label_prefix`.
                prompt_writter.change_label_affix(label_affix: str): None; change the label affix of the prompt writter into the `label_affix`.
                prompt_writter.change_query_prefix(query_prefix: str): None; change the query prefix of the prompt writter into the `query_prefix`.
                prompt_writter.change_label_space(label_space: list[str]): None; change the label space of the prompt writter into the `label_space`.
            write_prompt(demos_indexes: list[int], query_index: int): str; write the prompt for the `demos_indexes`-th demonstrations and the `query_index`-th examples of the test set.
                  Prompt will be structured as: (notice that all the \n here are not default, you should add it if you want to split the instruction and the following input texts)
                  <prompt_writter.instruction> 
                  [ (for multiple-input tasks)
                    <prompt_writter.input_text_prefixes[0]> <prompt_writter.triplet_dataset.demonstration.get_input_text(index)[0]> <prompt_writter.input_text_prefixes[0]>
                    <prompt_writter.input_text_prefixes[1]> <prompt_writter.triplet_dataset.demonstration.get_input_text(index)[1]> <prompt_writter.input_text_prefixes[1]>
                    ...
                    <prompt_writter.label_prefix> <prompt_writter.label(index)> <prompt_writter.label_afffix>
                  ] * k (k = demostration numbers)
                  <prompt_writter.query_prefix>
                  [ (for multiple-input tasks)
                    <prompt_writter.input_text_prefixes[0]> <prompt_writter.triplet_dataset.test.get_input_text(index)[0]> <prompt_writter.input_text_prefixes[0]>
                    <prompt_writter.input_text_prefixes[1]> <prompt_writter.triplet_dataset.test.get_input_text(index)[1]> <prompt_writter.input_text_prefixes[1]>
                    ...
                    <prompt_writter.label_prefix> [MASKED]
                  ]

                  while, if the `use_noisy_channel` is True, we return a list of prompts w.r.t. various labels (label_iter), and each prompt will be structured as:
                  <prompt_writter.instruction> 
                  [ (for multiple-input tasks)
                    <prompt_writter.label_prefix> <prompt_writter.label(index)> <prompt_writter.label_afffix>
                    <prompt_writter.input_text_prefixes[0]> <prompt_writter.triplet_dataset.demonstration.get_input_text(index)[0]> <prompt_writter.input_text_prefixes[0]>
                    <prompt_writter.input_text_prefixes[1]> <prompt_writter.triplet_dataset.demonstration.get_input_text(index)[1]> <prompt_writter.input_text_prefixes[1]>
                    ...
                  ] * k (k = demostration numbers)
                  <prompt_writter.label_prefix> <prompt_writter.label_iter> <prompt_writter.label_afffix>
                  <prompt_writter.query_prefix>
                  [ (for multiple-input tasks)
                    <prompt_writter.input_text_prefixes[0]> <prompt_writter.triplet_dataset.test.get_input_text(index)[0]> <prompt_writter.input_text_prefixes[0]>
                    <prompt_writter.input_text_prefixes[1]> <prompt_writter.triplet_dataset.test.get_input_text(index)[1]> <prompt_writter.input_text_prefixes[1]>
                    ...
                  ]

            write_prompt_from_dataline(demos_lines: list[(list[str], str)], query_line: list[str]): str; write the prompt for the `demos_lines`-th demonstrations and the `query_line`-th examples of the test set.
              e.g., the demos_lines can be [(["thoughtful , provocative and entertaining ."], "positive"), (["don't be fooled by the impressive cast list - eye see you is pure junk ."], "negative")], and the query_line can be [""].
              (Notice that in this library, a input text is a list of strings to suite our library towards the multi-input tasks.)
              An example output will be: "review: thoughtful , provocative and entertaining . sentiment: positive\nreview: don't be fooled by the impressive cast list - eye see you is pure junk . sentiment: negative\nreview:  sentiment: "
            example(k = 8): str; get an example prompt with `k` demonstrations. For debug.
    """
    
    def __init__(
            self, 
            triplet_dataset: triplet_dataset, 
            use_noisy_channel = False,
            pseudo_query_generater = None,
        ):
        self._triplet_dataset = triplet_dataset
        self.reset()
        if use_noisy_channel:
            self.use_noisy_channel()
        self.pseudo_prompt = pseudo_query_generater
    
    def reset(self): #get_input_image
        self._instruction = copy.deepcopy(self._triplet_dataset.demonstration.get_instruction())
        self._input_text_prefixes = copy.deepcopy(self._triplet_dataset.demonstration.get_input_text_prefixes())
        self._input_text_affixes = copy.deepcopy(self._triplet_dataset.demonstration.get_input_text_affixes())
        self._label_prefix = copy.deepcopy(self._triplet_dataset.demonstration.get_label_prefix())
        self._label_affix = copy.deepcopy(self._triplet_dataset.demonstration.get_label_affix())
        self._query_prefix = copy.deepcopy(self._triplet_dataset.test.get_query_prefix())
        self._label_space = self._triplet_dataset.demonstration.get_label_space()
        self._random_for_example = stable_random.stable_random()
        self._noisy_channel = False
        self._random_for_label_error = stable_random.stable_random()
        self.label_wrong_rate = 0
        self.cut_by_length = 0
    
    def set_label_wrong_rate(self, label_wrong_rate: float):
        self.label_wrong_rate = label_wrong_rate

    def use_noisy_channel(self, new_label_affix = " ", new_last_input_affix = "\n"):
        if not self._noisy_channel:
            self._noisy_channel = True
            self._label_affix = new_label_affix
            self._input_text_affixes[-1] = new_last_input_affix

    def cancel_noisy_channel(self):
        if self._noisy_channel:
            self._noisy_channel = False
            self._label_affix = self._triplet_dataset.demonstration.get_label_affix()
            self._input_text_affixes[-1] = self._triplet_dataset.demonstration.get_input_text_affixes()[-1]

    def __str__(self) -> str:
        return (
            "--- In-context Learning prompt writter ---" + 
            "\n\tdemonstrations set: " + self._triplet_dataset.demonstration.get_dataset_name().replace('\n', '\\n') + 
            "\n\tqueries set: " + self._triplet_dataset.test.get_dataset_name().replace('\n', '\\n') + 
            "\n\tinstruction: " + self._instruction.replace('\n', '\\n') + 
            "\n\tinput text prefixes: " + str(self._input_text_prefixes).replace('\n', '\\n') + 
            "\n\tinput text affixes: " + str(self._input_text_affixes).replace('\n', '\\n') + 
            "\n\tlabel prefix: " + self._label_prefix.replace('\n', '\\n') + 
            "\n\tlabel affix: " + self._label_affix.replace('\n', '\\n') + 
            "\n\tquery prefix: " + self._query_prefix.replace('\n', '\\n') + 
            "\n\tlabel space: " + str(self._label_space).replace('\n', '\\n')
        )
    
    def __repr__(self):
        ret = self.__str__() + "\n"
        ret += "\n------------An Example of Prompt------------\n"
        ret += str(self.example())
        ret += "\n-------------------------------------------"
        ret += "\nWith:\n"
        ret += self._triplet_dataset.__repr__()
        return ret
    
    def get_config_dict(self):
        return {
            "instruction": copy.deepcopy(self._instruction),
            "input_text_prefixes": copy.deepcopy(self._input_text_prefixes),
            "input_text_affixes": copy.deepcopy(self._input_text_affixes),
            "label_prefix": copy.deepcopy(self._label_prefix),
            "label_affix": copy.deepcopy(self._label_affix),
            "query_prefix": copy.deepcopy(self._query_prefix),
            "label_space": copy.deepcopy(self._label_space),
            "label_wrong_rate": self.label_wrong_rate,
            "use_noisy_channel": self._noisy_channel,
        }
    
    def set_config_dict(self, config_dict: dict):
        self.reset()
        if "instruction" in config_dict:
            self.change_instruction(config_dict["instruction"])
        if "input_text_prefixes" in config_dict:
            self.change_input_text_prefixes(config_dict["input_text_prefixes"])
        if "input_text_affixes" in config_dict:
            self.change_input_text_affixes(config_dict["input_text_affixes"])
        if "label_prefix" in config_dict:
            self.change_label_prefix(config_dict["label_prefix"])
        if "label_affix" in config_dict:
            self.change_label_affix(config_dict["label_affix"])
        if "query_prefix" in config_dict:
            self.change_query_prefix(config_dict["query_prefix"])
        if "label_space" in config_dict:
            self.change_label_space(config_dict["label_space"])
        if "label_wrong_rate" in config_dict:
            self.set_label_wrong_rate(config_dict["label_wrong_rate"])
        if "use_noisy_channel" in config_dict:
            if config_dict["use_noisy_channel"]:
                self.use_noisy_channel()
            else:
                self.reset()
    
    # def get_label_of_test_samples(self, query_index: int):
    #     if query_index < 0 or query_index >= len(self._triplet_dataset.test):
    #         raise ValueError("Index out of range.")
    #     return self._triplet_dataset.test.get_label(query_index)

    def change_instruction(self, instruction: str):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(instruction) is not str:
            raise ValueError("Instruction should be a string.")
        self._instruction = copy.deepcopy(instruction)

    def change_input_text_prefixes(self, input_text_prefixes: list[str]):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(input_text_prefixes) is not list:
            raise ValueError("Input text prefixes should be a list.")
        for prefix in input_text_prefixes:
            if type(prefix) is not str:
                raise ValueError("Input text prefixes should be a list of strings.")
        self._input_text_prefixes = copy.deepcopy(input_text_prefixes)
    
    def change_input_text_affixes(self, input_text_affixes: list[str]):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(input_text_affixes) is not list:
            raise ValueError("Input text affixes should be a list.")
        for affix in input_text_affixes:
            if type(affix) is not str:
                raise ValueError("Input text affixes should be a list of strings.")
        if len(input_text_affixes) != self.input_element_numbers:
            raise ValueError("The number of input text affixes should be equal to the number of input elements.")
        self._input_text_affixes = copy.deepcopy(input_text_affixes)
    
    def change_label_prefix(self, label_prefix: str):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(label_prefix) is not str:
            raise ValueError("Label prefix should be a string.")
        self._label_prefix = copy.deepcopy(label_prefix)
    
    def change_label_affix(self, label_affix: str):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(label_affix) is not str:
            raise ValueError("Label affix should be a string.")
        self._label_affix = copy.deepcopy(label_affix)
    
    def change_query_prefix(self, query_prefix: str):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(query_prefix) is not str:
            raise ValueError("Query prefix should be a string.")
        self._query_prefix = copy.deepcopy(query_prefix)

    def change_label_space(self, label_space: list[str]):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(label_space) is not list:
            raise ValueError("Label space should be a list.")
        for label in label_space:
            if type(label) is not str:
                raise ValueError("Label space should be a list of strings.")
        self._label_space = label_space

    def get_label_space(self):
        # Return a deep copy of the label space.
        return self._label_space
    
    def replace_space_to_label(self):
        # Notice: use this function will reset the label space and the label prefix.
        self.reset()
        new_label_space = self.get_label_space()
        for i in range(len(new_label_space)):
            new_label_space[i] = ' ' + new_label_space[i]
        self.change_label_prefix(self._label_prefix[:-1])
        self.change_label_space(new_label_space)
    
    def write_prompt(self, demos_indexes: list[int], query_index: int = None):
        # Use the indexes of the demonstrations and the query to write a prompt.
        # demos_indexes: [demo1, demo2, ..., demok]
        # query_index: query
        if self.label_wrong_rate < 0 or self.label_wrong_rate > 1:
            raise ValueError("The label wrong rate should be in [0, 1].")
        if query_index is None and not self.pseudo_prompt:
            raise ValueError("The query index should be given. If you want to use the pseudo query, please set the pseudo query generater.")
        wrong_label_number = int(len(demos_indexes) * self.label_wrong_rate)
        if wrong_label_number != 0 and wrong_label_number / len(demos_indexes) != self.label_wrong_rate:
            warnings.warn("The number of wrong labels is not an integer.")
        demo_text_lines = []
        demo_image_lines = []
        query_text_line = []
        query_image_line = []
        
        wrong_labels = self._random_for_label_error.sample_n_elements_from_list(demos_indexes, wrong_label_number, allow_repetition = False)
        for demosindex in demos_indexes:
            # 可能是wrong label
            if demosindex < 0 or demosindex >= len(self._triplet_dataset.demonstration):
                raise ValueError("Index out of range.")
            label_token = self._triplet_dataset.demonstration.get_label(demosindex)
            if demosindex in wrong_labels:
                label_token = self._triplet_dataset.demonstration.get_label_space()[
                    (self._triplet_dataset.demonstration.find_index_from_label(label_token) + 1) % len(self._triplet_dataset.demonstration.get_label_space())
                ]
            demo_text_lines.append((self._triplet_dataset.demonstration.get_input_text(demosindex), label_token))
            demo_image_lines.append((self._triplet_dataset.demonstration.get_input_image(demosindex)))
        if self.pseudo_prompt:
            query_text_line = next(self.pseudo_prompt)
        else:
            if query_index < 0 or query_index >= len(self._triplet_dataset.test):
                raise ValueError("Index out of range.")
            query_text_line = self._triplet_dataset.test.get_input_text(query_index)
            query_image_line.append(self._triplet_dataset.test.get_input_image(query_index))
        return self.write_prompt_from_dataline(demo_text_lines, demo_image_lines, query_text_line, query_image_line, self.cut_by_length)


    def write_prompt_from_dataline(self, demos_lines: list[(list[str], str)], demos_img_lists: list[str], query_line: list[str], query_img_list: list[str], cut_by_length = 0):
        """ 
            You can organize your own data line and use this function to makeup the prompt for inference or calibration.
            For example, in the contextual calibration http://arxiv.org/abs/2102.09690, you can use the following parameters to makeup the prompt for calibration:
            self.write_prompt_from_dataline(
              [
                  (["thoughtful , provocative and entertaining ."], "positive"), 
                  (["don't be fooled by the impressive cast list - eye see you is pure junk ."], "negative")
              ], 
              [""]
            ) for a k = 2 scenario.
            And the output is: "review: thoughtful , provocative and entertaining . sentiment: positive\nreview: don't be fooled by the impressive cast list - eye see you is pure junk . sentiment: negative\nreview:  sentiment: "
            demos_line: [(<demo1> [input1, input2, ...], label_word), (<demo2> [input1, input2, ...], label_word), ..., (<demok> [input1, input2, ...], label_word)]
            demos_img_line: [(<demo1> [img1, img2, ...]), (<demo2> [img1, img2, ...]), ..., (<demok> [img1, img2, ...])]
            query_line: [input1, input2, ...]
            query_img_line: [img1, img2, ...]
            Return: List[str]: prompts for every label token.
        """

        #...先不考虑这个吧，
        if self._noisy_channel:
            ret = []
            for label in self._label_space:
                prompt = self._instruction
                i = 0
                for demos in demos_lines: #demos 的text input
                    prompt += self._label_prefix + self._label_space[self._triplet_dataset.demonstration.find_index_from_label(demos[1])] + self._label_affix
                    for i in range(self._triplet_dataset.demonstration.get_input_element_numbers()):
                        prompt += self._input_text_prefixes[i] + demos[0][i] + self._input_text_affixes[i]
                prompt += self._label_prefix + label + self._label_affix + self._query_prefix
                for i in range(self._triplet_dataset.test.get_input_element_numbers()):
                    prompt += self._input_text_prefixes[i] + query_line[i] + self._input_text_affixes[i]
                ret.append(prompt[:len(prompt) - cut_by_length])
            return ret
            
        # DIRECT
        # Return: str
        else:
            text_input_list = []
            start_signal = True
            for demos in demos_lines:
                for i in range(self._triplet_dataset.demonstration.get_input_element_numbers()):
                    if start_signal:
                        prompt = self._instruction + self._input_text_prefixes[i] + demos[0][i] + self._input_text_affixes[i]
                        prompt += self._label_prefix + self._label_space[self._triplet_dataset.demonstration.find_index_from_label(demos[1])] + self._label_affix
                        text_input_list.append(prompt)
                    else:
                        prompt = self._input_text_prefixes[i] + demos[0][i] + self._input_text_affixes[i]
                        prompt += self._label_prefix + self._label_space[self._triplet_dataset.demonstration.find_index_from_label(demos[1])] + self._label_affix
                        text_input_list.append(prompt)

            q_prompt = self._query_prefix
            for i in range(self._triplet_dataset.test.get_input_element_numbers()):
                q_prompt += self._input_text_prefixes[i] + query_line[i] + self._input_text_affixes[i]
            q_prompt += self._label_prefix

            text_input_list.append(q_prompt)

            image_input_list = demos_img_lists + query_img_list # image 相加

            return {"text": text_input_list, "image":image_input_list}


    def get_prompt_dir(self, demos_indexes: list[int], query_index: int = None):
        # Use the indexes of the demonstrations and the query to write a prompt.
        # demos_indexes: [demo1, demo2, ..., demok]
        # query_index: query
        if self.label_wrong_rate < 0 or self.label_wrong_rate > 1:
            raise ValueError("The label wrong rate should be in [0, 1].")
        if query_index is None and not self.pseudo_prompt:
            raise ValueError("The query index should be given. If you want to use the pseudo query, please set the pseudo query generater.")
        wrong_label_number = int(len(demos_indexes) * self.label_wrong_rate)
        if wrong_label_number != 0 and wrong_label_number / len(demos_indexes) != self.label_wrong_rate:
            warnings.warn("The number of wrong labels is not an integer.")
        demo_text_lines = []
        demo_image_lines = []
        query_text_line = []
        query_image_line = []
        
        wrong_labels = self._random_for_label_error.sample_n_elements_from_list(demos_indexes, wrong_label_number, allow_repetition = False)
        for demosindex in demos_indexes:
            # 可能是wrong label
            if demosindex < 0 or demosindex >= len(self._triplet_dataset.demonstration):
                raise ValueError("Index out of range.")
            label_token = self._triplet_dataset.demonstration.get_label(demosindex)
            if demosindex in wrong_labels:
                label_token = self._triplet_dataset.demonstration.get_label_space()[
                    (self._triplet_dataset.demonstration.find_index_from_label(label_token) + 1) % len(self._triplet_dataset.demonstration.get_label_space())
                ]
            demo_text_lines.append((self._triplet_dataset.demonstration.get_input_text(demosindex), label_token))
            demo_image_lines.append((self._triplet_dataset.demonstration.get_input_image(demosindex)))
        if self.pseudo_prompt:
            query_text_line = next(self.pseudo_prompt)
        else:
            if query_index < 0 or query_index >= len(self._triplet_dataset.test):
                raise ValueError("Index out of range.")
            query_text_line = self._triplet_dataset.test.get_input_text(query_index)
            query_image_line.append(self._triplet_dataset.test.get_input_image(query_index))
        return self.get_prompt_from_dataline(demo_text_lines, demo_image_lines, query_text_line, query_image_line)
    

    def get_prompt_from_dataline(self, demos_lines: list[(list[str], str)], demos_img_lists: list[str], query_line: list[str], query_img_list: list[str]):
        text_input_list = []
        label_list = []
        start_signal = True
        for demos in demos_lines:
            for i in range(self._triplet_dataset.demonstration.get_input_element_numbers()):
                if start_signal:
                    prompt = self._instruction + self._input_text_prefixes[i] + demos[0][i] + self._input_text_affixes[i]
                    label= self._label_prefix + self._label_space[self._triplet_dataset.demonstration.find_index_from_label(demos[1])] + self._label_affix
                    label_list.append(label)
                    text_input_list.append(prompt)
                else:
                    prompt = self._input_text_prefixes[i] + demos[0][i] + self._input_text_affixes[i]
                    label = self._label_prefix + self._label_space[self._triplet_dataset.demonstration.find_index_from_label(demos[1])] + self._label_affix
                    label_list.append(label)
                    text_input_list.append(prompt)
                    text_input_list.append(prompt)

        q_prompt = self._query_prefix
        for i in range(self._triplet_dataset.test.get_input_element_numbers()):
            q_prompt += self._input_text_prefixes[i] + query_line[i] + self._input_text_affixes[i]

        
        text_input_list.append(q_prompt)

        image_input_list = demos_img_lists + query_img_list # image 相加

        return {"question": text_input_list, "image":image_input_list , "label": label_list} # demo_n +1 , demo_n+1. demo_n
    
        
    def example(self, k = 8):
        if k < 0 or k > len(self._triplet_dataset.demonstration):
            raise ValueError("Invalid number of demonstrations.")
        Dindexes = self._random_for_example.sample_index_set(k, len(self._triplet_dataset.demonstration))
        Qindex = self._random_for_example.get_int_from_range(0, len(self._triplet_dataset.test))
        return self.write_prompt(Dindexes, Qindex)