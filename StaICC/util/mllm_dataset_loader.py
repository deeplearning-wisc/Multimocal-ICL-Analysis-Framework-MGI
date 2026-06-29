# Adapt datasets into a list_formed class
from . import stable_random
from . import configs
import warnings
import copy

class basic_datasets_loader():
    # Interface for prompt_writer. 
    # Prompt will be structured as: 
    # <basic_datasets_loader._instruction>
    # [ (for multiple-input tasks)
    #   <basic_datasets_loader.get_input_text_prefixes[0]> <basic_datasets_loader.get_input_text(index)[0]> <basic_datasets_loader.get_input_text_prefixes[0]>
    #   <basic_datasets_loader.get_input_text_prefixes[1]> <basic_datasets_loader.get_input_text(index)[1]> <basic_datasets_loader.get_input_text_prefixes[1]>
    #   ...
    #   <basic_datasets_loader.get_label_prefix> <basic_datasets_loader.get_label(index)> <basic_datasets_loader.get_label_afffix>
    # ] * k (k = demostration numbers)
    # <basic_datasets_loader.get_query_prefix>
    # [ (for multiple-input tasks)
    #   <basic_datasets_loader.get_input_text_prefixes[0]> <basic_datasets_loader.get_input_text(index)[0]> <basic_datasets_loader.get_input_text_prefixes[0]>
    #   <basic_datasets_loader.get_input_text_prefixes[1]> <basic_datasets_loader.get_input_text(index)[1]> <basic_datasets_loader.get_input_text_prefixes[1]>
    #   ...
    #   <basic_datasets_loader.get_label_prefix> [MASKED]
    # ]
    def __init__(self):
        self._hgf_dataset = None  # Huggingface Dataset Class. Will be overloaded by datasets.load_dataset.
        self._instruction = ""  # STRING. Instruction for the dataset in the begining of prompts. Can't be None.
        self._input_text_prefixes = ["Text: "] # LIST of STRING. Prefixes for the input text.
        self._input_image_prefixes = ["Image: "] # LIST of STRING. Prefixes for the input text.
        self._input_text_affixes = [" "] # LIST of STRING. Affixes for the input text.
        self._label_prefix = "Label: " # STRING. Prefix for the label.
        self._label_affix = "\n" # STRING. Affix for the label.
        self._query_prefix = "" # STRING. Prefix for the query.
        self._label_space = [""] # LIST of STRING. Space for the label. Will be overloaded by the dataset.
        self._ground_truth_label_space = None # LIST of STRING. Ground truth label space. Will be overloaded by the dataset.
        self._reducted_label_space = None # LIST of STRING. Reducted label space. Will be overloaded by the dataset.
        self._label_mapping = {} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.table = None # LIST of (LIST of STRING, STRING). The table form of the dataset. Will be create by _transform_hgf_dataset_to_table.
        self._package_path = __package__[0:-5]

        self._long_text_classification = False

        self.input_element_numbers = 1 # INT. Number of input elements. According to the dataset.
        self.label_space_numbers = 1 # INT. Number of labels. According to the dataset.
        self.dataset_name = "" # STRING. Name of the dataset. Will be overloaded by the dataset.
    
    def _complie_dataset(self):
        # This function is used to transform the huggingface dataset to a table. And shuffle, cut the overlength data.
        # And also calculate the label_space_numbers and input_element_numbers.
        # Finally, delete the _hgf_dataset.
        pass

    def _shuffle(self):
        randomer = stable_random.stable_random()
        index = randomer.sample_index_set(len(self), len(self))
        self.table = [self.table[i] for i in index]

    def __len__(self) -> int:
        # Should return the number of elements in the dataset.
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[list[str], str]:
        # Should return a (list of strings, string). 
        # list of string: The length is the number of input elements.
        # string: The label.
        if self.has_task_type:
            return (self.get_input_text(index), self.get_input_image(index), self.get_label(index), self.get_input_task_type(index))
        else:
            return (self.get_input_text(index), self.get_input_image(index), self.get_label(index))
    
    def __str__(self) -> str:
        return (
            "--- basic dataset loader ---" + 
            "\n\tdataset name: " + self.dataset_name + 
            "\n\tlength: " + str(len(self)).replace('\n', '\\n') + 
            "\n\tinstructions: " + self._instruction.replace('\n', '\\n') + 
            "\n\tinput_text_prefixes: " + str(self._input_text_prefixes).replace('\n', '\\n') + 
            "\n\tinput_image_prefixes: " + str(self._input_image_prefixes).replace('\n', '\\n') + 
            "\n\tinput_text_affixes: " + str(self._input_text_affixes).replace('\n', '\\n') + 
            "\n\tlabel_prefix: " + self._label_prefix.replace('\n', '\\n') + 
            "\n\tlabel_affix: " + self._label_affix.replace('\n', '\\n') + 
            "\n\tquery_prefix: " + self._query_prefix.replace('\n', '\\n') + 
            "\n\tlabel_space: " + str(self._label_space).replace('\n', '\\n') + 
            "\n\tfor long text classification: " + str(self._long_text_classification)
        )
    
    def __repr__(self):
        ret = self.__str__()
        ret += '\n\tElements: '
        ret += str(self.table[0])
        ret += ' + ' + str(len(self) - 1) + " more."
        return ret

    def _automatic_cut_by_length(self):
        # This function is used to cut the dataset by length. 
        # The length is defined by the standard settings.
        if self._long_text_classification:
            self._cut_by_length(configs.STANDARD_SETTINGS["cut_by_length_remain_long"], False)
        else:
            self._cut_by_length()

    def _cut_by_length(self, length = configs.STANDARD_SETTINGS["cut_by_length_remain_short"], remain_short = True):
        # This function is used to cut the dataset by length.
        if remain_short and length != configs.STANDARD_SETTINGS["cut_by_length_remain_short"]:
            warnings.warn(configs.WARNING_SETTINGS["tampering"])
        exclude_list = []
        for i in range(0, len(self.table)):
            if remain_short:
                if self.get_total_length_of_one_data(i) > length:
                    exclude_list.append(i)
            else:
                if self.get_total_length_of_one_data(i) < length:
                    exclude_list.append(i)
        self.table = [self.table[i] for i in range(0, len(self)) if i not in exclude_list]

    def full_label_token(self):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if self._ground_truth_label_space is None:
            warnings.warn("Not applicable on this dataset.")
            return
        self._label_space = self._ground_truth_label_space
    
    def reduct_label_token(self):
        if self._reducted_label_space is None:
            warnings.warn("Not applicable on this dataset.")
            return
        self._label_space = self._reducted_label_space

    def rename_dataset(self, new_name: str):
        # This function is used to rename the dataset.
        if type(new_name) is not str:
            raise ValueError("Dataset name should be a string.")
        self.dataset_name = new_name

    def cut_by_index(self, index: int):
        # This function is used to cut the dataset by index.
        if index < 0 or index > len(self):
            raise ValueError("Index out of range.")
        self.table = self.table[0:index]
        return self
    
    def get_dataset(self):
        return self.table
    
    def get_input_element_numbers(self):
        return self.input_element_numbers
    
    def get_dataset_name(self):
        return self.dataset_name

    def get_input_text_prefixes(self):
        return self._input_text_prefixes
    
    def get_input_text_affixes(self):
        return self._input_text_affixes
    
    def get_label_prefix(self):
        return self._label_prefix
    
    def get_label_affix(self):
        return self._label_affix
    
    def get_instruction(self):
        return self._instruction
    
    def get_query_prefix(self):
        return self._query_prefix
    
    def get_label_space(self):
        return self._label_space
    
    def get_alternate_template(self):
        return self.alternate_template
    
    def change_instruction(self, instruction: str):
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["basic_dataset_template_protect"])
            return
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(instruction) is not str:
            raise ValueError("Instruction should be a string.")
        self._instruction = instruction

    def change_input_text_prefixes(self, input_text_prefixes: list[str]):
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["basic_dataset_template_protect"])
            return
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(input_text_prefixes) is not list:
            raise ValueError("Input text prefixes should be a list.")
        for prefix in input_text_prefixes:
            if type(prefix) is not str:
                raise ValueError("Input text prefixes should be a list of strings.")
        if len(input_text_prefixes) != self.input_element_numbers:
            raise ValueError("The number of input text prefixes should be equal to the number of input elements.")
        self._input_text_prefixes = input_text_prefixes
    
    def change_input_text_affixes(self, input_text_affixes: list[str]):
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["basic_dataset_template_protect"])
            return
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(input_text_affixes) is not list:
            raise ValueError("Input text affixes should be a list.")
        for affix in input_text_affixes:
            if type(affix) is not str:
                raise ValueError("Input text affixes should be a list of strings.")
        if len(input_text_affixes) != self.input_element_numbers:
            raise ValueError("The number of input text affixes should be equal to the number of input elements.")
        self._input_text_affixes = input_text_affixes
    
    def change_label_prefix(self, label_prefix: str):
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["basic_dataset_template_protect"])
            return
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(label_prefix) is not str:
            raise ValueError("Label prefix should be a string.")
        self._label_prefix = label_prefix
    
    def change_label_affix(self, label_affix: str):
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["basic_dataset_template_protect"])
            return
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(label_affix) is not str:
            raise ValueError("Label affix should be a string.")
        self._label_affix = label_affix
    
    def change_query_prefix(self, query_prefix: str):
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["basic_dataset_template_protect"])
            return
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(query_prefix) is not str:
            raise ValueError("Query prefix should be a string.")
        self._query_prefix = query_prefix

    def change_label_space(self, label_space: list[str]):
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if type(label_space) is not list:
            raise ValueError("Label space should be a list.")
        for label in label_space:
            if type(label) is not str:
                raise ValueError("Label space should be a list of strings.")
        self._label_space = label_space

    def get_input_text(self, index: int) -> list[str]:
        # Should return a list of strings. The length is the number of input elements.
        return self.table[index][0]

    def get_input_image(self, index: int) -> list[str]:
        # Should return a list of strings. The length is the number of input elements.
        return self.table[index][1]

    def get_label(self, index: int):
        # Should return a string. Should call the _label_mapping.
        # print("self.table[index][2]",type(self.table[index][2]))
        return self.label_index_to_text(self.table[index][2])

    def get_input_task_type(self, index: int) -> list[str]:
        # Should return a list of strings. The length is the number of input elements.
        return self.table[index][3]
      
    def label_index_to_text(self, label_index: int):
        return self._label_space[self._label_mapping[label_index]]
    
    def find_index_from_label(self, label: str) -> int:
        # Should return the index of the label in the label space.
        return self._label_space.index(label)

    def get_total_length_of_one_data(self, index: int) -> int:
        # Should return the total length of one data. 
        ret = 0
        data = self.get_input_text(index)
        for element in data:
            ret += len(element)
        return ret

    def split(self, split_indexes: list[list[int]]):
        ret = []
        for indexes in split_indexes:
            new_dataset = copy.deepcopy(self)
            new_dataset.table = [new_dataset.table[i] for i in indexes]
            ret.append(new_dataset)
        return ret


class clock(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]


        self._label_prefix = "Answer: "
        self.dataset_name = "clock" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": [""], #, "Identify the single minority in the image."
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }
        self.train_labels = set()

        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/clock/train.jsonl",
                    "validation": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/clock/train.jsonl",
                    "test": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/clock/train.jsonl"
                }
            )['train']
        self._complie_dataset()


        # 如果只是做 ICL，不强制需要 label_space，保留一个去重后的字符串列表即可
        all_labels = sorted(self.train_labels, key=lambda x: int(x))  # 按数值排序
        self._label_space = all_labels
        self.label_space_numbers = len(self._label_space)
        self._label_mapping = {label: idx for idx, label in enumerate(self._label_space)}


    def _complie_dataset(self):
        self.table = []
        
        if "task_type" in self._hgf_dataset[0].keys():
            self.has_task_type = True
            
        else:
            self.has_task_type = False

        # 这里不要动 self._label_mapping 了
        for i in range(len(self._hgf_dataset)):
            text = self._hgf_dataset[i]["text"]
            img = self._hgf_dataset[i]["img"]
            label = str(self._hgf_dataset[i]["label"])
            if self.has_task_type:
                task_type = self._hgf_dataset[i]['task_type']
                self.table.append(([text], img, label, task_type))
            else:
                self.table.append(([text], img, label))
            self.train_labels.add(label)
        del self._hgf_dataset
        self._automatic_cut_by_length()
        self._shuffle()

    # 覆盖基类的 get_label，让它直接返回字符串 label
    def get_label(self, index: int):
        return self.table[index][2]

    # 如果别的地方调用了 label_index_to_text，可以让它支持两种输入类型
    def label_index_to_text(self, label_index):
        # 如果传进来的是字符串本身，直接返回
        if isinstance(label_index, str):
            return label_index
        # 如果传的是索引数字，则从 label_space 中取
        return self._label_space[label_index]
    

class operator_induction(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]


        self._label_prefix = "Answer: "
        self.dataset_name = "operator_induction" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": [""], #, "Identify the single minority in the image."
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }
        self.train_labels = set()

        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/operator_induction/train.jsonl",
                    "validation": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/operator_induction/train.jsonl",
                    "test": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/operator_induction/train.jsonl"
                }
            )['train']
        self._complie_dataset()

        all_labels = sorted(self.train_labels)  # train_labels 里包含所有可能的整数结果，含负数
        self._label_space = [str(l) for l in all_labels]
        self.label_space_numbers = len(self._label_space)
        self._label_mapping = {str(l): idx for idx, l in enumerate(all_labels)}
    
    # def _complie_dataset(self):
    #     self.table = []

    #     for i in range(0, len(self._hgf_dataset)):
    #         self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
    #         label = self._hgf_dataset[i]["label"]
    #         self.train_labels.add(label)  # 
    #     del self._hgf_dataset

    #     self._automatic_cut_by_length()
    #     self._shuffle()
    def _complie_dataset(self):
        self.table = []
        
        if "task_type" in self._hgf_dataset[0].keys():
            self.has_task_type = True
            
        else:
            self.has_task_type = False

        # 这里不要动 self._label_mapping 了
        for i in range(len(self._hgf_dataset)):
            text = self._hgf_dataset[i]["text"]
            img = self._hgf_dataset[i]["img"]
            label = str(self._hgf_dataset[i]["label"])
            if self.has_task_type:
                task_type = self._hgf_dataset[i]['task_type']
                self.table.append(([text], img, label, task_type))
            else:
                self.table.append(([text], img, label))
            self.train_labels.add(label)
        del self._hgf_dataset
        self._automatic_cut_by_length()
        self._shuffle()

class shape_ood(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()
        self.has_task_type = False

        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]
        self._label_space = ["circle", "triangle", "square", "star",  "yellow", "blue", "green", "red", "black", "orange", "purple", "pink", "brown", "gray"] # LIST of STRING. Space for the label. Will be overloaded by the dataset. 
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "Answer: "
        self._label_mapping = {"circle":0, "triangle":1, "square":2, "star":3,  "yellow":4, "blue":5, "green":6, "red":7, "black":8, "orange":9, "purple":10, "pink":11, "brown":12, "gray":13} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "shape_ood" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": [""], #, "Identify the single minority in the image."
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }


        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count/train_shape.jsonl",
                    "validation": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count/train_shape.jsonl",
                    "test": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count/train_shape.jsonl"
                }
            )['train']
        self._complie_dataset()

    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()

class color_ood(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()
        self.has_task_type = False
        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]
        self._label_space = ["circle", "triangle", "square", "star",  "yellow", "blue", "green", "red", "black", "orange", "purple", "pink", "brown", "gray"] # LIST of STRING. Space for the label. Will be overloaded by the dataset. 
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "Answer: "
        self._label_mapping = {"circle":0, "triangle":1, "square":2, "star":3,  "yellow":4, "blue":5, "green":6, "red":7, "black":8, "orange":9, "purple":10, "pink":11, "brown":12, "gray":13} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "color_ood" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": [""], #, "Identify the single minority in the image."
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }


        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count/train_color.jsonl",
                    "validation": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count/train_color.jsonl",
                    "test": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count/train_color.jsonl"
                }
            )['train']
        self._complie_dataset()

    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()

class shape_ood_text_only(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()
        self.has_task_type = False
        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]
        self._label_space = ["circle", "triangle", "square", "star",  "yellow", "blue", "green", "red", "black", "orange", "purple", "pink", "brown", "gray"] # LIST of STRING. Space for the label. Will be overloaded by the dataset. 
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "Answer: "
        self._label_mapping = {"circle":0, "triangle":1, "square":2, "star":3,  "yellow":4, "blue":5, "green":6, "red":7, "black":8, "orange":9, "purple":10, "pink":11, "brown":12, "gray":13} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "shape_ood_text_only" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": ["Identify the single minority (either color or shape) in the sentence. Output with one lowercase word."], #, 
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }


        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count_text_only/train_shape.jsonl",
                    "validation": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count_text_only/train_shape.jsonl",
                    "test": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count_text_only/train_shape.jsonl"
                }
            )['train']
        self._complie_dataset()

    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class color_ood_text_only(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]
        self._label_space = ["circle", "triangle", "square", "star",  "yellow", "blue", "green", "red", "black", "orange", "purple", "pink", "brown", "gray"] # LIST of STRING. Space for the label. Will be overloaded by the dataset. 
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "Answer: "
        self._label_mapping = {"circle":0, "triangle":1, "square":2, "star":3,  "yellow":4, "blue":5, "green":6, "red":7, "black":8, "orange":9, "purple":10, "pink":11, "brown":12, "gray":13} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "color_ood_text_only" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1
        self.has_task_type = False
        self.alternate_template = {
            "instruction": ["Identify the single minority (either color or shape) in the sentence. Output with one lowercase word."], #, 
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }


        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count_text_only/train_color.jsonl",
                    "validation": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count_text_only/train_color.jsonl",
                    "test": "/u/y/u/yuwang/yuwang/ICL_Circuit/data/shapes_count_text_only/train_color.jsonl"
                }
            )['train']
        self._complie_dataset()

    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()



class hateful_meme(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["input: "]
        self._label_space = ["true", "false"] # LIST of STRING. Space for the label. Will be overloaded by the dataset.
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "hatefulness: "
        self._label_mapping = {0:0, 1:1} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "hateful_meme" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": ["", "How would you describe the overall feeling of the movie based on this sentence? ", "Please classify the sentiment of the following sentence. "],
            "input_text_prefixes": [["input: "], ["meme: "]],
            "label_prefix": ["hatefulness: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        
        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/home/yuw/ICL_Circuit/data/hateful-memes/train.jsonl",
                    "validation": "/home/yuw/ICL_Circuit/data/hateful-memes/dev.jsonl",
                    "test": "/home/yuw/ICL_Circuit/data/hateful-memes/test.jsonl"
                }
            )['train']
        self._complie_dataset()

    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()

class snli(basic_datasets_loader):
    # https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset?resource=download
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["text: "]
        self._input_image_prefixes = ["image: "]
        self._label_space = ["+", "*", "-"] # LIST of STRING. Space for the label. Will be overloaded by the dataset. 
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "Relation: "
        self._label_mapping = {"entailment":0, "neutral":1, "contradiction":2} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "snli" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": ["", "How would you evaluate whether the sentence can be inferred from the image?", "Please determine whether the sentence can be inferred from the given image."],
            "input_text_prefixes": [["text: "], ["Text: "], ["sentence: "]],
            "input_image_prefixes": [["image: "], ["Image: "]],
            "label_prefix": ["Relation: ", "Answer: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }


        import datasets
        self._hgf_dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "/home/yuw/ICL_Circuit/data/snli/train.jsonl",
                    "validation": "/home/yuw/ICL_Circuit/data/snli/dev.jsonl",
                    "test": "/home/yuw/ICL_Circuit/data/snli/test.jsonl"
                }
            )['train']
        self._complie_dataset()

    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["img"], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


