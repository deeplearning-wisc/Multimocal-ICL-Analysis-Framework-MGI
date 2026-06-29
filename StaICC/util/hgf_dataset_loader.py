# Adapt datasets into a list_formed class
from . import stable_random
from . import configs
import warnings
import copy
import pickle
import pkgutil

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
        self._input_text_prefixes = ["Input: "] # LIST of STRING. Prefixes for the input text.
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
        return (self.get_input_text(index), self.get_label(index))
    
    def __str__(self) -> str:
        return (
            "--- basic dataset loader ---" + 
            "\n\tdataset name: " + self.dataset_name + 
            "\n\tlength: " + str(len(self)).replace('\n', '\\n') + 
            "\n\tinstructions: " + self._instruction.replace('\n', '\\n') + 
            "\n\tinput_text_prefixes: " + str(self._input_text_prefixes).replace('\n', '\\n') + 
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


    def get_image_text(self, index: int) -> list[str]:
        # Should return a list of strings. The length is the number of input elements.
        return self.table[index][0]


    def get_label(self, index: int) -> str:
        # Should return a string. Should call the _label_mapping.
        return self.label_index_to_text(self.table[index][1])
    
    def label_index_to_text(self, label_index: int) -> str:
        return copy.deepcopy(self._label_space[self._label_mapping[label_index]])
    
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


# 4 Sentiment Datasets 
class glue_sst2(basic_datasets_loader):
    # https://aclanthology.org/D13-1170/
    # https://arxiv.org/abs/1804.07461
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["sentence: "]
        self._label_space = ["negative", "positive"] # LIST of STRING. Space for the label. Will be overloaded by the dataset.
        self.label_space_numbers = len(self._label_space)
        self._label_prefix = "sentiment: "
        self._label_mapping = {0:0, 1:1} # DICT. INT to INT. Mapping from label index from _hgf_dataset to the label index of _label_space. Will be overloaded by the dataset.
        self.dataset_name = "GLUE-SST2" # STRING. Name of the dataset. Will be overloaded by the dataset.
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": ["", "How would you describe the overall feeling of the movie based on this sentence? ", "Please classify the sentiment of the following sentence. "],
            "input_text_prefixes": [["sentence: "], ["text: "], ["review: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("glue", "sst2")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/sst2.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/sst2.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["sentence"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class rotten_tomatoes(basic_datasets_loader):
    # https://arxiv.org/abs/cs/0506075
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["review: "]
        self._label_space = ["negative", "positive"] 
        self._label_prefix = "sentiment: "
        self._label_mapping = {0:0, 1:1} 
        self.dataset_name = "rotten_tomatoes" 
        self._long_text_classification = long_text_classification
        self.label_space_numbers = len(self._label_space)
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": ["", "How would you describe the overall feeling of the movie based on this sentence? ", "Please classify the sentiment of the following sentence. "],
            "input_text_prefixes": [["review: "], ["text: "], ["sentence: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("cornell-movie-review-data/rotten_tomatoes")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/rotten_tomatoes.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/rotten_tomatoes.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()
        

class financial_phrasebank(basic_datasets_loader):
    # https://arxiv.org/abs/1307.5336
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["sentence: "]
        self._label_space = ['negative', 'neutral', 'positive']
        self._label_prefix = "sentiment: "
        self._label_mapping = {0:0, 1:1, 2:2} 
        self.dataset_name = "financial_phrasebank" 
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "What is the attitude towards the financial news in this sentence? ", "What is the emotional response to the financial news in this sentence? "],
            "input_text_prefixes": [["sentence: "], ["text: "], ["news: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("financial_phrasebank", "sentences_allagree")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/financial_phrasebank.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/financial_phrasebank.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["sentence"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class sst5(basic_datasets_loader):
    # https://aclanthology.org/D13-1170/
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["sentence: "]
        self._label_space = None
        self._ground_truth_label_space = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        self._reducted_label_space = ['poor', 'bad', 'neutral', 'good', 'great']
        self._label_mapping = {0:0, 1:1, 2:2, 3:3, 4:4} 
        self._label_prefix = "sentiment: "
        self.dataset_name = "SST5" 
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1
        self.reduct_label_token()
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "How would you describe the overall feeling of the movie based on this sentence? ", "What mood does this sentence convey about the movie? "],
            "input_text_prefixes": [["sentence: "], ["text: "], ["review: "]],
            "label_prefix": ["sentiment: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("SetFit/sst5", "sentences_allagree")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/sst5.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/sst5.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()

# 2 Topic Classification Datasets
class trec(basic_datasets_loader):
    # https://www.aclweb.org/anthology/C02-1150
    # https://www.aclweb.org/anthology/H01-1069
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["question: "]
        self._label_space = None
        self._ground_truth_label_space = ['abbreviation', 'entity', 'description and abstract concept', 'human being', 'location', 'numeric value']
        self._reducted_label_space = ['short', 'entity', 'description', 'person', 'location', 'number'] # https://arxiv.org/pdf/2305.19148
        self._label_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5} 
        self._label_prefix = "target: "
        self.dataset_name = "TREC" 
        self.reduct_label_token()
        self.label_space_numbers = len(self._label_space)
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1

        self.alternate_template = {
            "instruction": ["", "What is the topic of the question? ", "What is the primary focus of this question? "],
            "input_text_prefixes": [["question: "], ["text: "], ["sentence: "]],
            "label_prefix": ["target: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("CogComp/trec")
            self._hgf_dataset = datasets.concatenate_datasets([self._hgf_dataset['train'], self._hgf_dataset['test']])
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/trec.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/trec.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["coarse_label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class agnews(basic_datasets_loader):
    # https://arxiv.org/abs/1509.01626
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["news: "]
        
        self._label_space = None
        self._ground_truth_label_space = ['world', 'sports', 'business', 'sci/tech']
        self._reducted_label_space = ['world', 'sports', 'business', 'science']
        self._label_mapping = {0:0, 1:1, 2:2, 3:3} 
        self._label_prefix = "topic: "
        self.dataset_name = "AGNews" 
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1
        self.reduct_label_token()
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "What is the topic of the news? ", "What is the news focused on? "],
            "input_text_prefixes": [["news: "], ["text: "], ["sentence: "]],
            "label_prefix": ["topic: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("ag_news")
            self._hgf_dataset = datasets.concatenate_datasets([self._hgf_dataset['train'], self._hgf_dataset['test']])
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/agnews.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/agnews.dataset')
            self.table = pickle.loads(pickle_file)
            
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class subjective(basic_datasets_loader):
    # https://dl.acm.org/doi/10.5555/2390665.2390688
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["review: "]
        self._label_space = None
        self._ground_truth_label_space = ['objective', 'subjective']
        self._reducted_label_space = ['false', 'true']
        self._label_mapping = {0:0, 1:1} 
        self._label_prefix = "subjectiveness: "
        self.dataset_name = "Subjective" 
        self._long_text_classification = long_text_classification
        self.reduct_label_token()
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "Does this sentence reflect a personal opinion? ", "Is this sentence expressing a personal opinion or stating a fact? "],
            "input_text_prefixes": [["review: "], ["text: "], ["sentence: "]],
            "label_prefix": ["subjectiveness: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("SetFit/subj")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/subjective.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/subjective.dataset')
            self.table = pickle.loads(pickle_file)
    
    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset
        self._automatic_cut_by_length()
        self._shuffle()
        


class tweet_eval_emotion(basic_datasets_loader):
    # https://aclanthology.org/S18-1001/
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["tweet: "]
        
        self._label_space = None
        self._ground_truth_label_space = ['anger', 'joy', 'optimism', 'sadness']
        self._reducted_label_space = ['anger', 'joy', 'positive', 'sad']
        self._label_mapping = {0:0, 1:1, 2:2, 3:3} 
        self._label_prefix = "emotion: "
        self.dataset_name = "tweet_eval_emotion" 
        self._long_text_classification = long_text_classification
        self.reduct_label_token()
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "What feeling does this sentence convey? ", "What emotion does this sentence express? "],
            "input_text_prefixes": [["tweet: "], ["text: "], ["sentence: "]],
            "label_prefix": ["emotion: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("tweet_eval", "emotion")
            self._hgf_dataset = datasets.concatenate_datasets([self._hgf_dataset['train'], self._hgf_dataset['validation'], self._hgf_dataset['test']])
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/tweet_eval_emotion.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/tweet_eval_emotion.dataset')
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class tweet_eval_hate(basic_datasets_loader):
    # https://aclanthology.org/S19-2007/
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["tweet: "]
        self._label_space = None
        self._ground_truth_label_space = ['non-hate', 'hate']
        self._reducted_label_space = ['normal', 'hate']
        self._label_mapping = {0:0, 1:1} 
        self._label_prefix = "hate speech: "
        self.dataset_name = "tweet_eval_hate" 
        self._long_text_classification = long_text_classification
        self.reduct_label_token()
        self.input_element_numbers = 1
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "Does this sentence contain hate speech? ", "Is this sentence an example of hate speech? "],
            "input_text_prefixes": [["tweet: "], ["text: "], ["sentence: "]],
            "label_prefix": ["hate speech: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("tweet_eval", "hate")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/tweet_eval_hate.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/tweet_eval_hate.dataset')
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class hate_speech_18(basic_datasets_loader):
    #
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__()

        self._input_text_prefixes = ["tweet: "]
        
        self._label_space = None
        self._ground_truth_label_space = ["noHate", "hate", "idk/skip", "relation"]
        self._reducted_label_space = ['normal', 'hate', 'skip', 'relation']
        self._label_mapping = {0:0, 1:1, 2:2, 3:3} 
        self._label_prefix = "hate speech: "
        self.dataset_name = "hate_speech_18" 
        self._long_text_classification = long_text_classification
        self.input_element_numbers = 1
        self.reduct_label_token()
        self.label_space_numbers = len(self._label_space)

        self.alternate_template = {
            "instruction": ["", "Does this sentence contain hate speech? ", "Is this sentence an example of hate speech? "],
            "input_text_prefixes": [["tweet: "], ["text: "], ["sentence: "]],
            "label_prefix": ["hate speech: ", "label: ", "Label: "],
            "label_affix": ["\n", " ", "\t"],
        }

        if not from_cache:
            import datasets
            self._hgf_dataset = datasets.load_dataset("hate_speech18")['train']
            self._complie_dataset()
        else:
            # with open("./StaICC/cached_dataset/hate_speech_18.dataset", "rb") as pickle_file:
            pickle_file = pkgutil.get_data(self._package_path, 'cached_dataset/hate_speech_18.dataset')
            self.table = pickle.loads(pickle_file)

    def _complie_dataset(self):
        self.table = []
        for i in range(0, len(self._hgf_dataset)):
            self.table.append(([self._hgf_dataset[i]["text"]], self._hgf_dataset[i]["label"]))
        del self._hgf_dataset

        self._automatic_cut_by_length()
        self._shuffle()


class un_reducted_trec(trec):
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__(long_text_classification, from_cache)
        self.full_label_token()


class un_reducted_agnews(agnews):
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__(long_text_classification, from_cache)
        self.full_label_token()


class un_reducted_tweet_eval_emotion(tweet_eval_emotion):
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__(long_text_classification, from_cache)
        self.full_label_token()


class un_reducted_sst5(sst5):
    def __init__(self, long_text_classification = False, from_cache = True):
        super().__init__(long_text_classification, from_cache)
        self.full_label_token()