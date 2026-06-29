import copy
from . import configs
import warnings

class stable_random():
    def __init__(self, seed=configs.STANDARD_SETTINGS["random_seed"]):
        if seed != configs.STANDARD_SETTINGS["random_seed"]:
            warnings.warn(configs.WARNING_SETTINGS["tampering"])
        self._seed = seed
        self._current_X = seed

    def _next(self):
        self._current_X = (self._current_X * configs.STANDARD_SETTINGS["random_A"] + configs.STANDARD_SETTINGS["random_B"]) % configs.STANDARD_SETTINGS["random_C"]
        return self._current_X
    
    def get_float(self):
        return self._next() / configs.STANDARD_SETTINGS["random_C"]
    
    def get_int_from_range(self, start, end):
        return int(start + (end - start) * self.get_float())
    
    def sample_one_element_from_list(self, list):
        return list[int(self.get_float() * len(list))]
    
    def sample_n_elements_from_list(self, list, n, allow_repetition=False):
        if allow_repetition == False and n > len(list):
            raise ValueError("n should be less than the length of the list")
        list_copy = copy.deepcopy(list)
        ret = []
        for _ in range(n):
            loca = int(self.get_float() * len(list_copy))
            ret.append(list_copy[loca])
            if not allow_repetition:
                list_copy.pop(loca)
        return ret
    
    def sample_index_set(self, sample_number, max_index, allow_repetition=False):
        if allow_repetition == False and sample_number > max_index:
            raise ValueError("sample_number should be less than max_index")
        index_list = list(range(max_index))
        return self.sample_n_elements_from_list(index_list, sample_number, allow_repetition)
    
    def shuffle_list(self, list):
        return self.sample_n_elements_from_list(list, len(list), allow_repetition=False)