# StaICC: Standardized Toolkit for In-context Classification

<p align="center">
  <a href="https://www.hakaze-c.com/">Hakaze Cho</a>, et al.
  <br>
  <br>
  <a href="https://github.com/hc495/StaICC/blob/master/LICENSE"><img alt="Static Badge" src="https://img.shields.io/badge/license-Apache--2.0-yellow?style=flat"></a>
  <a href="https://arxiv.org/abs/2501.15708"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2501.15708-red?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2501.15708"></a>
</p>

This is a standardized toolkit for in-context classification by Hakaze Cho (Yufeng Zhao), descirbed in paper [StaICC: Standardized Evaluation for Classification Task in In-context Learning](https://arxiv.org/abs/2501.15708).

## Content

1. [Installation](#installation)
2. [Introduction](#introduction)
3. [Quick Start](#quick-start)
4. [Custom Experiment](#custom-experiment)
5. [Detailed Documentation](#detailed-documentation)

## Installation

We ensure that under normal usage, this library only relies on Python's default dependency libraries.

You need only to download a release pack of StaICC and unfold it into your work path with the top folder `StaICC` in your work path, like:

```
--- work_path
 |--- StaICC
 | |-- __init__.py
 | |-- ...
 |--- experiment_code.py
 |--- ...
```

Also, we release PyPl package `StaICC`. You can use:

```
pip install StaICC
```

to install this library.

## Introduction

`StaICC` is a standardized benchmark for in-context classification tasks. It provides a unified interface for in-context classification tasks, including the following components:

### Sub-benchmarks

`StaICC` provides several sub-benchmarks for in-context classification evaluations. Please refer to the [Sub-benchmarks](#sub-benchmarks) section for details of usage. The following table lists the sub-benchmarks we provide:

| Name | Import name | Describe |
|:---:|:---:|:---:|
| StaICC-Normal | `from StaICC import Normal` | A standard classification accuracy-based benchmark for normal classification tasks. |
| StaICC-Diagnosis: Bias | `from StaICC import Triplet_bias` | A prediction logits bias (3 types) detector. |
| StaICC-Diagnosis: Noise Sensitivity | `from StaICC import GLER` | A demonstration label noise sensitivity detector. |
| StaICC-Diagnosis: Template Sensitivity | `from StaICC import Template_sens` | A template sensitivity detector against 9 prompt templates. |
| StaICC-Diagnosis: Demonstration Sensitivity| `from StaICC import Demo_sens` | A demonstration sensitivity detector against the given demonstraions in the context. |

#### StaICC-Normal

`StaICC-Normal` is a standard classification accuracy-based benchmark for normal classification tasks. It returns the averaged metrics of accuracy, averaged truelabel likelihood, macro F1, and expected calibration error.

#### StaICC-Diagnosis: Bias

`StaICC-Diagnosis: Bias` is a prediction logits bias detector of 3 types:

1. **Contextual Bias**: Introduced by [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2106.06328), contextual bias measures the bias when some demonstrations and an empty query is fed into the model. We use the entropy of the averaged prediction probabilites as the metric.

2. **Domainal Bias**: Introduced by [Mitigating Label Biases for In-context Learning](http://arxiv.org/abs/2305.19148), domainal bias measures the bias when some demonstrations and a query of randomly sampled tokens from the test dataset is fed into the model. We use the entropy of the averaged prediction probabilites as the metric.

3. **Posterior Bias**: Measures the bias (we use KL divergence for the metric) from the predicted probability to the frequency of the ground-truth label.

You can use them dividedly by `from StaICC import Contextual_bias, Domain_bias, Post_bias`.

#### StaICC-Diagnosis: Noise Sensitivity

`StaICC-Diagnosis: Noise Sensitivity` is a noise sensitivity detector of the correlation of accuracy against demonstration label noise. It uses the Generalized Label Error Rate (GLER) as the metric. The GLER is defined in [Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations](http://arxiv.org/abs/2205.12685), which is the slope of the curve of the prediction probability against the label correctness in the demonstration. Larger GLER indicates higher noise sensitivity.

#### StaICC-Diagnosis: Template Sensitivity

`StaICC-Diagnosis: Template Sensitivity` is a template sensitivity detector against 9 prompt templates. It uses the prediction consistency as a negative metric to the sensitivity: for one set of demonstraions and query, we make up 9 prompts with different templates, and calculate the prediction consistency of the model. Lower prediction consistency indicates higher template sensitivity.

#### StaICC-Diagnosis: Demonstration Sensitivity

`StaICC-Diagnosis: Demonstration Sensitivity` is a demonstration sensitivity detector against 8 demonstraions sets for each query. It uses the prediction consistency as a negative metric to the sensitivity: for one query, we make up 8 prompts with different set of demonstraions, and calculate the prediction consistency of the model. Lower prediction consistency indicates higher template sensitivity.

### Datasets

In StaICC, we use the following original datasets:

|Index| Name | Task | Label Space | Citation |
|:---:|:---:|:---:|:---:|:---:|
|0|GLUE-SST2 | Sentiment Classification | negative, positive | https://aclanthology.org/D13-1170/|
|1|Rotten Tomatoes | Sentiment Classification | negative, positive | https://arxiv.org/abs/cs/0506075|
|2|Financial Phrasebank | Sentiment Classification | negative, neutral, positive | https://arxiv.org/abs/1307.5336|
|3|SST5 | Sentiment Classification | very negative, negative, neutral, positive, very positive | https://aclanthology.org/D13-1170/|
|4|TREC | Topic Classification | abbreviation, entity, description and abstract concept, human being, location, numeric value | https://www.aclweb.org/anthology/C02-1150|
|5|AGNews | Topic Classification | world, sports, business, sci/tech | https://arxiv.org/abs/1509.01626|
|6|Subjective | Subjectivity Classification | objective, subjective|https://dl.acm.org/doi/10.5555/2390665.2390688|
|7|Tweet Eval Emotion | Sentiment Classification | anger, joy, optimism, sadness|https://aclanthology.org/S18-1001/|
|8|Tweet Eval Hate |Hate Speech Classification | non-hate, hate|https://aclanthology.org/S19-2007/|
|9|Hate Speech 18|Hate Speech Classification| noHate, hate, idk/skip, relation| https://huggingface.co/datasets/odegiber/hate_speech18 |

## Quick Start

A standard process of the usage of StaICC is shown as below.

### 1. Write your ICL inference

You should write a function or partial function with a prototype `my_function(prompt: str, label_space: list[str]) -> Union[list[float], int]`. Make sure the name of the formal parameter is consistent with the above. __Typically__, the parameter `prompt` is fed with a `str` variable with a ICL-formatted string, and the `label_space` is fed with a `list[str]` to describe which token in the vocabulary should the model focus as the label. The return value should be a `list[float]` or `int` to describe the prediction probability / logits (if you pass a logits, we will calculate softmax) or prediction label, aligned with the `label_space`.

You can refer to the functions in `prefabricate_inference/model_kernel.py` as examples. Also, as a quick start, you can reload these functions by `functools.partial` as shown below. (if you import `StaICC.prefabricate_inference.model_kernel`, make sure you have dependencies of `torch` and `transformers >= 4.43`)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from StaICC.prefabricate_inference import model_kernel
import functools

tokenizer = AutoTokenizer.from_pretrained("<huggingface_model_name>") 
model = AutoModelForCausalLM.from_pretrained("<huggingface_model_name>").cuda()

my_inference = functools.partial(
    model_kernel.standard_ICL_inference_with_torch_Causal_LM, 
    model = model, 
    tokenizer = tokenizer, 
    cache_empty = torch.cuda.empty_cache, 
    return_hidden_state = False, 
    return_full_vocab_prob = False
) 
```

### 2. Load a sub-benchmark and instantiate it

Choose one sub-benchmark introduced in [Introduction](#introduction). As a start, you can choose `StaICC-Normal` as a trial:

```python
from StaICC import Normal
benchmark = Normal()
```

### 3. Test your inference function

```python
result_dictionary = benchmark(my_inference)
```

A typical output is a dictionary as:

```python
'Divided results': {
    'GLUE-SST2': {
        'accuracy': 0.580078125,
        'averaged_truelabel_likelihood': 0.5380006989248597,
        'macro_F1': 0.49802777081100796,
        'expected_calibration_error_1': 0.08539132766414247},
    'rotten_tomatoes': {
        'accuracy': 0.5693359375,
        'averaged_truelabel_likelihood': 0.5196389624283041,
        'macro_F1': 0.5286056525483442,
        'expected_calibration_error_1': 0.03482341199255428},},
    ...
'Averaged results': {
    'accuracy': 0.40244140625,
    'averaged_truelabel_likelihood': 0.38355543726209307,
    'macro_F1': 0.2863139242653286,
    'expected_calibration_error_1': 0.33533775010105704}
```

## Custom Experiment

In our implementation, we use one `experimentor` object for each dataset. So, you can customize your experiment by setting the parameters of the `experimentor` object. One you load a sub-benchmark like `Normal`, you can access the `experimentor` object by `benchmark[dataset_index]`. For example, you can access the `experimentor` object of the `GLUE-SST2` dataset by `Normal[0]`.

When you access the `experimentor`, you can control the experiment. We support the following custom settings of experiment with respect to each step in the ICL pipeline:

### Demonstration Sampling

The demonstration sampling is controled by `experimentor.demonstration_sampler`, which is a `list[list[int]]` shaped object. Each element in the `experimentor.demonstration_sampler` is a list of indices of the demonstrations sequence assigned for the corresponding test sample.

#### Various Demonstration Numbers

You are recommended to define the demonstration number in the initialization of the sub-benchmark. The default value is 4, and you can set the `k` parameter in the instantiation of the sub-benchmark.

For example:

```python
from StaICC import Normal
benchmark = Normal(k = 16)
```

- We didn't set a upper limit for this parameter, but some of the demonstrations may be repeated if the `k` exceed the existing number of the demonstration samples.

Also, you can set the expected demonstration number after the initialization by `experimentor.set_k(k)`.

#### Manual Demonstration Sampling

You can use `experimentor.set_demonstration_sampler(sampler)` function to manually sample the demonstrations for each test sample. You can input any list-styled object `sampler` with the same length as the test samples, and each element in your `sampler` should be a list of indices of the demonstrations you want to use for the corresponding test sample, in sequence.

In this processing, you are likely to need access to these demonstration and test sets. You can access them by `experimentor.demonstration_set()` and `experimentor.test_set()`. 

**Tips**

- You must align the `len(sampler)` with the length of the test set `len(experimentor.test_set())`.
- Notice that setting a `sampler` will set the repeat experiment times from 2 to 1.
- To reset the `sampler` to default, you can call `experimentor.reset_demonstration_sampler()`.

**An example:** we repeat the k-NN demonstration experiment proposed by paper [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804). The full code are shown in file `prefabricate_inference/prompt_template_edit.py`, and the key part about the manual demonstration is shown below:

```python
class SA_ICL():
...
    def _encode_demonstrations(self):
        count = 0
        for demo in self.experimentor.demonstration_set():
            if len(demo[0]) == 0:
                continue
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
                    )[-1] # Get the last hidden state as the encoding vector.
                )
            except:
                continue
        self.TopK_anchors = np.array(self.TopK_anchors)

    def _get_top_k_indexes(self, test_sample, k):
        self._encode_demonstrations()
        distance = []
        test_sample_encoded = model_kernel.standard_ICL_inference_with_torch_Causal_LM(
            prompt = test_sample, 
            model = self.model, 
            tokenizer = self.tokenizer, 
            label_space = self.label_space, 
            cache_empty = self.cache_empty,
            calibration_function = None,
            return_hidden_state = True
        )[-1] # Get the last hidden state as the encoding vector.

        # Calculate the distance between the test sample and each anchor (encoded demonstration samples by _encode_demonstrations).
        for anchor in self.TopK_anchors:
            distance.append(np.linalg.norm(test_sample_encoded - np.array(anchor)))
        ret = []
        for _ in range(k):
            ret.append(functional.argmin(distance))
            distance[functional.argmin(distance)] = 1e10
        return ret

    def set_TopK_to_demonstration(self, k):
        demonstration_sampler = []
        for i in range(len(self.experimentor.test_set())):
            demonstration_sampler.append(self._get_top_k_indexes(self.experimentor.test_set()[i][0], k))
        # We operate the demonstration sampler by the following line.
        self.experimentor.set_demonstration_sampler(demonstration_sampler)
```

### Prompt Template Editing

The ICL prompt assembly is controlled by `experimentor.prompt_former`. `prompt_former` has the following members to control the prompt template:

- `prompt_former._instruction`: the instruction at the beginning of the prompt. Type: `str`. Can be adjusted by `prompt_former.change_instruction(new_instruction: str)`.
- `prompt_former._input_text_prefixes`: the prefixes of the input text. Type: `list[str]`, the length should be the same as the number of the input text (for example, 1 for the SST-2, and 2 for the RTE). Can be adjusted by `prompt_former.change_input_text_prefixes(new_prefixes: list[str])`.
- `prompt_former._input_text_affixes`: the affixes of the input text. Type: `list[str]`, the length should be the same as the number of the input text (for example, 1 for the SST-2, and 2 for the RTE). Can be adjusted by `prompt_former.change_input_text_affixes(new_affixes: list[str])`.
- `prompt_former._label_prefix`: the prefix of the label. Type: `str`. Can be adjusted by `prompt_former.change_label_prefix(new_prefix: str)`.
- `prompt_former._label_affix`: the affix of the label. Type: `str`. Can be adjusted by `prompt_former.change_label_affix(new_affix: str)`.
- `prompt_former._query_prefix`: the prefix of the query. Type: `str`. Notice: after the query_prefix, we still add `_input_text_affixes[0]`. Can be adjusted by `prompt_former.change_query_prefix(new_prefix: str)`.
- `prompt_former._label_space`: the label space of the dataset. Type: `list[str]`. Can be adjusted by `prompt_former.change_label_space(new_label_space: list[str])`. Notice: this change of label space also reflects to the input to the inference function.
- `prompt_former._label_wrong_rate`: the rate of the wrong label in the demonstrations. Type: `float`. Can be adjusted by `prompt_former.change_label_wrong_rate(new_rate: float)`.
- `prompt_former._use_noisy_channel`: whether to use the noisy channel inference. Type: `bool`. Can be enabled by `prompt_former.use_noisy_channel()`, and disabled by `prompt_former.disable_noisy_channel()`.

And the prompt will be generated like:
```
(notice that all the '\n', '[ ]' and ' ' shown as the format here are not default, you should add it if you want to split the instruction)

<prompt_former.instruction> 
[
  <prompt_former.input_text_prefixes[0]> 
  <prompt_former.triplet_dataset.demonstration.get_input_text(index)
  <prompt_former.input_text_prefixes[0]>

  <prompt_former.input_text_prefixes[1]> 
  <prompt_former.triplet_dataset.demonstration.get_input_text(index)
  <prompt_former.input_text_prefixes[1]>
  ...
  <prompt_former.label_prefix> 
  <prompt_former.label(index)> 
  <prompt_former.label_afffix>
] * k (k = demostration numbers)
<prompt_former.query_prefix>
[
  <prompt_former.input_text_prefixes[0]> 
  <prompt_former.triplet_dataset.test.get_input_text(index)>
  <prompt_former.input_text_prefixes[0]>

  <prompt_former.input_text_prefixes[1]> 
  <prompt_former.triplet_dataset.test.get_input_text(index)>
  <prompt_former.input_text_prefixes[1]>
  ...
  <prompt_former.label_prefix> [MASKED]
]
```

You can also use `set_config_dict(config_dict)` to set the prompt template by a dictionary, and load the current setting by `get_config_dict()`. The dictionary should have the following keys:

``` python
{
    'instruction': str,
    'input_text_prefixes': list[str],
    'input_text_affixes': list[str],
    'label_prefix': str,
    'label_affix': str,
    'query_prefix': str,
    'label_space': list[str],
    'label_wrong_rate': float,
    'use_noisy_channel': bool
}
```

Notice that you can not use the dictionary to save all the status of the `prompt_former`, only the above keys are valid.

**Tips**

- You can call `prompt_former.example()` to observe an example of the prompt.
- These templates are defaultly defined in the `hgf_dataset_loader.py`. Call `prompt_former.reset()` to reset the prompt template to default.

### Custom Inference

You can use a custom inference function for each dataset, which can be set by the inference function `experimentor.auto_run(forward_inference = my_inference)` where the `forward_inference` should be a function with the prototype `forward_inference(prompt: str, label_space: list[str]) -> Union[list[float], int]`; or `<sub-benchmark>.auto_run(list_of_forward_inference = my_inferences)`, where the `list_of_forward_inference` should be a list of functions with the prototype `forward_inference(prompt: str, label_space: list[str]) -> Union[list[float], int]`, with index aligned with the dataset index.

If you wish to perform any **preprocessing** on the inference function, such as learning a calibration, you should complete this process in advance (note: we have prepared some additional data for such preprocessing in `experimentor.calibration_set()`, it is a standard [`basic_datasets_loader` object](#dataset_loader)) while maintaining the function interface as described above. There are only two exceptions: (descirbed below) 1. You intend to use a batched inference process, providing an input list and requesting an output list. 2. You wish to directly evaluate existing prediction values.

#### Batched Inference

If you want to use a batched inference process, you can set `batched_inference=True` in the `auto_run` function. The prototype of the batched inference function should be `batched_inference(prompts: list[str], label_space: list[str]) -> list[list[float]]` or `batched_inference(prompts: list[str], label_space: list[str]) -> list[int]`. An example with [Batch Calibration](https://arxiv.org/abs/2309.17249) is shown in `examples/batched_inference.ipynb`.

#### Preentered Prediction

If you already have all the inference results (`list[list[float]]` for probabilites / logits, or `list[int]` for label index) aligned with the `experimentor.prompt_set()`, you can directly input them by the `preentered_prediction`, a `list[list[float]]` object to store the pre-entered prediction of the model. The shape should be `(len(experimentor.prompt_set()), len(get_label_space()))`. When you use `preentered_prediction`, `forward_inference` will be ignored.

#### Calibration

You can train a calibration function above the normal output of LMs, by the remained `experimentor.calibration_set()` and set it to the inference function. We have some standard calibration functions in `StaICC.prefabricate_inference.standard_calibration`, and the `model_kernel.standard_ICL_inference_with_torch_Causal_LM` can be adopt to these calibration functions. An example with [Hidden Calibration](https://arxiv.org/abs/2406.16535) is shown in `examples/calibration.ipynb`.

#### Noisy Channel Inference

Noisy Channel use a resevered prompt like `<label><input_text><label><input_text>...` as the input. 

Simply, noisy channel inference is a method to build a reversed prompt for each label candidate like:

```
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
```

Refer to [Noisy Channel Language Model Prompting for Few-Shot Text Classification](https://aclanthology.org/2022.acl-long.365/) for details. 

To use this inference, you should set the `noisy_channel = True` when you load the benchmark. For example,

```python
from StaICC import Normal
benchmark = Normal(k = 16, noisy_channel = True)
```

One more example is shown in `examples/noisy_channel.ipynb`.

<!-- ## Examples

More examples are shown in the `examples` folder.

<span id="prompt_sample"></span>
### Example 1: Use manual demonstration sequence in your experiment

### Example 2: Use different inference function for each dataset

 -->

### Custom Metric

You can simply set the `return_outputs=True` in the `auto_run` function to return the direct outputs of the inference function, then conduct your own metric calculation. Or, you can add your metric, which should be shaped like `metric(ground_truth: list[int], prediction: list[list[float]]) -> float`, by the `experimentor.add_metric(name: str, metric: Callable[ground_truth: list[int], prediction: list[list[float]]])` function.

**Tips**

- You should not customlize your metric on the StaICC-Diagnosis benchmarks, the metrics here should be predefined.

## Detailed Documentation

**Bottom Dataset Loader and Interface**

<span id="dataset_loader"></span>
### `basic_datasets_loader` class

The `basic_datasets_loader` class is a class to load the dataset and define the inference behavior on these dataset. Generally, you should not access this class directly. It is recommended to access it through the `*_set()` functions of `single_experimentor`.

Notice: if you access this class from `*_set()` functions of `single_experimentor`, you should be only care about the following functions:

#### `__getitem__(index: int) -> Tuple[list[str], int]`

Get the input texts (notice that for multi-input task, the input texts can be multiple, so we use `list[str]`) and the label of one data indexed by parameter `index`. The return value is a tuple of the input texts and the label index.

#### `label_index_to_text(index: int) -> str`

Transfer label index to label text.

#### `split(split_indexes: list[list[int]]) -> list[basic_datasets_loader]`

Given the list of indexes list of the splits, return the list of `basic_datasets_loader` objects of the splits.

Control the split numbers by the `len(split_indexes)`, and control the split size by the `len(split_indexes[i])`, and enumerate the element index in each split by the `split_indexes[i][j]`.

### `triplet_dataset` class

The `triplet_dataset` class is a class to load the dataset and divide it into demonstraion set, calibration set and test set. `triplet_dataset` divide one `basic_datasets_loader` object into three parts: `demonstration_set`, `calibration_set`, and `test_set`, all the 3 are new `basic_datasets_loader` return from `basic_datasets_loader.split()`.

Also, this class should be hide from the users.

### `demonstration_sampler` class

The `demonstration_sampler` class is a `list[list[int]]`-like class with stable random sampling to control the demonstration sampling process. Typically, for each query indexed with `i`, the `demonstration_sampler[i]: list[int]` is a list of indices of the demonstrations sequence assigned for the corresponding test sample.

Also, you should not access this class directly. If you want to set your own sample list, you should use the `experimentor.set_demonstration_sampler` to set a `list[list[int]]`-like object to the experimentor.

### `prompt_writter` class

As described in [Custom Experiment](#custom-experiment), the `prompt_writter` class is a class to control the prompt template. You can access the `prompt_writter` object by the `experimentor.prompt_former`. The `prompt_writter` object has the following members to control the prompt template:

#### `reset() -> None`

Reset the `prompt_writter` settings to default.

#### `set_label_wrong_rate(rate: float) -> None`

Set the rate of the wrong label in the demonstrations. The parameter `rate` is a `float` object, and should be in the range of `[0, 1]`. `0` means no wrong label, and `1` means all the labels are wrong.

#### `use_noisy_channel() -> None`

Enable the noisy channel prompting. Do not use this function if you do not know what it is. If you want to use the noisy channel inference, you should set the `noisy_channel = True` when you load the benchmark. See [Custom Experiment](#custom-experiment) for details.

Notice that this function will reload the `label_afffix` and `input_text_affixes[-1]` to the noisy channel format, since the major spiltor between the demonstrations in the noisy channel mode is the `input_text_affixes[-1]`, instead the `label_afffix`.

#### `cancel_noisy_channel() -> None`

Disable the noisy channel prompting.

#### `get_config_dict() -> dict`; `set_config_dict(config_dict: dict) -> None`

For convenience's sake, you can set the prompt template by the `set_config_dict(config_dict)` function, and load the current setting by the `get_config_dict()` function. The dictionary have the following keys, but you only need to set the keys you want to change:

``` python
{
    'instruction': str,
    'input_text_prefixes': list[str],
    'input_text_affixes': list[str],
    'label_prefix': str,
    'label_affix': str,
    'query_prefix': str,
    'label_space': list[str],
    'label_wrong_rate': float,
    'use_noisy_channel': bool
}
```

While, you can also use the individual functions to set the prompt template as described in [Custom Experiment](#custom-experiment).

#### `replace_space_to_label(label: str) -> str`

In some cases, you may want to use label space like `[' positive', ' negative']` with a space in the head, instead of `['positive', 'negative']`, since some tokenizer may treat them differently, and the `label_space` here will be fed into the inference function defaultly. You can use this function to replace the space in the tail of `label_prefix` to the head of `label_space`.

Notice that use this function will firstly reset the prompt template to the default. You should not use this function if you do not know what it is.

#### `write_prompt(demos_indexes, query_index) -> str`

Access the `triplet_dataset`, fetch the required demonstrations and query in the parameters, and write the prompt. The `demos_indexes` is a sequence of indices of the demonstrations assigned for the corresponding test sample, and the `query_index` is the index of the test sample.

The `query_index` can be a `None` when `self.pseudo_prompt` is defined, to produce a prompt with a pseudo query.

#### `write_prompt_from_dataline(demos_lines: list[(list[str], str)], query_line: list[str], cut_by_length = 0) -> str`

Write the prompt from the data (string). The `demos_lines` is a list of the demonstrations strings, formatted as `[(demonstration inputs: list[str], label: str) * k]` and the `query_line` is the query. Notice that we support the multi-input task, so the input text object is `list[str]`. The `cut_by_length` is a `int` object to cut the prompt by the string length.

**Experimenor and Benchmark**

### `single_experimentor` class

As the basic module of StaICC, the `single_experimentor` class is a class to control the experiment process of a single dataset. 

#### `add_metric(name: str, metric: Callable[ground_truth: list[int], prediction: list[list[float]]])`

Add a metric to the experiment. The parameter `name` is the name of the metric, and the parameter `metric` is a callable object with the prototype `metric(ground_truth: list[int], prediction: list[list[float]]) -> float`.

#### `set_k(k: int)`

Set the expected demonstration number for each test sample. The parameter `k` is the expected demonstration number.

#### `get_k() -> int`

Get the expected demonstration number `k`.

#### `get_repeat_times() -> int`

Get the repeat times for each test samples. Default is 2.

#### `set_out_of_domain_mode() -> None`

Resample the demonstrations for each test sample to make sure the ground-truth label of the test sample is not in the demonstrations. 

#### `set_in_domain_mode() -> None`

Resample the demonstrations for each test sample to make sure the ground-truth label of the test sample is in the demonstrations.

#### `set_demonstration_sampler(sampler: list[list[int]]) -> None`

Set the demonstration sampler for each test sample. The parameter `sampler` is a `list[list[int]]` object. Each element in the `sampler` is a list of indices of the demonstrations sequence assigned for the corresponding test sample.

#### `reset_demonstration_sampler() -> None`

Reset the demonstration sampler to default.

Will cancel the `set_out_of_domain_mode()`, `set_in_domain_mode()`, and `set_demonstration_sampler(sampler)`.

#### `demonstration_set() -> basic_datasets_loader`

Return the demonstration set of the dataset as a `basic_datasets_loader` object.

#### `test_set() -> basic_datasets_loader`

Return the test set of the dataset as a `basic_datasets_loader` object.

#### `calibration_set() -> basic_datasets_loader`

Return the calibration set of the dataset as a `basic_datasets_loader` object.

#### `prompt_set() -> list[str]`

Return the full prompt set to be input to the inference function.

#### `auto_run(forward_inference = None, preentered_prediction = None, batched_inference = False, return_outputs = False) -> dict`

Run the experiment with the given inference function. Also override the `__call__` method. The parameters are:

**You should provide either `forward_inference` or `preentered_prediction`.**

- `forward_inference`: The inference function to be used in the experiment. Basically (without `batched_inference`), the prototype of the inference function should be `forward_inference(prompt, label_space) -> Union[list[float], int]`. An example is shown in [Quick Start](#quick-start).
- `preentered_prediction`: If you already have all the inference results (`list[list[float]]` or `list[int]`) aligned with the `experimentor.prompt_set()`, you can directly input them by the `preentered_prediction`, a `list[list[float]]` object to store the pre-entered prediction of the model. The shape should be `(len(experimentor.prompt_set()), len(get_label_space()))`.
- `batched_inference`: If you want to use a batched inference process, you can set `batched_inference=True`. The prototype of the batched inference function should be `batched_inference(prompts: list[str], label_space: list[str]) -> list[list[float]]` or `batched_inference(prompts: list[str], label_space: list[str]) -> list[int]`.
- `return_outputs`: If you want to return the direct outputs of the inference function, you can set `return_outputs=True`. The outputs will be stored in the `outputs` field of the return dictionary.

The return value is a 2- or 3-turple, as: `(result_dictionary, success_indicator, direct_outputs)`.
- `result_dictionary`: The dictionary of the metric results.
    Defaultly, we provide the following metrics:
    - `accuracy`: The accuracy of the prediction.
    - `averaged_truelabel_likelihood`: The averaged likelihood of the ground-truth label. Only effective when the predicted probability is returned as the prediction.
    - `macro_F1`: The macro F1 score of the prediction.
    - `expected_calibration_error_1`: The expected calibration error of the prediction. Only effective when the predicted probability is returned as the prediction.
- `success_indicator`: A boolean value to indicate whether the experiment is successful. If the experiment is successful, the value is `True`, otherwise, the value is `False`.
- `direct_outputs`: The direct outputs of the inference function. Only returned when `return_outputs=True`. Formatted as a dictionary with keys: `ground_truth, predictions, predicted_probabilities`.

## Citation

If you find this work useful for your research, please cite [our paper](https://arxiv.org/abs/2501.15708):

```
@article{cho2025staicc,
  title={StaICC: Standardized Evaluation for Classification Task in In-context Learning},
  author={Cho, Hakaze and Inoue, Naoya},
  journal={arXiv preprint arXiv:2501.15708},
  year={2025}
}
```

Also, please cite all the original paper of the datasets used in this work.

```
@inproceedings{SST2andSST5,
  address = {Seattle, Washington, USA},
  author = {Socher, Richard  and
    Perelygin, Alex  and
    Wu, Jean  and
    Chuang, Jason  and
    Manning, Christopher D.  and
    Ng, Andrew  and
    Potts, Christopher},
  booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
  month = {October},
  pages = {1631--1642},
  publisher = {Association for Computational Linguistics},
  title = {Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank},
  url = {https://www.aclweb.org/anthology/D13-1170},
  year = {2013}
}

@inproceedings{MR,
  author = {Pang, Bo and Lee, Lillian},
  booktitle = {Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL'05)},
  pages = {115--124},
  title = {Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales},
  url = {http://arxiv.org/abs/cs/0506075v1},
  year = {2005}
}

@article{FP,
  author = {P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal = {Journal of the Association for Information Science and Technology},
  title = {Good debt or bad debt: Detecting semantic orientations in economic texts},
  url = {http://arxiv.org/abs/1307.5336v2},
  volume = {65},
  year = {2014}
}

@inproceedings{TREC1,
  author = {Li, Xin  and  Roth, Dan},
  booktitle = {{COLING} 2002: The 19th International Conference on Computational Linguistics},
  title = {Learning Question Classifiers},
  url = {https://www.aclweb.org/anthology/C02-1150},
  year = {2002}
}

@inproceedings{TREC2,
  author = {Hovy, Eduard  and
    Gerber, Laurie  and
    Hermjakob, Ulf  and
    Lin, Chin-Yew  and
    Ravichandran, Deepak},
  booktitle = {Proceedings of the First International Conference on Human Language Technology Research},
  title = {Toward Semantics-Based Answer Pinpointing},
  url = {https://www.aclweb.org/anthology/H01-1069},
  year = {2001}
}

@inproceedings{AGNews,
  author = {Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
  booktitle = {NIPS},
  title = {Character-level Convolutional Networks for Text Classification},
  url = {https://www.semanticscholar.org/paper/51a55df1f023571a7e07e338ee45a3e3d66ef73e},
  year = {2015}
}

@inproceedings{subjective,
  abstract = {Variants of Naive Bayes (NB) and Support Vector Machines (SVM) are often used as baseline methods for text classification, but their performance varies greatly depending on the model variant, features used and task/dataset. We show that: (i) the inclusion of word bigram features gives consistent gains on sentiment analysis tasks; (ii) for short snippet sentiment tasks, NB actually does better than SVMs (while for longer documents the opposite result holds); (iii) a simple but novel SVM variant using NB log-count ratios as feature values consistently performs well across tasks and datasets. Based on these observations, we identify simple NB and SVM variants which outperform most published results on sentiment analysis datasets, sometimes providing a new state-of-the-art performance level.},
  address = {USA},
  author = {Wang, Sida and Manning, Christopher D.},
  booktitle = {Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Short Papers - Volume 2},
  location = {Jeju Island, Korea},
  numpages = {5},
  pages = {90¨C94},
  publisher = {Association for Computational Linguistics},
  series = {ACL '12},
  title = {Baselines and bigrams: simple, good sentiment and topic classification},
  url = {https://www.semanticscholar.org/paper/5e9fa46f231c59e6573f9a116f77f53703347659},
  year = {2012}
}

@inproceedings{TEE,
  abstract = {We present the SemEval-2018 Task 1: Affect in Tweets, which includes an array of subtasks on inferring the affectual state of a person from their tweet. For each task, we created labeled data from English, Arabic, and Spanish tweets. The individual tasks are: 1. emotion intensity regression, 2. emotion intensity ordinal classification, 3. valence (sentiment) regression, 4. valence ordinal classification, and 5. emotion classification. Seventy-five teams (about 200 team members) participated in the shared task. We summarize the methods, resources, and tools used by the participating teams, with a focus on the techniques and resources that are particularly useful. We also analyze systems for consistent bias towards a particular race or gender. The data is made freely available to further improve our understanding of how people convey emotions through language.},
  address = {New Orleans, Louisiana},
  author = {Mohammad, Saif  and
    Bravo-Marquez, Felipe  and
    Salameh, Mohammad  and
    Kiritchenko, Svetlana},
  booktitle = {Proceedings of the 12th International Workshop on Semantic Evaluation},
  doi = {10.18653/v1/S18-1001},
  editor = {Apidianaki, Marianna  and
    Mohammad, Saif M.  and
    May, Jonathan  and
    Shutova, Ekaterina  and
    Bethard, Steven  and
    Carpuat, Marine},
  month = {June},
  pages = {1--17},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2018 Task 1: Affect in Tweets},
  url = {https://aclanthology.org/S18-1001/},
  year = {2018}
}

@inproceedings{TEH,
  abstract = {The paper describes the organization of the SemEval 2019 Task 5 about the detection of hate speech against immigrants and women in Spanish and English messages extracted from Twitter. The task is organized in two related classification subtasks: a main binary subtask for detecting the presence of hate speech, and a finer-grained one devoted to identifying further features in hateful contents such as the aggressive attitude and the target harassed, to distinguish if the incitement is against an individual rather than a group. HatEval has been one of the most popular tasks in SemEval-2019 with a total of 108 submitted runs for Subtask A and 70 runs for Subtask B, from a total of 74 different teams. Data provided for the task are described by showing how they have been collected and annotated. Moreover, the paper provides an analysis and discussion about the participant systems and the results they achieved in both subtasks.},
  address = {Minneapolis, Minnesota, USA},
  author = {Basile, Valerio  and
    Bosco, Cristina  and
    Fersini, Elisabetta  and
    Nozza, Debora  and
    Patti, Viviana  and
    Rangel Pardo, Francisco Manuel  and
    Rosso, Paolo  and
    Sanguinetti, Manuela},
  booktitle = {Proceedings of the 13th International Workshop on Semantic Evaluation},
  doi = {10.18653/v1/S19-2007},
  editor = {May, Jonathan  and
    Shutova, Ekaterina  and
    Herbelot, Aurelie  and
    Zhu, Xiaodan  and
    Apidianaki, Marianna  and
    Mohammad, Saif M.},
  month = {June},
  pages = {54--63},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in {T}witter},
  url = {https://aclanthology.org/S19-2007/},
  year = {2019}
}

@inproceedings{hate_speech_18,
  address = {Brussels, Belgium},
  author = {de Gibert, Ona  and
    Perez, Naiara  and
    Garc{\'\i}a-Pablos, Aitor  and
    Cuadros, Montse},
  booktitle = {Proceedings of the 2nd Workshop on Abusive Language Online ({ALW}2)},
  doi = {10.18653/v1/W18-5102},
  month = {October},
  pages = {11--20},
  publisher = {Association for Computational Linguistics},
  title = {{Hate Speech Dataset from a White Supremacy Forum}},
  url = {https://www.aclweb.org/anthology/W18-5102},
  year = {2018}
}
```