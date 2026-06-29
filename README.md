# Why Multimodal In-Context Learning Lags Behind? Unveiling the Inner Mechanisms and Bottlenecks

<p align="center">
  <a href="https://rain305f.github.io/">Yu Wang</a>, <a href="https://pages.cs.wisc.edu/~sharonli/">Sharon Li</a>.
  <br>
  <a href="https://github.com/deeplearning-wisc/Multimocal-ICL-Analysis-Framework-MGI/blob/main/LICENSE"><img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-yellow?style=flat&link=https%3A%2F%2Fgithub.com%2Fhc495%2FICL_Circuit%2Fblob%2Fmaster%2FLICENSE"></a>
  <a href="https://aclanthology.org/2026.acl-long.622/"><img src="https://img.shields.io/badge/ACL_2026-Accepted-blue?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DxizpnYNvQq"></a>
  <a href="https://arxiv.org/pdf/2604.13403"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2604.13403-red?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2410.04468"></a>
</p>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>
<h5 align="center">

This repo contains the official code for the following paper published at ACL 2026:
> Yu Wang, Sharon Li. **"Why Multimodal In-Context Learning Lags Behind? Unveiling the Inner Mechanisms and Bottlenecks"** 

## 🚀 Updates

- **Coming soon** — Dataset release
- **June 28, 2026** — Code released 🎉
- **April 15, 2026** — We released the paper on [Arxiv](https://arxiv.org/pdf/2604.13403).


## Overview

### Abstract
*In-context learning (ICL) enables models to adapt to new tasks via inference-time demonstrations. Despite its success in large language models, the extension of ICL to multimodal settings remains poorly understood in terms of its internal mechanisms and how it differs from text-only ICL. In this work, we conduct a systematic analysis of ICL in multimodal large language models. Using identical task formulations across modalities, we show that multimodal ICL performs comparably to text-only ICL in zero-shot settings but degrades significantly under few-shot demonstrations. To understand this gap, we decompose multimodal ICL into task mapping construction and task mapping transfer, and analyze how models establish cross-modal task mappings, and transfer them to query samples across layers. Our analysis reveals that current models lack reasoning-level alignment between visual and textual representations, and fail to reliably transfer learned task mappings to queries. Guided by these findings, we further propose a simple inference-stage enhancement method that reinforces task mapping transfer. Our results provide new insights into the mechanisms and limitations of multimodal ICL and suggest directions for more effective multimodal adaptation.*


## 🔧 Installation

### 0. Requirement

1. Two GPU with more than 80GB VRAM and CUDA (Ver. `12.4` recommended) are strongly required to run all the experiments.
2. Network connection to `huggingface` is needed to download the pre-trained model. And a `huggingface` user token with access to the [`Qwen2.5-VL-7B/32B`] and [`Gemma-3-12B/27B`] model is recommended to run a part of the experiments.


### 1. Clone the repository

```bash
git clone https://github.com/deeplearning-wisc/Multimocal-ICL-Analysis-Framework-MGI.git
```

### 2. Environment Installation and Dataset Preparation

**Direct Installation**

```bash
conda env create -f environment.yaml
conda activate icl_analysis
```



## 🚀 Quick Start

Due to the fact that this paper consists of many relatively independent experiments, we provide a unified script-based approach to generate the main visualizations. The detailed experiment instructions are included in each section below.

### 1. Fig. 4 - Layer-wise Demo Label Token Attention to Image Patches

```bash
python Experiments/vis_attention_analysis.py \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --mode image_vis \
  --output-dir results/vis_attention
```

Visualizes demo label token attention to image patches layer-by-layer with heatmap overlay on the original images.

### 2. Fig. 5 - Layer-wise Attention Ratios to Evidence Regions

```bash
python Experiments/finding1_attntion_curve_indemo.py
```

Shows layer-wise attention ratios to correct/false/irrelevant evidence regions for both correct and incorrect predictions.

### 3. Fig. 6 - Attention Difference Between Correct and Error Samples

```bash
python Experiments/finding2_attntion_curve_query2image.py

```

Computes and visualizes the attention difference (Correct - Error) in layer-wise last-token attention to evidence regions for both models.

### 4. Tab. 2 (UAS) - ICL Circuit Analysis

```bash
python control_att_exp.py
```

### 5. Our MGI - Enhancement Method with adaptive mask

```bash
python IMP_mllm_ours_v1.py 
```


### 5. Our MGI - Enhancement Method with GT mask

```bash
python IMP_mllm_ours_gt_mask.py
```


### 5. Visualization Code
refer to  `plot.ipynb`

## 👍 Acknowledgement
* This project builds upon [ICL_Circuit](https://github.com/hc495/ICL_Circuit) and [`StaICC`](https://github.com/hc495/StaICC). We thank the authors for their excellent work.


## Citation

If you find this work useful for your research, please cite [our paper](https://arxiv.org/pdf/2604.13403):

```
@inproceedings{wang2026multimodal,
  title={Why Multimodal In-Context Learning Lags Behind? Unveiling the Inner Mechanisms and Bottlenecks},
  author={Wang, Yu and Li, Sharon},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={13670--13685},
  year={2026}
}
```
