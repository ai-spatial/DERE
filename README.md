# 🌍 DERE: Decomposition-and-Resembling for Global Carbon Flux Prediction

Code for the accepted paper at the KDD AI4Science Track:
> **Knowledge-Guided Learning for Global Carbon Flux Prediction: Integrating High-Level Remote Sensing with Bottom-Up Physical Modeling**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21514993.svg)](https://doi.org/10.5281/zenodo.21514993)


DERE is a knowledge-guided learning framework for global carbon flux prediction. It integrates process-based model simulations, high-level remote sensing observations, and in-situ flux measurements to predict carbon flux variables, including **GPP**, **RECO**, and **NEE**.

## 🧩 Overview

Process-based ecosystem models provide important physical knowledge for carbon flux prediction, but they often have limited flexibility to directly incorporate increasingly available observations. In particular, high-level remote sensing observations, such as aggregated plant functional type information, cannot be directly matched with bottom-up sub-processes in ecosystem models.

DERE addresses this challenge through a **decomposition-and-resembling** design. It first decomposes simulation outputs into physically meaningful intermediate components, and then resembles them with high-level remote sensing labels and sparse in-situ observations. The framework further uses probabilistic label expansion and uncertainty-aware finetuning to improve the use of sparse carbon flux measurements.

## 🚀 Features

* Knowledge-guided decomposition-and-resembling framework
* Integration of process-based simulations, remote sensing labels, and in-situ observations
* High-level remote sensing supervision for bottom-up sub-processes
* CSDI-based probabilistic label expansion for sparse in-situ flux observations
* Uncertainty-aware fine-tuning with imputed observations
* Baseline and KGML implementations across multiple time-series backbones
* Comparison models including Transformer, Informer, FEDformer, iTransformer, TimeXer, and SimpleTM

## 📁 Code Organization

```text
DERE/
├── README.md
├── .gitignore
├── DERE-main/                         # Main DERE pipeline and Transformer/Informer experiments
│   ├── 01_Baseline_Informer.py
│   ├── 01_Baseline_Transformer.py
│   ├── 02_KGML_Informer.py
│   ├── 02_KGML_Transformer.py
│   ├── Step01_DERE_Train_3PureModels_CompetitionModel.py
│   ├── Step02_DERE_Finetune_CompetitionModel.py
│   ├── Step03_DERE_Train_PFTModel.py
│   ├── Step04_DERE_Finetune_with_InSitu.py
│   ├── Step05_DERE_InSitu_imputation_CSDI-main/
│   ├── Step06_DERE_Finetune_with_InSitu_imputation.py
│   ├── data/
│   ├── exp/
│   ├── models/
│   └── utils/
├── FEDformer_iTransformer_TimeXer/    # FEDformer, iTransformer, and TimeXer experiments
└── SimpleTM/                          # SimpleTM experiments
```

The main code for DERE is located in **`DERE-main/`**. This folder contains the proposed DERE pipeline, Transformer/Informer baselines, KGML variants, model definitions, experiment scripts, and utility functions.

The folders **`FEDformer_iTransformer_TimeXer/`** and **`SimpleTM/`** contain additional comparison models used in the paper.

## 🔬 DERE Pipeline

The proposed DERE framework is implemented as a sequential pipeline:

1. **Train pure component models and the competition model**

   ```text
   DERE-main/Step01_DERE_Train_3PureModels_CompetitionModel.py
   ```

2. **Finetune the competition model**

   ```text
   DERE-main/Step02_DERE_Finetune_CompetitionModel.py
   ```

3. **Train the PFT model using high-level remote sensing labels**

   ```text
   DERE-main/Step03_DERE_Train_PFTModel.py
   ```

4. **Finetune with in-situ carbon flux observations**

   ```text
   DERE-main/Step04_DERE_Finetune_with_InSitu.py
   ```

5. **Perform CSDI-based in-situ label imputation**

   ```text
   DERE-main/Step05_DERE_InSitu_imputation_CSDI-main/
   ```

6. **Finetune with imputed in-situ observations**

   ```text
   DERE-main/Step06_DERE_Finetune_with_InSitu_imputation.py
   ```

## 🧠 Comparison Models

The repository includes the baseline and KGML models used in the paper.

| Model        | Location                          |
| ------------ | --------------------------------- |
| Transformer  | `DERE-main/`                      |
| Informer     | `DERE-main/`                      |
| FEDformer    | `FEDformer_iTransformer_TimeXer/` |
| iTransformer | `FEDformer_iTransformer_TimeXer/` |
| TimeXer      | `FEDformer_iTransformer_TimeXer/` |
| SimpleTM     | `SimpleTM/`                       |

Example scripts include:

```text
DERE-main/01_Baseline_Transformer.py
DERE-main/01_Baseline_Informer.py
DERE-main/02_KGML_Transformer.py
DERE-main/02_KGML_Informer.py
FEDformer_iTransformer_TimeXer/01_Baseline_FEDformer.py
FEDformer_iTransformer_TimeXer/01_Baseline_iTransformer.py
FEDformer_iTransformer_TimeXer/01_Baseline_TimeXer.py
FEDformer_iTransformer_TimeXer/02_KGML_FEDformer.py
FEDformer_iTransformer_TimeXer/02_KGML_iTransformer.py
FEDformer_iTransformer_TimeXer/02_KGML_TimeXer.py
SimpleTM/01_Baseline_SimpleTM.py
SimpleTM/02_KGML_SimpleTM.py
```

## 📊 Data

The processed DERE research dataset is publicly available on Hugging Face:

**Dataset:** https://huggingface.co/datasets/ai-spatial/DERE

The release contains the multidimensional arrays and metadata used in this work, including:

- global ED simulation inputs and outputs
- age-specific ED plant functional type fractions
- ESA CCI plant functional type observations
- LiDAR-derived forest-age weights
- in-situ GPP, RECO, and NEE observations
- fixed training and testing splits
- feature and target definitions
- normalization statistics
- source and licensing information

Detailed array dimensions, metadata, and source information are provided in the dataset repository.

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{xu2026knowledge,
  author    = {Shuo Xu and Zhihao Wang and Ruohan Li and Ruichen Wang and Lei Ma and George C. Hurtt and Xiaowei Jia and Yiqun Xie},
  title     = {Knowledge-Guided Learning for Global Carbon Flux Prediction: Integrating High-Level Remote Sensing with Bottom-Up Physical Modeling},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  address   = {Jeju Island, Republic of Korea},
  publisher = {ACM},
  doi       = {10.1145/3770855.3818927}
}
```


## 📬 Contact

For questions or feedback, feel free to reach out:

- **Shuo Xu** — [shuoxu98@umd.edu](mailto:shuoxu98@umd.edu)
- **Yiqun Xie** — [xie@umd.edu](mailto:xie@umd.edu)

