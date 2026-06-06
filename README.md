# DERE: Decomposition-and-Resembling for Global Carbon Flux Prediction

This repository provides the implementation of **DERE**, a knowledge-guided learning framework for global carbon flux prediction. The method integrates simulation data, remote-sensing-derived variables, and in-situ flux observations to predict carbon flux variables, including **GPP**, **RECO**, and **NEE**.

This code accompanies the paper:

> **Knowledge-Guided Learning for Global Carbon Flux Prediction: Integrating High-Level Remote Sensing with Bottom-Up Physical Modeling**

## Overview

DERE is designed to improve carbon flux prediction by combining high-level observational data with bottom-up physical modeling. The framework decomposes carbon flux prediction into physically meaningful components and then resembles them to produce final flux estimates. It also uses probabilistic label expansion and uncertainty-aware finetuning to better leverage sparse in-situ carbon flux observations.

The repository includes the full DERE pipeline and several comparison models used in the paper.

## Code Organization

```text
DERE/
├── DERE-main/
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
├── FEDformer_iTransformer_TimeXer/
├── SimpleTM/
└── README.md
```

The main code for the proposed method is located in **`DERE-main/`**. It implements the complete DERE framework, including sequential steps from pre-training to in-situ finetuning and imputation-enhanced finetuning.

## DERE Pipeline

The full DERE workflow is organized as follows:

1. **Train pure component models and the competition model**

   ```text
   DERE-main/Step01_DERE_Train_3PureModels_CompetitionModel.py
   ```

2. **Finetune the competition model**

   ```text
   DERE-main/Step02_DERE_Finetune_CompetitionModel.py
   ```

3. **Train the PFT-specific model**

   ```text
   DERE-main/Step03_DERE_Train_PFTModel.py
   ```

4. **Finetune with in-situ observations**

   ```text
   DERE-main/Step04_DERE_Finetune_with_InSitu.py
   ```

5. **Perform CSDI-based in-situ imputation**

   ```text
   DERE-main/Step05_DERE_InSitu_imputation_CSDI-main/
   ```

6. **Finetune with imputed in-situ observations**

   ```text
   DERE-main/Step06_DERE_Finetune_with_InSitu_imputation.py
   ```

## Comparison Models

The repository also includes comparison models used in the paper.

| Model        | Location                          |
| ------------ | --------------------------------- |
| Informer     | `DERE-main/`                      |
| Transformer  | `DERE-main/`                      |
| FEDformer    | `FEDformer_iTransformer_TimeXer/` |
| iTransformer | `FEDformer_iTransformer_TimeXer/` |
| TimeXer      | `FEDformer_iTransformer_TimeXer/` |
| SimpleTM     | `SimpleTM/`                       |

Example baseline and KGML scripts include:

```text
DERE-main/01_Baseline_Informer.py
DERE-main/01_Baseline_Transformer.py
DERE-main/02_KGML_Informer.py
DERE-main/02_KGML_Transformer.py
FEDformer_iTransformer_TimeXer/01_Baseline_FEDformer.py
FEDformer_iTransformer_TimeXer/01_Baseline_iTransformer.py
FEDformer_iTransformer_TimeXer/01_Baseline_TimeXer.py
FEDformer_iTransformer_TimeXer/02_KGML_FEDformer.py
FEDformer_iTransformer_TimeXer/02_KGML_iTransformer.py
FEDformer_iTransformer_TimeXer/02_KGML_TimeXer.py
SimpleTM/01_Baseline_SimpleTM.py
SimpleTM/02_KGML_SimpleTM.py
```

## Requirements

The code is implemented in Python with PyTorch. The main dependencies include:

```text
python
pytorch
numpy
pandas
scikit-learn
matplotlib
pyyaml
tqdm
scipy
sympy
einops
pywavelets
```

Additional dependencies may be required by specific comparison models or the CSDI-based imputation module.

## Data

The data used in this study are from publicly available sources, including in-situ carbon flux observations, process-based model simulations, and remote-sensing-derived plant functional type (PFT) labels.

Specifically, the in-situ flux observations are based on public carbon flux benchmark datasets such as CarbonSense. The process-based model data include input conditions and simulation outputs from CarbonGlobe, which focuses on physical model emulation. The remote-sensing-derived PFT labels are obtained from ESA CCI products. Please refer to the paper for detailed descriptions of the datasets, preprocessing procedures, and citations.

Due to data size and data-sharing considerations, the processed datasets are not directly included in this repository. Users should download the original datasets from the corresponding public sources and update the data paths in the scripts before running the experiments.

## Usage

After preparing the data, run the scripts according to the desired experiment. For the proposed DERE framework, follow the sequential pipeline in `DERE-main/`:

```bash
cd DERE-main
python Step01_DERE_Train_3PureModels_CompetitionModel.py
python Step02_DERE_Finetune_CompetitionModel.py
python Step03_DERE_Train_PFTModel.py
python Step04_DERE_Finetune_with_InSitu.py
cd Step05_DERE_InSitu_imputation_CSDI-main
python 05_DERE_InSitu_imputation.py
cd ..
python Step06_DERE_Finetune_with_InSitu_imputation.py
```

For baseline or KGML experiments, run the corresponding scripts directly. For example:

```bash
cd DERE-main
python 01_Baseline_Transformer.py
python 02_KGML_Transformer.py
```

Please check and modify dataset paths, model settings, training settings, and output directories in each script before running.

## Outputs

The scripts save training logs, checkpoints, and prediction results according to the paths specified in the corresponding experiment files. Please create the required output folders before running the experiments if they are not automatically generated.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{xu2026dere,
  title     = {Knowledge-Guided Learning for Global Carbon Flux Prediction: Integrating High-Level Remote Sensing with Bottom-Up Physical Modeling},
  author    = {Xu, Shuo and Wang, Zhihao and Li, Ruohan and Wang, Ruichen and Ma, Lei and Hurtt, George C. and Jia, Xiaowei and Xie, Yiqun},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```

Please update the citation once the official proceedings information is available.

## Acknowledgment

This repository includes implementations or adapted components related to several time-series forecasting and imputation models, including Informer, Transformer, FEDformer, iTransformer, TimeXer, SimpleTM, and CSDI. We thank the authors of these methods for making their work available to the community.

## License

Please refer to the license information in this repository. The CSDI-based imputation module contains its own license file in `DERE-main/Step05_DERE_InSitu_imputation_CSDI-main/`.
