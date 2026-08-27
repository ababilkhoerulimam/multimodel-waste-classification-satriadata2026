<div align="center">
  <h1>BDC Satria Data 2026 - Waste Classification</h1>
  <p><strong>Multiclass Image Classification with 4-Model Vision Ensemble and Leakage-Aware Validation</strong></p>

  <p align="center">
    <img src="https://img.shields.io/badge/Competition-Satria_Data_BDC_2026-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/Task-Waste_Classification-green?style=flat-square" alt="Task">
    <img src="https://img.shields.io/badge/Metric-Macro_F1_Score-orange?style=flat-square" alt="Metric">
    <img src="https://img.shields.io/badge/Language-Python_3-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Framework-PyTorch_&_timm-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="Framework">
    <img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" alt="Status">
  </p>

  <p align="center">
    A high-precision computer vision pipeline built for Problem 1 (Waste Classification) of Big Data Challenge Satria Data 2026, combining modern convolutional and vision transformer architectures with test-time augmentation, out-of-fold weight optimization, and comprehensive data cleaning.
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Overview

Big Data Challenge (BDC) Satria Data 2026 Problem 1 focuses on automated image classification across three distinct waste categories:
- Class 0: Recyclable (9,999 images, 37.7%)
- Class 1: Electronic (3,961 images, 14.9%)
- Class 2: Organic (12,567 images, 47.4%)

The competition imposes a strict constraint of only 3 lifetime submissions, making rigorous offline validation and leakage-free ensembling critical to prevent overfitting to the public leaderboard.

### Evaluation Metric

The competition evaluates submissions using Macro-Averaged F1-Score:

$$
\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

Because Class 1 (Electronic) is the minority class with roughly 14.9% representation, Macro-F1 heavily penalizes models with weak recall or precision on electronic waste.

## Data Quality & Leakage Audit

A comprehensive hash-based and visual audit revealed key dataset phenomena:

1. Train-Test Duplicate Leakage: MD5 hashing detected 97 test images that are exact byte duplicates of training images (95 Organic, 2 Electronic, 0 Recyclable). These samples are isolated from cross-validation folds to prevent artificial score inflation and predicted via deterministic hash override.
2. Within-Train Duplicates: 62 duplicate groups exist within the training set, with Electronic showing the highest duplicate rate (1.51%).
3. Multi-Batch Data Cleaning:
   - Batch 0: Identified byte-identical cross-class conflict (`O_8873.jpg` vs `R_799.jpg`) where the jute bag is correctly Recyclable.
   - Batch 1: Fixed 4 mislabeled items via Grad-CAM error analysis.
   - Batch 2: Evaluated 128 high-confidence out-of-fold (OOF) model disagreements, leading to 59 corrected labels and 56 off-task noise images dropped from training.
4. Robust Image Mode Conversion: The dataset contains 312 non-RGB images (293 Palette, 17 RGBA, 2 Grayscale). A custom loader composites RGBA images onto solid white backgrounds and standardizes all inputs to RGB.

## Validation Strategy

To guarantee zero leakage and consistent fold distributions:

$$
\text{CV Split} = \text{StratifiedGroupKFold}(k=5, \text{group}=\text{duplicate-group-id}, \text{seed}=42)
$$

Grouping by exact and near duplicate IDs ensures that near-identical image pairs never appear simultaneously in train and validation folds. Class balance deviation across all 5 folds remains strictly under 0.02 percentage points.

## Model Architectures & Training Regime

The solution utilizes four diverse vision backbones:

1. ConvNeXt V2-Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k_384`): Modern convolutional architecture acting as the primary anchor, trained at 384x384 resolution with Exponential Moving Average (EMA decay 0.999) and Label Smoothing (epsilon 0.1).
2. DINOv2 Small (`vit_small_patch14_dinov2.lvd142m`): Self-supervised Vision Transformer with strong geometric and semantic feature representations, fine-tuned at 224x224 resolution.
3. EVA-02 Base (`eva02_base_patch14_448.mim_in22k_ft_in1k`): Masked Image Modeling ViT backbone trained at high resolution (448x448).
4. SigLIP SO400M (`vit_so400m_patch14_siglip_384.webli`): WebLI-pretrained multimodal vision backbone trained at 384x384 resolution with CLIP-style normalization.

### Training Configuration

- Optimizer: AdamW with discriminative learning rates (backbone LR 1e-5 to 5e-6, classifier head LR 1e-4) and weight decay 0.05.
- Learning Rate Schedule: Cosine annealing with 7.5% linear warmup.
- Mixed Precision: Native `torch.amp.autocast('cuda')` with `torch.amp.GradScaler('cuda')`.
- Regularization: Exponential Moving Average (EMA decay 0.999), Label Smoothing (0.1), RandAugment, ColorJitter, RandomHorizontalFlip.

## Ensemble & Inference Pipeline

The inference pipeline combines four complementary models through multi-stage calibration:

1. 8-Pass Test-Time Augmentation (TTA):
   - Original image
   - Horizontal flip
   - Vertical flip
   - 90 degree rotation
   - 180 degree rotation
   - 270 degree rotation
   - Five-crop center and corner extractions
   - Five-crop with horizontal flip
2. Out-of-Fold Probability Optimization:
   - Nelder-Mead simplex optimization and systematic grid search on 5-fold OOF probability distributions to find optimal model blending weights.
3. Deterministic Leakage Override:
   - MD5 hash lookup overrides predictions for the 97 exact train-test duplicate images.
4. Locked ID Sequence Output:
   - Predictions mapped to integers (`0`, `1`, `2`) and formatted to match `submission.csv` template ID order (1 to 1458).

## Experimental Results

| Model / Strategy | Resolution | 5-Fold Mean CV (Macro-F1) | Fold Std Dev |
|---|---|---|---|
| ConvNeXt V2-Tiny (Baseline) | 224x224 | 0.9806 | 0.0020 |
| ConvNeXt V2-Tiny (Clean Data + EMA) | 384x384 | 0.9827 | 0.0016 |
| DINOv2 Small | 224x224 | 0.9801 | 0.0022 |
| EVA-02 Base | 448x448 | 0.9835 | 0.0014 |
| SigLIP SO400M | 384x384 | 0.9842 | 0.0012 |
| Dual Blend (ConvNeXt + DINOv2 + 8-Pass TTA) | Multi | 0.9858 | 0.0011 |
| Quad Blend (ConvNeXt + DINOv2 + EVA-02 + SigLIP + TTA) | Multi | 0.9874 | 0.0009 |

## Repository Structure

```text
satria-data-bdc-2026/
├── .gitignore                                    # Git exclusion rules
├── LICENSE                                       # MIT License
├── README.md                                     # Project documentation
├── requirements.txt                              # Pinned Python dependencies
├── docs/                                         # Technical documentation and audit logs
│   ├── project_state.md                          # Full chronological project log
│   ├── repository_understanding.md               # Repository architecture deep-dive
│   └── eda/                                      # Dataset profiling, leakage, and distribution audits
│       ├── dataset_profile.md
│       ├── distribution_report.md
│       ├── duplicate_candidates.csv
│       ├── feature_audit.md
│       ├── insight_report.md
│       ├── leakage_candidates.csv
│       ├── leakage_report.md
│       ├── mislabel_candidates.csv
│       ├── outlier_candidates.csv
│       ├── quality_issues.csv
│       ├── quality_report.md
│       ├── recommendation_report.md
│       └── temporal_report.md
├── notebooks/                                    # Sequential Jupyter workflows
│   ├── 01_baseline_mislabel_audit.ipynb          # Single-fold baseline & error diagnostic
│   ├── 02_multimodel_training_448.ipynb          # Multi-architecture high-res training
│   ├── 03_convnext_dinov2_dual_ensemble.ipynb    # 5-fold dual ensemble with 8-pass TTA
│   └── 04_quad_ensemble_nelder_mead.ipynb        # 4-model blend with Nelder-Mead optimization
├── reports/                                      # Optimization and sample audit artifacts
│   ├── top50_grid_search_weights.csv             # Grid search ensemble weight rankings
│   ├── unstable_deep_audit.csv                   # Deep error audit for borderline cases
│   └── unstable_samples_audit.csv                # Sample audit across conflicting model outputs
└── submissions/                                  # Competition submission files
    ├── 97.3.csv
    ├── 97.8.csv
    ├── EVA02_SIGLIP_CONVNEXT_DINO.csv
    ├── SIGLIP_DOMINANCE.csv
    ├── sub4_top3_ensemble.csv
    └── submission_Siglip_dominance.csv
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with 16GB VRAM or Kaggle Tesla T4 environment
- CUDA 11.8 or higher with PyTorch GPU support

### Installation

```bash
git clone https://github.com/ababilkhoerulimam/satria-data-bdc-2026.git
cd satria-data-bdc-2026
pip install -r requirements.txt
```

### Running the Pipelines

1. Exploratory Data Analysis & Leakage Audit:
   Review `docs/eda/dataset_profile.md` and execute `notebooks/01_baseline_mislabel_audit.ipynb` to inspect class distributions and mislabel candidates.
2. Multi-Model 5-Fold Training:
   Execute `notebooks/02_multimodel_training_448.ipynb` or `notebooks/03_convnext_dinov2_dual_ensemble.ipynb` across folds 0-4 to train backbones with EMA and AMP enabled.
3. Ensemble Calibration & Submission Generation:
   Execute `notebooks/04_quad_ensemble_nelder_mead.ipynb` to optimize blending weights on out-of-fold predictions, run 8-pass test-time augmentation, apply deterministic hash overrides, and export submission files to `submissions/`.

## Team

- Ababil - ML Architect & Pipeline Engineer
- Jeremy - Exploratory Data Analysis & Data Quality Audit
- Vierico - Business Strategy & Submission Risk Management
