# BDC Satria Data 2026 — Complete Repository Understanding
## Phase 1: Repository Summary | Phase 2: Project History | Phase 3: Key Findings | Phase 4: Final Strategy | Phase 5: Notebook Architecture

---

## PHASE 1 — REPOSITORY UNDERSTANDING SUMMARY

### Project Identity
| Field | Value |
|---|---|
| Competition | Big Data Challenge (BDC) Satria Data 2026 — Problem 1: Waste Classification |
| Platform | Satria Data (not Kaggle) |
| Deadline | 30 Juli 2026 16:00 WIB |
| Metric | **Macro-averaged F1-Score** |
| Submission Budget | **3 total — CRITICAL CONSTRAINT** |
| Team | Ababil (ML Architect), Jeremy (EDA/Exploration), Vierico (Business Strategy) |

### Repository Structure
```
satria-data-bdc-2026/
├── projectstate.md              ← MAIN PROJECT TRUTH (93KB, 741 lines)
├── submission.csv               ← Template with 1,458 IDs (order locked)
├── train_master_with_folds.csv  ← Final training manifest (2.86MB)
├── train_master_with_groups.csv ← Groups for StratifiedGroupKFold
├── train_test_overlap.csv       ← 97 train-test exact duplicate pairs
├── train_duplicate_groups.csv   ← All within-train duplicate groups
├── near_duplicate_candidates.csv ← 2 near-dup pairs in Electronic
├── Miss_Label_Report_1.xlsx     ← Batch 2 mislabel report (Recyclable→Organic)
├── Miss_Label_Report_2.xlsx     ← Batch 2 mislabel report (Organic→Recyclable)
├── image.png                    ← Team workflow diagram
├── ConvNext.ipynb               ← Full 5-fold training notebook (11MB)
├── satria-data-dinov2-exe_tuning.ipynb ← DINOv2 experiment (9.7MB)
├── src/
│   ├── README.md                ← Complete pipeline code reference (Cells 0–34)
│   └── src.md                   ← Kaggle-vs-local dual environment setup
├── strategy/
│   ├── SONNET 5.md              ← Original blueprint strategy (2 Juli 2026)
│   ├── REVISI DARI SONNET 5.md  ← Revised final strategy (ConvNeXt V2-Tiny as anchor)
│   ├── Kimi agent.md            ← Alternative strategy proposal (broader scope)
│   ├── MINIMAX.md               ← Alternative strategy
│   ├── SONNET 4.6 MEDIUM.txt    ← Model selection notes
│   └── ZAI GLM.txt              ← Additional strategy notes
├── agents/
│   ├── v1/ → v2/ → v3/         ← Agent role definitions (iteratively refined)
│   └── (AGENTS_ababil, AGENTS_jeremy, AGENTS_vierico for each version)
└── notebook prototype/
    ├── context.md               ← Agent system prompt used during prototyping
    ├── README.md                ← EDA cell code reference
    ├── paper.md                 ← Paper analysis (Yasin & Koklu 2024)
    └── (various prototype notebooks)
```

### Dataset Facts
- **Train**: 26,527 images in 3 class folders: `0_Recyclable` (9,999 / 37.7%), `1_Electronic` (3,961 / 14.9%), `2_Organic` (12,567 / 47.4%)
- **Test**: 1,458 images (flat folder, labeled 1.jpg–1458.jpg)
- **Target**: `predicted` column in `submission.csv` — Mapping: 0=Recyclable, 1=Electronic, 2=Organic
- **ID order in submission.csv is LOCKED — must not be reordered**

### Critical Data Facts (All Confirmed)
1. **Leakage**: 97/1,458 test files (6.65%) are exact MD5 duplicates of train files (95 Organic, 2 Electronic, 0 Recyclable)
2. **Within-train duplicates**: 62 groups (1 cross-class + 61 within-class). Electronic has highest proportion (1.51%)
3. **Cross-class mislabel**: `O_8873.jpg` = `R_799.jpg` (byte-identical, jute bag — correctly Recyclable). Confirmed from paper too.
4. **Image modes**: 312 non-RGB train files (293 Palette, 17 RGBA, 2 Grayscale) + 5 test Palette. All handled by `load_image_as_rgb()`.
5. **Electronic bimodal**: 68.8% are exact 150×150px stock icons, 31.2% are natural photos (median width 1028px)
6. **Zero corrupt files** in entire train+test dataset

---

## PHASE 2 — PROJECT HISTORY (Chronological Evolution)

### Stage 1 — Strategy Formation (1–2 Juli 2026)
- Initial strategy discussions across multiple LLMs (Kimi, Minimax, Sonnet 5)
- **Sonnet 5 Blueprint** (2 Juli): Recommended SwinV2-Tiny as anchor, EVA-02 as secondary
- **Revision** (2 Juli): Sonnet corrected itself — ConvNeXt V2-Tiny chosen as anchor for its forgiving training recipe under hardware uncertainty
- All strategic flags (Flag 1–7) closed with final decisions
- StratifiedGroupKFold locked: 5-fold, seed=42, group=duplicate_group_id

### Stage 2 — Data Pipeline Construction (3 Juli 2026)
- Built complete data pipeline (Cells 0–22 in notebook)
- MD5 hashing: identified 97 train-test overlaps, 62 duplicate groups, 2 near-dup pairs
- Built `train_master_with_groups.csv` (26,527 rows, 26,463 unique groups)
- Built `train_master_with_folds.csv` (5-fold split, balance deviation <0.02pp)
- Built `WasteDataset`, `train_transform`, `eval_transform`, `load_image_as_rgb()`
- Discovered 312 non-RGB train files (much more than EDA estimated ~19)
- Environment porting Kaggle ↔ Local (data_dir/output_dir pattern)

### Stage 3 — Baseline Training (3–4 Juli 2026)
- **exp001**: ConvNeXt V2-Tiny, plain CE, NO augmentation, fold 0 → CV **0.9823** (data DIRTY)
- F1 Electronic=0.994 — surprisingly high, suspected shortcut learning
- **Subpopulation check**: icon 150×150 recall=0.9927, natural recall=0.9918, gap=0.0009 → **shortcut hypothesis REJECTED**
- **Grad-CAM**: 24 samples analyzed (16 true-positive + 8 error cases) → **NO shortcut learning detected** → CutMix NOT activated
- Found 4 candidate mislabels via Grad-CAM error analysis

### Stage 4 — First Data Cleaning & Retraining (4–8 Juli 2026)
- **Batch 1 Relabel (4 Juli)**: 4 files confirmed mislabeled and relabeled:
  - `R_3825.jpg` → Electronic (laptop)
  - `R_3733.jpg` → Electronic (laptop)
  - `O_7776.jpg` → Electronic (control panel)
  - `battery_61.jpg` → Recyclable (bottles)
- **exp001-rerun**: Accidentally re-ran training on DIRTY data (loader not reloaded) → CV 0.9827 (not comparable)
- **exp002**: CB-Loss + Weighted Sampler, clean data → CV **0.9809** (F1 Electronic=0.992)
- **exp003** (8 Juli): Plain CE, clean data → CV **0.9815** — TRUE CLEAN BASELINE
- **Ablation conclusion**: CB-Loss HURT performance vs plain CE (-0.0006 on already clean data)

### Stage 5 — Architecture Comparison (8–9 Juli 2026)
- **exp004**: EfficientNetV2-S, wrong LR recipe, IMG_SIZE 224 → CV 0.9368 (epoch 1 loss=5.56, anomalous)
- **exp004b**: Head LR 1e-4→1e-3, warmup 7.5%→15% → Epoch 1 loss=5.39, NOT improved
- **exp004c**: Native config (mean/std=0.5, IMG_SIZE=300) → CV **0.9390** (marginal +0.0022)
- **Decision**: EfficientNetV2-S DROPPED — gap -4.25pp from ConvNeXt, Layer 2 ensemble DISABLED

### Stage 6 — Stabilization & Full 5-Fold Training (9 Juli 2026)
- **exp005**: ConvNeXt V2-Tiny + EMA(decay=0.999) + Label Smoothing(ε=0.1), fold 0 → CV **0.9817**
- Training curve more stable in epochs 9–15 vs exp003 (max delta per epoch < 0.0004)
- **Full 5-fold kfold-run** (9 Juli): All 5 folds trained with exp005 config
  - Fold 0: 0.9827, Fold 1: 0.9809, Fold 2: 0.9819, Fold 3: 0.9778, Fold 4: 0.9795
  - **Mean CV = 0.9806, Std = 0.0020** ← FINAL BASELINE RESULT

### Stage 7 — OOF Investigation & Gate 2 (9 Juli 2026)
- **Gate 1**: OOF predictions built from all 5 checkpoints (overall OOF F1 = 0.9806)
- **Gate 2**: 538 total confused cases (Recyclable↔Organic) → 128 high-confidence (≥0.85) extracted
- Manual verification by Ababil (`Miss_Label_Report_1.xlsx` + `Miss_Label_Report_2.xlsx`):
  - 51 → Relabel to Recyclable
  - 8 → Relabel to Organic
  - 56 → DROP (not waste photos at all: soil, person cooking, etc.)
  - 11 → Label already correct (model was wrong, data is fine)
- **Total Batch 2 impact**: 59 relabels + 56 drops = 115 files

### Current Status (26 Juli 2026)
- **Gate 3 PENDING**: Apply Batch 2 cleaning to `train_master_with_folds.csv` → retrain full 5-fold
- **Gate 4 PENDING**: Submit Submission #1 after Gate 3 retraining
- Hardware: Migrating from Kaggle T4 to RTX 5070 Ti local (~16GB VRAM, same constraints)
- Submissions used: **0 / 3**

---

## PHASE 3 — KEY FINDINGS

### Validation Findings
- StratifiedGroupKFold (5-fold, seed=42) delivers fold balance deviation <0.02pp (excellent)
- Full 5-fold mean CV = 0.9806, std = 0.0020 → stable, not single-fold overfit
- CV is THE primary source of truth — 0 submissions used, all experiments done offline

### Data Quality Findings
- **91.3% of 128 high-confidence model "errors" were actually DATA errors** — model was right, labels were wrong
- Cross-class mislabel rate appears higher than naive exact-hash method can detect
- 56 images were entirely off-task (not waste photos) — noise in dataset source
- Batch 0+1+2 total: 63 relabels + 154 drops = 217 files out of 26,527 (~0.82%)

### Leakage Findings
- **RESOLVED**: 97 test-train overlaps → excluded from CV, hash-override for final submission
- Group leakage in CV: PREVENTED via StratifiedGroupKFold (no group spans multiple folds)
- No shortcut learning detected via Grad-CAM (background, size)

### Architecture Findings
- **ConvNeXt V2-Tiny DOMINATES EfficientNetV2-S** by 4.25pp (0.9815 vs 0.9390)
- EfficientNetV2-S underperformed even with native pretrained config (IMG_SIZE=300, Inception-style normalization)
- Root cause: training recipe optimized for ConvNeXt does not transfer well to EfficientNet's MBConv/Fused-MBConv
- Layer 2 (multi-architecture ensemble) DISABLED — gap too large, would pull down performance

### Loss Function Findings
- **Class-Balanced Loss HURT** on this dataset (imbalance 3.2:1 not extreme enough)
- CB-Loss+Sampler: CV=0.9809, F1 Electronic=0.992
- Plain CE: CV=0.9815, F1 Electronic=0.995
- Root cause: Electronic near ceiling → oversampling amplified limited-sample overfitting; CB-Loss weight (1.56x) too aggressive for mild imbalance

### EMA + Label Smoothing Findings
- Gain is small (+0.0002) but training stability is real (epoch variance dropped from 0.0016 to <0.0004)
- Benefit: checkpoint terbaik mencerminkan performa asli model, bukan "lucky epoch"
- Worth using for all production runs (low cost, meaningful stabilization)

### Electronic Subpopulation Analysis
- Icon 150×150 recall = 0.9927, Natural recall = 0.9918 → gap = 0.0009
- **Shortcut learning via size/format = REJECTED** — model generalizes equally well to both subpopulations
- Electronic's high F1 is genuine (distinct visual signature: low brightness ~142, R≈G≈B metallic profile)

### Background/Shortcut Analysis
- Grad-CAM on 24 samples: all attention maps focus on object/discriminative features, not background
- **Shortcut background hypothesis = REJECTED** → CutMix NOT activated
- Recyclable's 38.8% plain-background rate did NOT cause shortcut learning

### Confusion Analysis (Recyclable↔Organic)
- 538 total confused cases (OOF): 239 Recyclable→Organic, 299 Organic→Recyclable
- Distribution is bidirectional and random (no clustering by filename, no systematic source)
- After Gate 3 cleaning, expected CV improvement (unquantified pre-retrain)

---

## PHASE 4 — FINAL PRODUCTION STRATEGY

### Problem Understanding
3-class waste image classification (Recyclable, Electronic, Organic). Metric = Macro-F1. Only 3 submissions total. CV must be trusted 100% — no room for LB probing.

**Key insight**: Organic/Recyclable sub-problem is near-saturation (even 2016 methods achieve ~96.6% F1). **Electronic is the ONLY variable that determines leaderboard position.**

### Dataset Understanding
- 26,527 train images from mixed sources. Electronic contains two fundamentally different sub-populations (stock icons + natural photos) in one label.
- Test set is 1,458 images. 97 are exact duplicates of train (ground truth known → hash-override).
- Data has ~0.82% confirmed label noise (after 3 batches of cleaning).

### Validation Philosophy
**StratifiedGroupKFold (5-fold, seed=42)** — LOCKED. Groups by `duplicate_group_id` ensure no duplicate pair ever splits across folds. Excluded from CV pool: 97 train-test overlaps + 1 confirmed mislabel (`O_8873.jpg`) = 98 files.

### Data Cleaning
**3 batches total:**
- Batch 0: Exclude 97 train-test duplicates from CV, exclude `O_8873.jpg` from training
- Batch 1 (4 Juli): Relabel 4 files (2 laptops, 1 control panel, 1 bottle → correct classes)
- Batch 2 (Gate 3 — PENDING): Relabel 59 files + drop 56 files (noise investigation)

**Rule**: Always backup CSV before overwriting. Use Python lists from projectstate.md directly — no need to re-read Excel.

### Feature Engineering / Input Pipeline
- **Images only** — no metadata (competition constraint)
- `load_image_as_rgb()`: handles RGBA (white composite), Palette (P → RGB), Grayscale (L → RGB)
- Train transform: `Resize(224, LANCZOS) → HFlip(0.5) → RandAugment(2, 7) → ColorJitter(0.15, 0.1, 0.1) → Normalize(ImageNet)`
- Eval transform: `Resize(224, LANCZOS) → Normalize(ImageNet)` — deterministic, no augmentation

### Model Architecture
**Single architecture: ConvNeXt V2-Tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k` via timm)
- 28.6M parameters, pretrained ImageNet-21K → ImageNet-1K fine-tuned
- Discriminative LR: backbone=1e-5, head=1e-4 (10× split)
- AMP enabled (`torch.amp.GradScaler('cuda')`) — mandatory for 16GB VRAM

### Training Recipe (Locked)
| Component | Value | Rationale |
|---|---|---|
| Loss | Plain CrossEntropy | CB-Loss hurt performance (ablation exp002 vs exp003) |
| Optimizer | AdamW | weight_decay=0.05 |
| Scheduler | Cosine + warmup | warmup_ratio=0.075 (7.5% of total steps) |
| Backbone LR | 1e-5 | Preserve ImageNet features |
| Head LR | 1e-4 | 10× discriminative ratio |
| EMA | decay=0.999 | Stabilizes training, save from shadow weights |
| Label Smoothing | ε=0.1 | Calibration against noise, small but stable gain |
| Resolution | 224×224 | LANCZOS interpolation |
| Batch size | 64 | ConvNeXt V2-Tiny @ 224px on 16GB VRAM |
| Epochs | 15 | Sufficient for convergence per exp005 |
| CutMix | NOT ACTIVE | Grad-CAM showed no shortcut learning |
| CB-Loss | NOT ACTIVE | Ablation proved it hurts |

### Ensemble Strategy (Layer 1 Only)
- **5-fold K-fold ensemble**: Average probabilities from 5 model checkpoints
- Simple average (equal weights) — weighted average is optional if time permits
- Layer 2 (multi-architecture) DISABLED: EfficientNetV2-S gap too large (-4.25pp)

### Inference Pipeline — Critical Details
1. Load `submission.csv` first → get ordered list of 1,458 IDs
2. Build test file paths: `{test_dir}/{id}.jpg`
3. For 97 overlap files: **set prediction directly from train label** (no model inference needed)
4. For remaining 1,361 files: run ensemble inference (5 models, average softmax)
5. Final predictions must maintain EXACT row order of submission.csv

### Post-Processing / Data Cleaning for Gate 3
```python
# Apply Batch 2 cleaning to train_master_with_folds.csv:
# 1. Backup first
# 2. Apply RELABEL_TO_RECYCLABLE (51 files) → label=0
# 3. Apply RELABEL_TO_ORGANIC (8 files) → label=2
# 4. Apply DROP_NOISE (56 files) → exclude_from_training=True
# 5. Rebuild DataLoaders (reload from CSV)
# 6. Retrain full 5-fold (same config)
```

### Submission Plan
| Slot | Content | Rationale |
|---|---|---|
| **Submission #1** | 5-fold ensemble, clean data (Gate 3), hash-override for 97 | First competitive submission. Validates CV→LB alignment. |
| **Submission #2** | Held in reserve. Depends on LB score from #1 | Only use if clear CV-validated improvement found |
| **Submission #3** | Reserve until deadline. Possible: full-data retrain (no held-out val) | Risk: no CV comparison. Only as last resort. |

---

## PHASE 5 — COMPLETE NOTEBOOK ARCHITECTURE

> This is the complete design for the final competition submission notebook.
> Production-ready. No exploratory debris. Kaggle Grandmaster quality.

---

### Design Principles
- **Single notebook tells the complete story**: from data → cleaning → training → inference → submission
- **No dead cells**: every cell has a clear purpose and expected output
- **Reproducible**: all randomness seeded, all configs declared upfront
- **Professional markdown**: each section reads like a paper section
- **Evidence-based**: key findings cited from project history, not invented

---

### NOTEBOOK OUTLINE: Cell 00 → Final Cell

---

#### **SECTION 0 — NOTEBOOK OVERVIEW**

---

**Cell 00 — Project Overview (Markdown)**
- Purpose: Introduce the competition, team, and this notebook's scope
- Content: Competition brief (task, metric, 3-submission constraint, deadline), dataset overview (26,527 train / 1,458 test / 3 classes), and statement that this notebook represents the final production pipeline distilled from all experiments
- Expected output: Rendered markdown block
- Why: Sets professional tone; establishes context for any reviewer/judge
- Dependencies: None
- Notes: Include a summary table of key project facts and submission plan

---

**Cell 01 — Configuration Block (Code)**
- Purpose: Centralize ALL hyperparameters and constants in one cell — single place to change anything
- Content:
  ```python
  # ─── Environment ───
  IS_KAGGLE = ...
  DATA_DIR, OUTPUT_DIR = ...

  # ─── Data Constants ───
  IMG_SIZE = 224
  N_SPLITS = 5
  SEED = 42
  BATCH_SIZE = 64
  NUM_WORKERS = 4

  # ─── Model Constants ───
  MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
  BACKBONE_LR = 1e-5
  HEAD_LR = 1e-4
  WEIGHT_DECAY = 0.05
  NUM_EPOCHS = 15
  WARMUP_STEPS_RATIO = 0.075
  EMA_DECAY = 0.999
  LABEL_SMOOTHING = 0.1

  # ─── Class Mapping ───
  CLASS_FOLDERS = ["0_Recyclable", "1_Electronic", "2_Organic"]
  LABEL_MAP = {0: "Recyclable", 1: "Electronic", 2: "Organic"}
  ```
- Expected output: Config printed as confirmation
- Why: Kaggle best practice — one cell to rule all hyperparameters
- Notes: Includes IS_KAGGLE flag for dual-environment compatibility

---

**Cell 02 — Environment Setup (Code)**
- Purpose: Detect Kaggle vs local environment, set data_dir and output_dir paths, validate dataset exists
- Content: `IS_KAGGLE` check, auto-detect data_dir (search for folder with train/ + test/ + submission.csv), set output_dir, assert all paths exist
- Expected output: Printed paths: data_dir, train_dir, test_dir, submission_path, output_dir; file counts
- Why: Same notebook must run on Kaggle and local without manual path editing
- Dependencies: Cell 01

---

**Cell 03 — Library Imports (Code)**
- Purpose: Import all required libraries
- Content: os, pathlib, pandas, numpy, PIL, torch, timm, sklearn, matplotlib, collections, hashlib, re
- Expected output: Library versions printed (torch, timm, sklearn)
- Why: Explicit imports at top = good practice, easy debugging
- Dependencies: Cell 01, 02

---

#### **SECTION 1 — DATA INTEGRITY & PIPELINE FOUNDATION**

---

**Cell 04 — Section Header (Markdown)**
- Purpose: Section break
- Content: "## Section 1: Data Integrity & Pipeline Foundation" + brief explanation of what this section does (builds the reproducible data manifest from scratch, incorporating all data cleaning decisions)
- Why: Professional notebook structure

---

**Cell 05 — MD5 Duplicate Detection (Code)**
- Purpose: Detect exact-MD5 duplicates between train and test sets, and within-train duplicates. Generates/loads `train_test_overlap.csv` and `train_duplicate_groups.csv`.
- Content: `compute_file_hash()`, `hash_all()` functions; generates overlap_df and groups_df. Also runs `near_duplicate_candidates.csv` detection via `(1)` filename pattern
- Expected output: "Train-test overlap: 97 pairs", "Exact-MD5 duplicate groups in train: 57 groups", "Near-duplicate candidates: 2 pairs"
- Why: This is the foundation of the leakage prevention strategy. Must be computed reproducibly.
- Notes: If CSV files already exist (e.g., uploaded to Kaggle), load them directly. If not, compute.

---

**Cell 06 — Build train_master (Code)**
- Purpose: Enumerate all 26,527 train files with labels, assign duplicate_group_ids, merge exclusion flags
- Content: Scan train_dir → build train_master DataFrame → merge exact-dup group IDs (with offset) → merge near-dup group IDs (NEAR_DUP_GROUP_START=900000) → flag exclude_from_cv (97 files) and exclude_from_training (`O_8873.jpg`)
- Expected output: train_master shape (26527, 7), column list, unique group count (26,463), exclusion counts (97 CV, 1 training)
- Why: Central data manifest used for all downstream operations
- Dependencies: Cell 05

---

**Cell 07 — Batch Data Cleaning: Relabel & Drop (Code)**
- Purpose: Apply all confirmed data cleaning decisions (Batch 1 + Batch 2) to train_master
- Content:
  - BATCH_1_RELABEL_MAP: 4 files (R_3825→Electronic, R_3733→Electronic, O_7776→Electronic, battery_61→Recyclable)
  - RELABEL_TO_RECYCLABLE: 51 files (from OOF investigation)
  - RELABEL_TO_ORGANIC: 8 files
  - DROP_NOISE: 56 files (exclude_from_training=True)
  - Backup CSV before modification, then apply
- Expected output: Counts of each action applied, distribution before/after cleaning
- Why: All cleaning decisions are evidence-based (Grad-CAM + OOF visual verification). Centralizing here makes the notebook reproducible and auditable.
- Notes: 4 anomalies in Batch 2 verified (O_1876.jpg, O_1635.jpg, O_6751.jpg, O_10298.jpg, O_819.jpg)

---

**Cell 08 — StratifiedGroupKFold Split (Code)**
- Purpose: Create 5-fold split ensuring duplicate groups stay within the same fold
- Content: Filter CV pool (exclude cv+training flags) → StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) → assign fold column → merge back to train_master → save train_master_with_folds.csv
- Expected output: Fold sizes (≈5,285–5,286 each), class balance per fold (deviaton <0.02pp), group integrity check PASSED
- Why: StratifiedGroupKFold is critical — prevents duplicate-pair leakage across folds and maintains class balance
- Dependencies: Cell 06, 07
- Notes: LOCKED — this exact split is the source of truth for all CV scores reported

---

**Cell 09 — Integrity Assertions (Code)**
- Purpose: Automated sanity checks before any training begins
- Content: Assert no group spans multiple folds, assert exclusion counts match, assert no duplicate filenames, assert class balance deviation <2pp, print summary
- Expected output: "All assertions PASSED. Ready for training."
- Why: Catches any data pipeline bugs silently introduced by refactoring
- Dependencies: Cell 08

---

**Cell 10 — Test Set Loader Setup (Code)**
- Purpose: Load submission.csv as the ordered reference for test inference; validate all 1,458 test files exist
- Content: Load submission.csv, build filename/filepath from id column, assert all files exist on disk
- Expected output: "All 1,458 test files found on disk, matching submission.csv order"
- Why: submission.csv defines the EXACT required output order — test loader must follow this order
- Notes: Critical constraint from competition rules — row order must match submission.csv exactly

---

#### **SECTION 2 — DATASET INSIGHTS (EDA SUMMARY)**

---

**Cell 11 — Section Header (Markdown)**
- Purpose: Section break + EDA findings summary
- Content: "## Section 2: Dataset Insights" + key EDA findings table (class distribution, Electronic bimodal, leakage stats, brightness profile)
- Why: Competition notebook should demonstrate understanding of the data, not just run a pipeline

---

**Cell 12 — Class Distribution & Visual Sample Grid (Code)**
- Purpose: Visualize class distribution and sample images per class
- Content: Bar chart of class counts (Organic 47.4%, Recyclable 37.7%, Electronic 14.9%) + 8 sample images per class in a 3×8 grid
- Expected output: Matplotlib figure (class distribution + image grid)
- Why: Demonstrates understanding of imbalance and visual diversity; essential for a Kaggle-quality notebook

---

**Cell 13 — Electronic Bimodal Analysis (Code)**
- Purpose: Demonstrate the icon-150×150 vs natural-photo split within Electronic
- Content: Load Electronic sample, flag is_small_150, show counts and proportions, display 5+5 comparison grid
- Expected output: "Electronic icon (150×150): 68.8% | Natural: 31.2%", comparison image grid
- Why: This is the most critical dataset insight — justifies the need for specialized handling of Electronic class
- Notes: Cite subpopulation recall check: icon recall=0.9927, natural=0.9918 → model generalizes to both

---

**Cell 14 — Key Dataset Risk Summary (Markdown)**
- Purpose: Present the 7 risk flags (Jeremy's EDA) and the decisions made for each
- Content: Risk flags table (Flag 1–7), severity, decision taken
- Why: Demonstrates methodical approach to data quality; essential for competition report

---

#### **SECTION 3 — IMAGE LOADING & AUGMENTATION PIPELINE**

---

**Cell 15 — Section Header (Markdown)**
- Purpose: Section break
- Content: "## Section 3: Image Loading & Augmentation Pipeline" + brief rationale for design decisions

---

**Cell 16 — Robust RGB Loader (Code)**
- Purpose: Define `load_image_as_rgb()` — handles all image modes safely
- Content:
  ```python
  def load_image_as_rgb(filepath):
      """RGBA → white composite, P/L/CMYK → .convert('RGB')"""
      ...
  ```
- Expected output: Verification on all non-RGB modes (P, RGBA, L) → all show OK
- Why: Dataset has 312 non-RGB train + 5 test palette files. Without this, model breaks silently.

---

**Cell 17 — Transforms Definition (Code)**
- Purpose: Define train and eval transform pipelines
- Content:
  ```python
  IMG_SIZE = 224  # from Cell 01
  train_transform = T.Compose([
      Resize(224, LANCZOS),
      RandomHorizontalFlip(0.5),
      RandAugment(num_ops=2, magnitude=7),
      ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
      ToTensor(),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
  ])
  eval_transform = T.Compose([Resize(224, LANCZOS), ToTensor(), Normalize(...)])
  ```
- Expected output: Transform configs printed
- Why: RandAugment light-medium + limited ColorJitter (brightness is discriminative for Organic/Recyclable). CutMix NOT used (Grad-CAM confirmed no shortcut learning).
- Notes: Augmentation choices are evidence-based, not default presets

---

**Cell 18 — WasteDataset Class (Code)**
- Purpose: Define PyTorch Dataset for train/val and test
- Content: `WasteDataset(df, transform, is_test)` — returns (image, label) for train/val, (image, filename) for test
- Expected output: Class definition + brief unit test on one sample
- Why: Custom Dataset needed to handle both labeled (train/val) and unlabeled (test) cases consistently
- Dependencies: Cell 16

---

#### **SECTION 4 — MODEL ARCHITECTURE**

---

**Cell 19 — Section Header (Markdown)**
- Purpose: Section break + architecture rationale
- Content: "## Section 4: Model Architecture — ConvNeXt V2-Tiny" + architecture decision justification (hardware constraints, training stability, experiment results table showing EfficientNetV2-S gap -4.25pp)
- Why: A professional notebook explains WHY the architecture was chosen, not just WHICH

---

**Cell 20 — Model Initialization (Code)**
- Purpose: Load ConvNeXt V2-Tiny with pretrained ImageNet-21K weights
- Content:
  ```python
  model = timm.create_model(
      "convnextv2_tiny.fcmae_ft_in22k_in1k",
      pretrained=True,
      num_classes=3
  )
  model = model.to(DEVICE)
  ```
- Expected output: Model summary (28.6M params), device confirmed (CUDA)
- Why: ConvNeXt V2-Tiny confirmed as champion architecture from ablation studies
- Notes: Print backbone parameter count and head parameter count separately

---

**Cell 21 — Discriminative LR Setup (Code)**
- Purpose: Split model parameters into backbone (low LR) and head (high LR) groups
- Content:
  ```python
  backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
  head_params = [p for n, p in model.named_parameters() if "head" in n]
  optimizer = AdamW([
      {"params": backbone_params, "lr": BACKBONE_LR},   # 1e-5
      {"params": head_params,     "lr": HEAD_LR},        # 1e-4
  ], weight_decay=WEIGHT_DECAY)
  ```
- Expected output: Param group counts printed (backbone: 27.8M, head: 3,843)
- Why: Discriminative LR is critical — backbone must update slowly to preserve ImageNet features while head learns the 3-class task quickly

---

**Cell 22 — EMA + Loss + Scheduler Setup (Code)**
- Purpose: Initialize EMA, loss function, LR scheduler, and AMP scaler
- Content:
  ```python
  criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # ε=0.1
  ema_model = copy.deepcopy(model)  # shadow weights
  scaler = torch.amp.GradScaler('cuda')  # AMP (non-deprecated form)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # cosine+warmup
  ```
- Expected output: "Training setup ready. Loss: CE + LS(0.1). EMA decay: 0.999. AMP: enabled."
- Why: EMA+LS ablation (exp003 vs exp005) showed +0.0002 macro F1 + significantly more stable training curve

---

#### **SECTION 5 — TRAINING LOOP**

---

**Cell 23 — Section Header (Markdown)**
- Purpose: Section break + training strategy summary
- Content: "## Section 5: Training Pipeline — 5-Fold Cross-Validation" + training recipe table (all hyperparameters), experiment history summary (exp001→exp005→kfold-run), final CV results table (Fold 0-4 scores)
- Why: Professional notebook documents the training strategy, not just the training loop

---

**Cell 24 — Training & Validation Functions (Code)**
- Purpose: Define `train_one_epoch()` and `validate_epoch()` helper functions
- Content:
  - `train_one_epoch()`: forward pass + AMP autocast + backward + optimizer step + EMA update + scheduler step
  - `validate_epoch()`: inference with EMA weights, compute per-class F1 + macro F1, return metrics dict
  - Both return epoch metrics dict (loss, acc, macro_f1, f1_per_class)
- Expected output: Function definitions with docstrings
- Why: Modular functions allow clean k-fold loop and easy debugging
- Notes: EMA inference uses shadow weights via `ema_model`, not `model` directly

---

**Cell 25 — Full 5-Fold Training Loop (Code)**
- Purpose: Train all 5 folds, save best checkpoint per fold, track OOF predictions
- Content:
  ```python
  oof_preds = np.zeros((len(cv_pool), 3))  # probability matrix for OOF

  for fold in range(N_SPLITS):
      # Build train/val DataLoaders for this fold
      # Initialize fresh model + optimizer + scheduler + EMA
      for epoch in range(NUM_EPOCHS):
          train_metrics = train_one_epoch(...)
          val_metrics = validate_epoch(...)
          if val_metrics["macro_f1"] > best_fold_f1:
              save_checkpoint(ema_model, fold, val_metrics["macro_f1"])
      # Store OOF predictions for this fold
      oof_preds[val_indices] = fold_val_probs
  ```
  - Resumable: `kfold_progress.json` tracks completed folds
  - Auto-cleanup: only best checkpoint per fold retained
- Expected output: Per-fold and per-epoch training logs; final summary "Mean CV: X.XXXX ± X.XXXX"
- Why: K-fold ensemble is the MINIMUM VIABLE ensemble strategy. Training all 5 folds gives us 5 diverse models.
- Notes: This cell will NOT re-run if checkpoints already exist (loaded from file instead)

---

**Cell 26 — CV Results Summary (Code + Markdown)**
- Purpose: Print final CV summary table and training history plots
- Content:
  - Table: Fold | Best CV | Best Epoch | F1 Recyclable | F1 Electronic | F1 Organic
  - Plot: Learning curves (train loss, val loss, val macro F1) for all 5 folds
  - Print: Mean ± Std across folds
- Expected output: Table + matplotlib figure
- Why: This is the core result of the notebook — must be clearly presented
- Notes: Include comparison to ablation experiments (exp003: 0.9815, exp005 single-fold: 0.9817, kfold mean: 0.9806)

---

**Cell 27 — OOF Analysis (Code)**
- Purpose: Compute OOF macro F1 and confusion matrix from OOF predictions
- Content:
  - Overall OOF macro F1 (should match kfold mean ≈ 0.9806)
  - Confusion matrix (3×3) with class labels
  - Per-class recall and precision
- Expected output: Confusion matrix plot + per-class metrics table
- Why: OOF analysis is the diagnostic tool that led to Batch 2 cleaning (538 Recyclable↔Organic confusions found)
- Notes: Also show high-confidence error cases (confidence ≥ 0.85) as these were the manually verified mislabels

---

#### **SECTION 6 — INFERENCE PIPELINE**

---

**Cell 28 — Section Header (Markdown)**
- Purpose: Section break
- Content: "## Section 6: Inference Pipeline" + explanation of the two-path inference strategy (hash-override for 97 + ensemble for 1,361)
- Why: Inference strategy is non-trivial and must be documented explicitly

---

**Cell 29 — Load All 5 Fold Checkpoints (Code)**
- Purpose: Load the 5 trained model checkpoints for ensemble inference
- Content:
  ```python
  checkpoints = []
  for fold in range(N_SPLITS):
      model_fold = timm.create_model(MODEL_NAME, pretrained=False, num_classes=3)
      model_fold.load_state_dict(torch.load(f"model_fold{fold}_ema_ls_cv*.pt"))
      model_fold.eval().to(DEVICE)
      checkpoints.append(model_fold)
  ```
- Expected output: "5/5 checkpoints loaded successfully. Fold CVs: [0.9827, 0.9809, 0.9819, 0.9778, 0.9795]"
- Why: All 5 models must be loaded before inference begins
- Dependencies: Cell 25 must have been run

---

**Cell 30 — Hash-Override for Train-Test Duplicates (Code)**
- Purpose: Directly assign labels for the 97 test files that are exact duplicates of train files
- Content:
  ```python
  # Load overlap mapping: test_id → train_label
  overlap_df = pd.read_csv("train_test_overlap.csv")
  override_map = dict(zip(overlap_df["test_id"], overlap_df["train_label"]))
  # override_map will be applied AFTER model inference to ensure correct priority
  ```
- Expected output: "Override map ready: 97 test IDs with known labels (95 Organic, 2 Electronic)"
- Why: These 97 files have KNOWN ground truth (they are identical to labeled train files). Using model prediction here would introduce unnecessary uncertainty.

---

**Cell 31 — Ensemble Inference on Test Set (Code)**
- Purpose: Run all 5 models on the 1,458 test files, average softmax probabilities, argmax for final label
- Content:
  ```python
  # Build test DataLoader (submission.csv order!)
  test_dataset = WasteDataset(submission_df, transform=eval_transform, is_test=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, ...)

  all_probs = np.zeros((1458, 3))
  for model_fold in checkpoints:
      fold_probs = run_inference(model_fold, test_loader)  # [1458, 3]
      all_probs += fold_probs

  all_probs /= N_SPLITS  # average ensemble
  predictions = np.argmax(all_probs, axis=1)

  # Apply hash-override for 97 known duplicates
  for test_id, train_label in override_map.items():
      idx = submission_df[submission_df["id"] == test_id].index[0]
      predictions[idx] = train_label
  ```
- Expected output: "Inference complete. Prediction distribution: Recyclable=X, Electronic=X, Organic=X"
- Why: Ensemble averaging reduces variance across folds. Hash-override ensures known labels are correct.
- Notes: shuffle=False MANDATORY for test loader to maintain submission.csv order

---

#### **SECTION 7 — SUBMISSION GENERATION**

---

**Cell 32 — Section Header (Markdown)**
- Purpose: Section break
- Content: "## Section 7: Submission Generation" + pre-submission sanity checklist

---

**Cell 33 — Submission Validation Checks (Code)**
- Purpose: Run automated sanity checks before generating final CSV
- Content:
  ```python
  # 1. Row count matches exactly
  assert len(predictions) == 1458

  # 2. No NaN or invalid labels
  assert set(np.unique(predictions)).issubset({0, 1, 2})

  # 3. Hash-override correctly applied (97 files)
  # verify each override_map entry

  # 4. ID order unchanged
  assert list(submission_df["id"]) == list(range(1, 1459))

  print("All pre-submission checks PASSED")
  ```
- Expected output: "All pre-submission checks PASSED. Ready to generate submission.csv"
- Why: Zero-waste submission policy — 3 slots total, cannot afford a bad submission from a formatting bug

---

**Cell 34 — Generate Final Submission CSV (Code)**
- Purpose: Write final predictions to submission.csv in correct format
- Content:
  ```python
  submission_df["predicted"] = predictions
  output_path = output_dir / "submission_final_kfold_cleandata.csv"
  submission_df[["id", "predicted"]].to_csv(output_path, index=False)
  print(f"Submission saved: {output_path}")
  print(f"Preview:\n{submission_df[['id', 'predicted']].head(10)}")
  print(f"\nPrediction distribution:\n{submission_df['predicted'].value_counts()}")
  ```
- Expected output: Saved CSV path + preview of first 10 rows + prediction distribution
- Why: Final step of the pipeline — must be clean and unambiguous

---

#### **SECTION 8 — PROJECT RETROSPECTIVE**

---

**Cell 35 — Experiment Log Summary (Markdown)**
- Purpose: Document all experiments run, their results, and decisions made
- Content: Full experiment table (exp001 through kfold-run), ablation conclusions, architecture comparison results
- Why: Demonstrates scientific rigor — the notebook is a complete record of the investigation, not just the final code

---

**Cell 36 — Key Lessons & Insights (Markdown)**
- Purpose: Summarize the most important technical learnings from this project
- Content:
  1. Electronic class is the sole determinant of leaderboard position (paper confirmed Organic/Recyclable near-saturated)
  2. CB-Loss hurt on this dataset — mild imbalance (3.2:1) doesn't justify oversampling ceiling-near class
  3. EMA+LS: small F1 gain but major stability gain — worth it for ensemble consistency
  4. 91.3% of high-confidence model errors were DATA errors — model was right, labels were wrong
  5. StratifiedGroupKFold prevents fold-leakage while maintaining near-perfect class balance
  6. Hash-override for 6.65% of test set = guaranteed correct labels (exploit data structure, not model)
  7. CutMix NOT needed — Grad-CAM confirmed genuine feature learning, not shortcut exploitation
- Why: Closing section of a Kaggle Grandmaster notebook always includes learnings

---

**Cell 37 — Final Note (Markdown)**
- Purpose: Closing acknowledgement and submission plan
- Content: Submission plan (Slot 1: this notebook's output, Slots 2–3 reserved for future improvements), competition constraints reminder (3 submissions, 30 Juli 2026 deadline), team credits (Ababil, Jeremy, Vierico)
- Why: Professional closure; sets up next iteration clearly

---

### NOTEBOOK SUMMARY TABLE

| Cell | Title | Type | Key Output |
|---|---|---|---|
| 00 | Project Overview | Markdown | Competition brief + scope |
| 01 | Configuration Block | Code | All hyperparameters in one place |
| 02 | Environment Setup | Code | data_dir, output_dir paths verified |
| 03 | Library Imports | Code | Versions confirmed |
| 04 | Section Header: Data Integrity | Markdown | — |
| 05 | MD5 Duplicate Detection | Code | 97 overlaps, 57 dup groups, 2 near-dup |
| 06 | Build train_master | Code | 26,527-row manifest with groups + exclusions |
| 07 | Batch Data Cleaning | Code | 63 relabels + 154 drops applied |
| 08 | StratifiedGroupKFold Split | Code | 5-fold assignment, balance verified |
| 09 | Integrity Assertions | Code | All checks PASSED |
| 10 | Test Set Loader Setup | Code | 1,458 files verified in submission order |
| 11 | Section Header: EDA | Markdown | — |
| 12 | Class Distribution + Samples | Code | Bar chart + image grid |
| 13 | Electronic Bimodal Analysis | Code | 68.8% icon vs 31.2% natural |
| 14 | Risk Flags Summary | Markdown | 7 flags + decisions table |
| 15 | Section Header: Pipeline | Markdown | — |
| 16 | Robust RGB Loader | Code | Handles P/RGBA/L modes |
| 17 | Transforms Definition | Code | Train + eval transforms |
| 18 | WasteDataset Class | Code | PyTorch Dataset definition |
| 19 | Section Header: Architecture | Markdown | — |
| 20 | Model Initialization | Code | ConvNeXt V2-Tiny loaded |
| 21 | Discriminative LR Setup | Code | Backbone 1e-5, head 1e-4 |
| 22 | EMA + Loss + Scheduler | Code | EMA + CE+LS + cosine+warmup |
| 23 | Section Header: Training | Markdown | — |
| 24 | Train/Val Helper Functions | Code | train_one_epoch, validate_epoch |
| 25 | Full 5-Fold Training Loop | Code | 5 checkpoints + OOF predictions |
| 26 | CV Results Summary | Code | Table + learning curves |
| 27 | OOF Analysis | Code | Confusion matrix + per-class metrics |
| 28 | Section Header: Inference | Markdown | — |
| 29 | Load 5 Checkpoints | Code | All 5 fold models ready |
| 30 | Hash-Override Setup | Code | 97 test IDs → known labels |
| 31 | Ensemble Inference | Code | 1,458 predictions generated |
| 32 | Section Header: Submission | Markdown | — |
| 33 | Submission Validation | Code | All sanity checks PASSED |
| 34 | Generate submission.csv | Code | Final CSV saved |
| 35 | Experiment Log | Markdown | All experiments documented |
| 36 | Key Lessons | Markdown | 7 critical insights |
| 37 | Final Note | Markdown | Submission plan + team credits |

**Total: 38 cells** (0-indexed Cell 00 through Cell 37)

---

> **Status**: This document represents the complete Phase 1–5 deliverable.
> Notebook implementation can begin as soon as this architecture is approved.
