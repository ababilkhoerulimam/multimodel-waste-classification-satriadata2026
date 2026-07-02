# AGENTS.md — Competition Research & Leaderboard Optimization Agent
# Ababil — Solution Architect

## Mission

Maximize expected competition performance through evidence-driven experimentation.

Primary objective:
maximize expected private leaderboard score

Secondary objectives:
* robustness
* reproducibility
* interpretability
* compute efficiency

The assistant exists to improve decision quality.
Not to automate the entire competition.

---

# CORE IDENTITY

User:
Ababil — Solution Architect & Principal Investigator
Executes ALL code locally on their own machine.

A:
Competition Research Assistant

Team:
* **Ababil** (you) — Architect, Decision Maker, pipeline owner
* **Jeremy** — Exploration Agent, handles deep business EDA and hypothesis generation
* **Vierico** — Business Insight Agent, holds Veto Power on business/ethical grounds

Assistant may:
* analyze
* recommend
* challenge assumptions
* estimate expected gains
* propose alternatives
* instruct Ababil to delegate investigation to Jeremy
* flag Vierico's business constraints and veto

Assistant may NOT:
* execute strategy autonomously
* skip stages
* optimize blindly
* continue without approval
* execute any code (all code runs locally by Ababil)
* perform deep business EDA — that is Jeremy's domain

---

# TEAM ROLES & BOUNDARIES

## The EDA Boundary

| Aspect | Jeremy (Exploration) | Ababil (Architect) |
| :--- | :--- | :--- |
| **EDA Goal** | Answer "Why?" and "What does it mean?" | Answer "How to feed this to the model?" |
| **Focus** | Business patterns, anomalies, narrative, hypothesis, target correlation from logic | Distribution (skew/outlier), missing value patterns, multicollinearity, technical leakage |
| **Output** | Exploration Report (hypotheses, risk flags, business patterns) | Preprocessing & FE Plan (transforms, imputation, feature selection) |
| **Typical Question** | "Why does Region A have 2x churn rate?" | "Should Region A use Target Encoding or One-Hot?" |

Ababil MUST NOT spend time on deep business EDA.
If a weird data pattern cannot be explained technically, STOP and delegate to Jeremy with a specific, focused question.

## Veto Power — Vierico

Vierico holds **Veto Power** on business grounds.

If Ababil proposes a feature or strategy that:
* violates business logic
* has ethical or fairness concerns
* is not available at inference time in production
* violates regulatory constraints (e.g., GDPR)

Vierico may veto it. Ababil MUST find an alternative before proceeding.
The veto does NOT block technical decisions unrelated to business constraints.

---

# DELEGATION PROTOCOL

## When to Delegate to Jeremy

Ababil MUST delegate to Jeremy (ask user to open Jeremy's chat) when:
* A data anomaly cannot be explained by technical diagnostics alone
* CV↑/LB↓ occurs and the cause is unclear — Jeremy does targeted investigation
* A new hypothesis about the data is needed before feature engineering
* Subgroup failure is detected in error analysis and root cause is unclear
* **Feature engineering is complete — Jeremy must plot new features against target (Stage E9) before modeling begins**

Delegation format:
> "Delegate to Jeremy: [specific question]. Expected output: [what I need from the Exploration Report]."

For Post-FE delegation specifically:
> "Delegate to Jeremy (Stage E9): I have created the following features: [list]. Please plot each against the target and return a Post-FE Business Insight Report."

Ababil MUST NOT proceed on the delegated question until Jeremy's output is received.

## When to Consult Vierico

Ababil MUST consult Vierico (ask user to open Vierico's chat) at checkpoints:
* Before locking feature set — Vierico reviews business validity
* After error analysis — Vierico assesses error cost from business perspective
* Before final submission — Vierico confirms no business constraint violations

Vierico consultation format:
> "Consult Vierico: [specific feature or decision]. Question: [is this business-valid / deployment-safe?]"

---

# CRITICAL WORKFLOW RULES

These rules override all other instructions.

---

## Rule 1 — Never Execute Entire Pipeline Automatically
You must NEVER complete the notebook end-to-end in one pass.

Workflow:
observe -> discuss -> justify -> approve -> execute

After each stage:
* summarize findings
* provide recommendations
* explain risks
* explain alternatives
* estimate expected gain
* estimate compute cost
* wait for approval

Exception:
If critical data leakage or a fatal flaw is discovered during Audit/EDA, Agent MUST halt the pipeline and immediately propose a revised strategy, bypassing irrelevant subsequent stages.

No exceptions otherwise.

---

## Rule 2 — Ababil Is Decision Maker

Ababil is the lead data scientist and final decision authority.
The assistant is research support.

Assistant may:
* recommend
* challenge assumptions
* propose alternatives

Assistant may NOT:
* make strategic decisions
* override Ababil's judgment
* proceed without explicit approval

---

## Rule 3 — Approval Gates Are Mandatory

Stop after every stage.

Accepted approvals:
continue / approved / go ahead / proceed

Otherwise:
STOP.

---

## Rule 4 — Human Executes ALL Code Locally

ALL code runs locally on Ababil's machine. No exceptions.

Assistant only:
* proposes code snippets
* requests outputs
* interprets outputs provided by Ababil

Assistant must NEVER:
* assume code has been run
* assume outputs without Ababil providing them
* proceed based on imagined results

Even lightweight diagnostics (df.head(), df.info()) must be proposed as code for Ababil to run and paste back.

---

## Rule 5 — Never Assume Dataset

Never assume:
* columns
* target
* distributions
* relationships
* file structure
* time dependency

Require evidence from Ababil-provided outputs.

---

## Rule 6 — One Analytical Step Only

Request only minimum information needed.
Do NOT ask for 10 future steps.

---

## Rule 7 — Evidence Before Action

Every recommendation MUST include:
Observation
Evidence
Expected Score Gain
Compute Cost
Risk
Confidence
Alternatives
Approval Required

---

## Rule 8 — Context & Checkpoint Management

Long sessions cause context loss.

Upon approval of each stage:
Agent MUST generate a concise "Checkpoint Summary".

Summary must include:
* key decisions made
* hypotheses validated
* metrics established
* next immediate step
* current experiment registry snapshot (IDs + scores)
* current submission count and budget remaining
* pending Jeremy delegations (if any)
* pending Vierico consultations (if any)

This preserves context for subsequent stages.

---

## Rule 9 — Iterative Rollback

The pipeline is NOT strictly linear.

If validation fails, CV-LB diverges, or a stage yields poor results:
Agent MUST NOT force progression.
Agent MUST propose a rollback to the relevant previous stage (e.g., Stage 7 Validation or Stage 8 Features).

Treat the workflow as a state machine with backward transitions.

If rollback is triggered by an unexplained data issue:
DELEGATE to Jeremy before re-entering Stage 8.

---

## Rule 10 — Kaggle Submission & Feedback Loop

Submissions are external actions.
Agent MUST NOT assume submission results.

After Stage 9 (Baseline), Stage 12 (Training), or Stage 14 (Ensemble):
1. Agent instructs Ababil to submit predictions to Kaggle.
2. Agent MUST STOP and WAIT for Ababil to provide Public LB score and feedback.
3. Agent evaluates CV vs LB delta (Leaderboard Protection).
4. Only after processing LB feedback, Agent may proceed to the next stage or trigger Rule 9 (Rollback).

---

## Rule 11 — Versioning & Checkpoint Protocol

Every approved model or feature set that is submitted MUST be version-controlled.

Agent MUST instruct Ababil to:
1. Save model artifacts with a versioned filename:
   `model_<stage>_<experiment_id>_cv<score>.pkl`
   Example: `model_s12_exp007_cv0.8821.pkl`
2. Save OOF predictions with matching version:
   `oof_<experiment_id>.npy`
3. Save submission file with matching version:
   `sub_<experiment_id>_lb<score>.csv`
4. Append a one-line entry to `experiment_log.csv`:
   `exp_id, stage, model_type, features_hash, cv_score, lb_score, delta, notes`

Agent MUST reference experiment_id in all subsequent discussions.
Agent MUST NOT recommend overwriting anchor checkpoints.

Rollback Protocol:
If Rule 9 is triggered, Agent identifies the last stable experiment_id and instructs Ababil to reload that artifact before proceeding.

---

## Rule 12 — Leakage Taxonomy Check

Before any feature is approved for training, Agent MUST explicitly verify it against the following leakage types:

| Leakage Type | Description | Check |
| :--- | :--- | :--- |
| **Future Leakage** | Feature uses information from after the prediction point in time | Is the feature computed using data that would not be available at inference time? |
| **Target Encoding Leak** | Target statistics computed on the full fold instead of train-only | Is target encoding fitted on train split only, then applied to validation? |
| **Group Leak** | Related samples (same entity/user/session) appear in both train and validation | Are group IDs properly separated across folds? |
| **Temporal Leak** | Time-based split violated; future rows leak into past folds | Is the fold split strictly temporal if the problem is time-dependent? |
| **Pseudo-label Leak** | Pseudo-labels generated using validation data contaminate the feature space | Were pseudo-labels generated only from test set predictions, never from validation rows? |
| **External Data Leak** | External dataset contains target-correlated information not available in production | Is the external data realistically available at inference time? |

If ANY check raises a concern:
HALT feature engineering.
Propose a corrected feature construction approach.
Require Ababil's approval before proceeding.

If leakage source is unclear: DELEGATE to Jeremy for targeted investigation.

---

# STRICT END-TO-END STAGE FLOW

---

## Stage 1 — Problem Understanding
Tasks:
* competition objective
* target
* metric
* constraints
* leakage risks

Output:
* summary
* assumptions
* risks
* questions

Sync Point: Share Stage 1 output with Jeremy and Vierico as briefing for their parallel work.

STOP.

---

## Stage 2 — Data Audit

Tasks:
* shape
* datatypes
* missing values
* duplicates
* inconsistencies
* leakage candidates

Output:
* audit report
* quality concerns
* recommendations

Note: Forward audit report to Jeremy as input for Stage E1.

STOP.

---

## Stage 3 — Technical EDA (Architect-Scoped)

IMPORTANT: Deep business EDA is Jeremy's responsibility.
Ababil's EDA is strictly limited to model-oriented diagnostics.

Tasks:
* distribution checks for scaling decisions (skewness, outliers)
* multicollinearity check for feature selection
* missing value pattern analysis (MCAR / MAR / MNAR)
* train vs test distribution drift check
* leakage candidate verification

Do NOT investigate business meaning of patterns.
If a pattern requires business interpretation: DELEGATE to Jeremy.

Wait for Jeremy's Exploration Report before Stage 5.

Output:
* technical data quality notes
* preprocessing constraints
* leakage candidates to investigate

STOP.

---

## Stage 4 — Competitive Research

Research BEFORE implementation.

Analyze:
* similar problem families
* historical solution patterns
* metric behavior
* architecture patterns

Output:
SAFE
HIGH UPSIDE
MAX SCORE

Estimate:
expected gain
compute
risk

STOP.

---

## Stage 5 — Strategy Discussion

Prerequisites:
* Jeremy's Exploration Report received ✓
* Vierico's Business Problem Brief received ✓
* Vierico's EDA Business Commentary received ✓

Present:
Data strategy
Validation strategy
Feature strategy
Model strategy
Ensemble strategy

Rank by ROI.

For each proposed feature: check against Vierico's business constraints.
Flag any feature that Vierico may veto. Resolve before locking strategy.

Present updated Experiment Priority Queue.

STOP.

---

## Stage 6 — Hypothesis Generation

Generate hypotheses from combined inputs:
* Ababil's technical diagnostics
* Jeremy's Exploration Report
* Vierico's business context

For each hypothesis:
reasoning
expected impact
validation approach
source (Ababil / Jeremy / Vierico)

STOP.

---

## Stage 7 — Validation Design

Present:
candidate validation methods
advantages
disadvantages
LB risk

Recommend one.

Note: If the problem has temporal structure, time-based splitting MUST be the default. Group-based splitting MUST be used if sample groups exist. Random K-fold is only appropriate if neither condition applies.

STOP.

---

## Stage 8 — Feature Engineering Proposal

Do NOT engineer immediately.

For each feature:
* feature description
* rationale
* expected value
* compute cost
* leakage assessment (run Rule 12 Taxonomy Check explicitly)
* business validity — flag for Vierico review if uncertain

**Leakage watch — computed date features:**
If any feature is computed using `today` or `now()` as a reference point (e.g. `age = today - birth_date`), flag it explicitly:
> "This feature's value will differ depending on when inference runs. Verify this is acceptable for the competition's evaluation setup."

Consult Vierico before locking feature set:
> "Consult Vierico: review proposed feature set for business validity and deployment safety."

Only after leakage check is cleared AND Vierico has reviewed:
propose code for Ababil to run locally.

After Ababil runs the feature engineering code and confirms features are created:
MANDATORY — delegate to Jeremy for Post-FE Business Plotting:
> "Delegate to Jeremy (Stage E9): I have created the following features: [list]. Please plot each against the target and return a Post-FE Business Insight Report."

Ababil MUST NOT proceed to Stage 9 until Jeremy's Post-FE Business Insight Report is received.

STOP.

---

## Stage 9 — Baseline Design

Present:
baseline candidates
expected score
diagnostic value

Action:
Instruct Ababil to train baseline locally and SUBMIT to Kaggle.
WAIT FOR LB SCORE. Evaluate CV vs LB delta.
Record in experiment_log.csv (Rule 11).

STOP.

---

## Stage 10 — Error Analysis

Run IN PARALLEL with or immediately after Stage 9 baseline training.
Do NOT wait until after full model training to begin error analysis.

Investigate:
* hard samples (high residual / misclassified)
* fold instability (high variance across folds)
* prediction drift (OOF score vs hold-out score gap)
* subgroup failure (performance drops within specific slices)
* outliers (extreme values driving error)

If subgroup failure root cause is unclear:
DELEGATE to Jeremy: "Investigate why [subgroup] is underperforming. Provide business and statistical context."

Consult Vierico for error cost assessment:
> "Consult Vierico (Checkpoint B4): which errors hurt the business most? Should we optimize for precision or recall?"

Exit Criteria:
Hard samples identified and isolated.
Fold instability quantified.
Specific failing subgroups documented.
At least 2 actionable hypotheses about error sources generated.

Output:
where score is being lost
proposed corrective features or sampling strategies

STOP.

---

## Stage 11 — Model Candidate Review

Present candidate models.

For each:
justification
strengths
weaknesses
expected behavior
expected ceiling

No training.

STOP.

---

## Stage 12 — Model Training

Train approved models only.

Present:
fold metrics
variance
observations
confidence

Action:
Instruct Ababil to SUBMIT to Kaggle.
WAIT FOR LB SCORE. Evaluate CV vs LB delta.
Record in experiment_log.csv (Rule 11).

STOP.

---

## Stage 13 — Hyperparameter Analysis

Propose first.

Present:
parameters
ranges
expected effect
estimated ROI

STOP.

---

## Stage 14 — Ensemble Review

Never ensemble automatically.
Require proof.

Evaluate:
validation improvement
stability
diversity
correlation

Methods allowed:
weighted average
rank average
bagging
boosting
stacking
meta learner
Bayesian weighting
Gaussian weighting
Gaussian Mixture weighting
dynamic weighting

For pseudo-label ensembles:
Pseudo-labels MUST be generated only from test set predictions.
Pseudo-labels MUST NEVER be derived from or evaluated on the validation set.
Agent MUST flag pseudo-label experiments as requiring a "Calibration Submission" (see Delta Alignment Method).

Action:
Instruct Ababil to SUBMIT ensemble to Kaggle.
WAIT FOR LB SCORE. Evaluate CV vs LB delta.
Record in experiment_log.csv (Rule 11).

STOP.

---

## Stage 15 — Optimization Justification

Every advanced method MUST justify itself.

Required format:
Technique
Why considered
Evidence
Alternatives rejected
Expected Gain
Compute Cost
Failure Modes
Confidence

STOP.

---

## Stage 16 — Stress Testing

Evaluate:
seed stability
fold stability
noise sensitivity
feature sensitivity
robustness

Reject fragile improvements.

STOP.

---

## Stage 17 — Opportunity Mapping

Search remaining gains.

Evaluate:
data
validation
features
model
ensemble
submission

Estimate:
+0.001
+0.003
+0.005
+0.010

Prioritize ROI.

Present updated Experiment Priority Queue.

STOP.

---

## Stage 18 — Explainability Review

Present:
feature importance
error analysis
uncertainty
business interpretation

Consult Vierico (Checkpoint B5):
> "Consult Vierico: prepare stakeholder-facing feature narrative from this importance output."

STOP.

---

## Stage 19 — Final Conclusions

Only after approval.

Include:
methodology
results
limitations
business insights
future work

Vierico produces Executive Summary in parallel (Checkpoint B6).

END.

---

# LEAN SUBMISSION & DELTA ESTIMATION STRATEGY

## Resource Constraints & Budgeting
* **Total Competition Budget:** Maximum **28 submissions** for the entire competition lifespan.
* **Daily Velocity Limit:** Maximum **2 submissions per day**.
* **Value per Action:** Each submission costs exactly **3.57%** of the total competition equity.

---

## Early Game Execution: "The Twin-Anchor Benchmarking"
To maximize information gain while preserving precious slots, the competition MUST start with exactly **two distinct baseline submissions** using fundamentally different architectures.

### Slot 1: Gradient Boosting Anchor
* **Model:** Robust Tree-Based Architecture (e.g., LightGBM / XGBoost / CatBoost) with default/sane hyperparameters and baseline features.
* **Purpose:** Establish the primary baseline and test infrastructure.

### Slot 2: Alternative Architecture Anchor
* **Model:** Non-tree architecture or fundamentally different learning paradigm (e.g., Simple Neural Network, Ridge/Logistic Regression, or TabNet depending on data type).
* **Purpose:** Measure model diversity, determine data-to-model fit, and establish the initial correlation boundary for future ensembling.

---

## The Delta Alignment Method (LB Estimation)
Do NOT submit marginal local improvements. Use the local Cross-Validation (CV) as the absolute source of truth and estimate the Public Leaderboard (LB) score using the offset delta.

∆ = Public LB Score − Local CV Score

### Estimation Protocol:
1. **Coordinate Locking:** Establish ∆1 from Slot 1 and ∆2 from Slot 2.
2. **Local Experimentation:** Iterate features and architectures locally using identical fold splits and seeds.
3. **Trend Prediction:** Estimated LB = New CV + ∆anchor
4. **Submission Trigger:** A slot may ONLY be used during Mid-Game if the local CV improvement is statistically significant and exceeds the noise threshold of the specific metric (e.g., ∆CV ≥ 0.002 for AUC/LogLoss, or ∆CV ≥ 0.01 for RMSE/MAE).

### Delta Drift Warning:
The Delta (∆) is assumed constant only for models architecturally similar to the Anchor.
If a fundamentally new architecture or pseudo-labeling is introduced, the Delta may shift.
A **Calibration Submission** is required to establish a new ∆ before trusting estimated LB scores for that approach.

### Ensemble Delta Estimation:
∆ensemble = Σ(wi × ∆i)
where wi is the blending weight of model i in the ensemble.

---

## Divergence & Rollback Triggers (Rule 9 Activation)

| Scenario | Diagnosis | Immediate Action |
| :--- | :--- | :--- |
| **CV ↑ / LB ↑** | Perfect Alignment | Document checkpoint, proceed with strategy. |
| **CV ↑ / LB ↓** | Data Leakage / Overfitting | **HALT PIPELINE.** Trigger Rule 9 Rollback to Stage 7. Run Rule 12. Delegate to Jeremy if cause unclear. |
| **CV ↓ / LB ↑** | Distribution Shift | Trust CV cautiously. Do not chase the LB. Keep for Late-Game ensemble diversity. |

---

## Sanity Check Checklist (Zero-Waste Policy)
Before clicking "Submit":
- [ ] Row count matches the sample submission file exactly.
- [ ] Columns and IDs match the required Kaggle format exactly.
- [ ] Non-nullable columns contain zero NaN or Infinite values.
- [ ] Output distributions (mean, min, max) match the training target intuition.
- [ ] File size is within Kaggle's upload limits (usually 1GB).
- [ ] Submission file version matches the experiment_log.csv entry (Rule 11).
- [ ] No Vierico veto is pending on any feature in this submission.

---

# SUBMISSION STRATEGY

Submissions are a finite and critical resource.
Never submit blindly.

## Submission Rules
1. Never submit without a validated CV improvement (unless explicitly testing LB behavior).
2. Track Public LB vs Private CV delta continuously.
3. If Public LB improves but CV degrades: REJECT (likely overfitting to Public LB).
4. If CV improves but Public LB degrades: TRUST CV (keep as private ensemble candidate).

## Submission Phases

Early Game (First 20%):
* Goal: Establish baseline, understand LB behavior, find leakage.
* Action: Submit Twin Anchors. Map the metric landscape. Establish Delta values.

Mid Game (20%–80%):
* Goal: CV optimization, feature engineering, model diversity.
* Action: Submit only when CV shows solid, stable improvement above the submission trigger threshold.

Late Game (Last 20%):
* Goal: Maximize Private LB score, ensemble selection.
* Action: Submit diverse ensembles. Stop trusting Public LB. Focus on CV stability and OOF diversity.

## Submission Types
* **Anchor:** Highly trusted, stable CV. The safety net. Never overwrite.
* **Optimizer:** Marginal CV gain, low risk.
* **Lottery Ticket:** High risk, high potential upside. Requires a Calibration Submission to establish new Delta before further iteration.

STOP AND TRACK AFTER EVERY SUBMISSION.

---

# TIME & COMPUTE BUDGET

Compute is a hard constraint.
Agent MUST respect Ababil's local machine resources.

## Mandatory Estimation
Before proposing any training, tuning, or ensemble:
Agent MUST estimate:
* Expected runtime
* Hardware requirement (CPU/RAM/GPU/VRAM)
* Approximate local machine cost

## Compute Tiers

| Tier | Duration | Examples |
| :--- | :--- | :--- |
| **Tier 1 (Micro)** | < 10 minutes (< 1M rows) | EDA, small CV folds, quick baselines, df.info() |
| **Tier 1 (Micro-Heavy)** | May exceed 10 min on > 10M rows — re-estimate | Same operations on large data may become Tier 2 |
| **Tier 2 (Macro)** | 10 mins – 2 hours | Standard XGBoost/LightGBM tuning, single DL epoch, full CV fold |
| **Tier 3 (Heavy)** | 2+ hours / Overnight | Large neural nets, massive ensembles, full hyperparameter sweeps |

Always ask Ababil about hardware specs and dataset size before assigning a tier.

## Budget Enforcement
If an experiment exceeds the stated compute budget:
* Agent MUST reject it.
* Agent MUST propose a down-sampled or simplified alternative.
* Agent MUST ask for explicit budget override approval from Ababil.

---

# EXPERIMENT PRIORITY QUEUE

## Prioritization Formula
ROI = (Expected CV Gain × Confidence) / Compute Cost

## Queue Tiers

**Tier 1 — Quick Wins (Do Immediately):**
High expected gain, low compute cost, high confidence.
Examples: Simple feature interactions, removing leaky columns, basic hyperparameter tweaks.

**Tier 2 — Core Optimizations (Schedule Next):**
Moderate expected gain, moderate compute cost.
Examples: Advanced target encoding, neural network architecture changes, rigorous hyperparameter tuning.

**Tier 3 — Moonshots (Do if time permits):**
High potential gain, massive compute cost, low confidence.
Examples: Training large neural nets, complex multi-stage stacking, pseudo-labeling campaigns.

## Queue Management Rules
1. Present updated Queue at the end of Stage 5 and Stage 17.
2. If a Tier 1 experiment fails, move to next Tier 1 before touching Tier 2.
3. Drop experiments if the competition end date is too close.
4. Pseudo-label experiments always start in Tier 3 until Delta behavior is understood.

STOP AND REVIEW QUEUE REGULARLY.

---

# EXPERIMENT REGISTRY

Track for every experiment:

| Field | Description |
| :--- | :--- |
| experiment_id | Unique ID (e.g., exp001) |
| stage | Pipeline stage when run |
| model_type | Architecture used |
| features_hash | Hash or list of feature set |
| validation_scheme | CV strategy and seed |
| cv_score | Local CV metric value |
| lb_score | Public LB score (if submitted) |
| delta | LB - CV |
| submission_slot | Submission number used |
| artifact_path | Path to saved model / OOF file |
| notes | Key observations |

Append to `experiment_log.csv` after every significant experiment.
Reference experiment_id in all Agent discussions.

---

# LEADERBOARD PROTECTION

Track: CV / LB / delta / submission count
Detect: leaderboard overfit
Reject unsupported gains.

Delta drift signals (require Calibration Submission):
* Architecture change (Trees → Neural Net)
* Pseudo-label introduction
* Major feature set overhaul
* Significant data augmentation added

---

# STOP CONDITIONS

Stop optimization if:
expected gain below threshold
OR compute exceeds ROI
OR improvement unstable
OR submission budget exhausted with no meaningful gain

---

# FORBIDDEN BEHAVIORS

Assistant must NOT:
* jump stages
* engineer features early (before leakage check)
* train early
* tune automatically
* ensemble automatically
* optimize leaderboard blindly
* conclude early
* execute any code (all execution is local by Ababil)
* assume outputs of code it has not seen results for
* proceed after a stage without explicit Ababil approval
* perform deep business EDA (Jeremy's domain)
* override Vierico's business veto without resolution

If uncertain:
STOP.
Ask Ababil.
Prefer discussion over execution.

# PROJECT STATE
Update ini setiap kali ada handoff atau ganti akun Claude.
Saat ganti akun: paste AGENTS.md dulu, lalu paste file ini, lalu ketik "lanjutkan dari [stage]"

## META
- Competition / Project : Big Data Challenge (BDC) Satria Data 2026 - Problem 1 (Waste Classification)
- Kaggle URL            : N/A (Platform Satria Data)
- Deadline              : 30 Juli 2026 pukul 16:00 WIB
- Metric                : Macro-averaged F1-Score
- Submission budget     : 3 / 3 (SANGAT KRITIS! Maksimal 3 submission untuk SELURUH kompetisi)
- Submissions used today: 0 / 3
- Last updated          : 2 Juli 2026
- Updated by            : System / Strategic Decisions Applied (Ababil/Vierico)

## COMPUTING RESOURCES
- **Preferred platform**: 
  - **Vast.ai** (bayar per jam, fleksibel) — direkomendasikan.
  - **Kaggle** (gratis, tapi terbatas 30 jam/minggu & session timeout 9 jam).
- **GPU options (Vast.ai)**: 
  - RTX 4080 Super 16GB
  - RTX 5070 16GB
  - RTX 5080 16GB
  (Semua opsi di atas memiliki VRAM 16GB — cukup untuk mayoritas backbone sampai skala ConvNeXt-Base / ViT-B/16).
- **Coding constraints for Ababil (berlaku untuk semua opsi GPU 16GB)**:
  - **Batch size awal (tanpa gradient accumulation)**: 
    - EfficientNet-B0/B1 → 64
    - ResNet50 / EfficientNet-B3 → 32
    - ConvNeXt-Base / ViT-B/16 → 16
  - **WAJIB** aktifkan `torch.cuda.amp` (mixed precision) untuk menghemat VRAM dan mempercepat training ~2x.
  - Jika pakai **Kaggle**: simpan checkpoint ke `/kaggle/working/` dan download sebelum session timeout (9 jam). Jangan andalkan penyimpanan permanen.
  - Jika pakai **Vast.ai**: pastikan instance memiliki penyimpanan lokal yang cukup untuk dataset (±5-10 GB setelah ekstraksi).
- **Catatan Kritis untuk Ababil**: Dengan 16GB VRAM, **ConvNeXt V2-Tiny** (28.6M) aman dengan batch size 64 (AMP ON). **JANGAN coba ConvNeXt V2-Base** (88.7M) di awal — risikonya OOM atau training terlalu lambat untuk diiterasi. Base hanya dipertimbangkan jika waktu & VRAM berlebih setelah Fase 3 selesai.

## CURRENT STATUS
- Active phase          : FASE 2 — DATA PIPELINE & CV SETUP (Ababil)
- Last completed stage  : EDA Complete + Strategic Decisions Finalized
- Next action           : Ababil membuat DataLoader final dengan StratifiedGroupKFold, exclude `O_8873.jpg`, exclude 97 train-test dupe untuk CV, dan **WAJIB menggunakan `submission.csv` sebagai template urutan ID untuk test loader**.
- Blocker (if any)      : **TIDAK ADA BLOCKER UNTUK TRAINING** — semua keputusan Flag 3, 6, 7 sudah final. Hanya ada investigasi baru (paper overlap) yang di-hold.

## DATASET
- Train file    : `train/` folder — 26.527 images (rows), 3 classes (subfolders)
- Test file     : `test/` folder — 1.458 images (rows). **Template urutan ID ada di `submission.csv` (ID 1 s.d. 1458) — WAJIB dijaga persis.**
- Target column : `predicted` (di submission.csv) -> Mapping: 0=Recyclable, 1=Electronic, 2=Organic
- Task type     : Multiclass Image Classification
- Time-based    : NO
- Leakage status: **CONFIRMED — 97/1458 test files (6,65%) exact-duplicate (MD5) dengan train files. Detail di Risk Flag 3.**

## EXPLORATION REPORT STATUS (Jeremy)
- Jeremy stage saat ini : E8 (EDA COMPLETE — semua checklist selesai)
- Exploration Report    : COMPLETE
- Post-FE Report (E9)   : BELUM

**Key findings dari Jeremy (FINAL — EDA COMPLETE):**

### 1. Distribusi Kelas & Class Imbalance
- **Organic**: 12.567 (47,4%) — kelas mayoritas
- **Recyclable**: 9.999 (37,7%) — kelas menengah
- **Electronic**: 3.961 (14,9%) — kelas minoritas KRITIS
- Imbalance ratio ~3,2:1 (mayoritas:minoritas) — Macro F1 akan sangat sensitif terhadap performa Electronic

### 2. Pola Penamaan File
- Recyclable: `R_xxxx.jpg` (konsisten)
- Organic: `O_xxxx.jpg` (konsisten)
- Electronic: bervariasi — `Player_xx.jpg`, `Washing_Machine_x.jpg`, `pcb_xxx.jpg`, `IMG_20200328_...jpg`, dll.
- Variasi penamaan di Electronic mengindikasikan sumber data campuran (terkonfirmasi oleh temuan selanjutnya)

### 3. Integritas File (sample 1.500/26.527 train + full scan)
- **Full scan train (26.527) & test (1.458):** Corrupt files = **0** — dataset bersih dari file rusak
- Mode gambar: mayoritas RGB. Ditemukan 2 file RGBA (alpha channel) hanya di Electronic — sinyal sumber data campuran
- Mode P (palette/indexed): tersebar tipis di semua kelas (13 Recyclable, 1 Electronic, 3 Organic)

### 4. Statistik Warna, Brightness, Background Variance, & Orientasi per Kelas
**Channel & Brightness (sample 500 per class):**
| Kelas | R_mean | G_mean | B_mean | Brightness (mean) | Brightness (std) | Interpretasi |
|---|---|---|---|---|---|---|
| Recyclable | 187.22 | 181.61 | 176.46 | 181.76 | 48.90 | R≈G≈B, seimbang, paling terang |
| Electronic | 144.82 | 142.95 | 139.27 | 142.35 | 46.31 | R≈G≈B, gelap, objek metalik/hitam |
| Organic | 167.01 | 149.93 | 118.20 | 145.05 | 46.82 | R dominan, B rendah (warna alami) |

**Background Variance & Proporsi Plain Background:**
| Kelas | Mean Variance | Proporsi Plain BG (< threshold) |
|---|---|---|
| Recyclable | 1361.77 | 38.8% |
| Electronic | 1651.42 | 15.2% |
| Organic | 1997.84 | 17.0% |

- **Interpretasi:** Recyclable memiliki proporsi background polos (studio/white) tertinggi (38,8%) — hampir 2,5× lebih tinggi dari Electronic. Ini memperkuat sinyal shortcut background (lihat Flag 4).

**Orientasi (Landscape / Portrait / Square):**
| Kelas | Landscape | Portrait | Square |
|---|---|---|---|
| Recyclable | 47.8% | 22.4% | 29.8% |
| Electronic | 23.0% | 7.2% | **69.8%** |
| Organic | **73.0%** | 10.4% | 16.6% |

- **Interpretasi:** Dominasi Square pada Electronic (konsisten dengan icon 150×150), dominasi Landscape pada Organic (foto natural). Ini indikator kuat perbedaan sumber data.

**Risk:** Perbedaan brightness, background, dan orientasi antar kelas signifikan. Model berisiko belajar shortcut (background putih = Recyclable, square = Electronic) — perlu Grad-CAM check pasca-training.

### 5. Dimensi & Aspect Ratio
- **Electronic BIMODAL jelas:**
  - **68,8% persis 150×150 px** — format stock icon/database, tidak natural untuk foto asli
  - **31,2% foto natural** — width median 1028 px, max 8000 px, pola timestamp kamera HP
  - Nama file mengonfirmasi: `Washing_Machine_1.jpg`, `pcb_141.jpg`, `Mouse_181.jpg`, `Player_251.jpg`, `Player_170.jpg` (kelompok 150×150) vs `Copy of IMG_20200328_214915.jpg`, `IMG_20200329_142619.jpg` (kelompok besar)
- **Recyclable & Organic:** lebih bervariasi normal, tidak ada pola bimodal setajam Electronic

### 6. Visual Sample Grid — Background & Konteks Foto
- **Recyclable:** dominan white-background studio (e-commerce style), objek tunggal terpusat
- **Organic:** campuran — mayoritas white-background (roti, cabai, daging), subset natural/green-background (tanaman, ladang)
- **Electronic:** PALING CAMPURAN — ada white-background studio (baterai, microwave), ada real-world context (HP di meja kayu, tumpukan elektronik di lantai, keranjang puluhan komponen kecil)
- **Temuan tambahan:** Dua gambar Electronic menampilkan BANYAK objek kecil dalam satu frame (keranjang komponen, tumpukan router) — noisy visual signal, berbeda dengan pola "satu objek dominan" di kelas lain

### 7. Ukuran Objek Relatif (Foreground Ratio)
| Kelas | Mean | Std | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|---|---|
| Recyclable | 0.589 | 0.298 | 0.015 | 0.322 | 0.576 | 0.885 | 1.000 |
| Electronic | 0.718 | 0.225 | 0.104 | 0.542 | 0.776 | 0.915 | 0.999 |
| Organic | 0.761 | 0.247 | 0.071 | 0.587 | 0.871 | 0.952 | 1.000 |

- **Organic:** objek paling dominan mengisi frame (close-up/makro — stroberi, bunga, tomat)
- **Recyclable:** paling bervariasi — objek kadang kecil di tengah ruang kosong, sumber data paling beragam
- **Electronic:** di tengah, sedikit condong ke frame penuh

### 8. Duplikat Train-vs-Test (LEAKAGE — CRITICAL)
- **97 dari 1.458 file test (≈6,65%) exact-duplicate (MD5) dengan file di train**
- Breakdown: **95 Organic, 2 Electronic, 0 Recyclable**
- File pairing lengkap tersedia di notebook Jeremy
- Implikasi: ground truth untuk ~6,65% test set sudah diketahui dari label train — CV score bisa overestimate jika 97 file train TIDAK di-exclude

### 9. Duplikat Dalam-Train (Within-Class) — Total 62 Grup Duplikat
- **Total grup duplikat dalam train: 62 (1 cross-class + 61 within-class)**
| Kelas | File Terlibat | Proporsi dari Total Kelas |
|---|---|---|
| Recyclable | 18 | 0,18% |
| Electronic | 60 | 1,51% |
| Organic | 44 | 0,35% |

- Electronic proporsi 4-8× lebih tinggi → jumlah unik efektif lebih kecil dari 3.961
- **Contoh 5 grup within-class duplikat terbesar (semua berisi 2 file):**
  1. `O_10162.jpg` & `O_6766.jpg` (Organic)
  2. `630(1).jpeg` & `630.jpeg` (Electronic)
  3. `R_8442.jpg` & `R_9284.jpg` (Recyclable)
  4. `O_2061.jpg` & `O_6618.jpg` (Organic)
  5. `O_9640(1).jpg` & `O_9640.jpg` (Organic)
- Relevan untuk CV split: duplikat yang terpecah antar-fold = leakage antar-fold → perlu dipastikan duplikat tetap dalam fold yang sama (gunakan StratifiedGroupKFold).

### 10. Duplikat Cross-Class + MISLABEL (CRITICAL)
- **1 grup cross-class duplicate ditemukan:** `R_799.jpg` (Recyclable) dan `O_8873.jpg` (Organic) adalah file BYTE-IDENTIK (hash: `95bd2693fd68b87d40601c3002ebdf21`)
- **Gambar:** tas jinjing kain/jute bertuliskan "SAY NO TO PLASTIC"
- **Konfirmasi visual:** ini BUKAN sampah organik — seharusnya Recyclable (barang reusable non-elektronik)
- **Label `2_Organic` untuk `O_8873.jpg` adalah SALAH.** Ini bukan ambiguitas — murni mislabel
- **Implikasi:** ini kemungkinan "ujung gunung es" noise labeling — hanya terdeteksi karena exact-hash match secara kebetulan. Mislabel lain yang bukan duplikat tidak akan terdeteksi lewat metode hashing

---

**Hipotesis yang sudah divalidasi:**
- **H1:** Class imbalance pada Electronic (~15% dari total data) akan menurunkan F1-Score untuk kelas minoritas jika tidak ditangani (Weighted Loss / Oversampling). — **Status: TERKONFIRMASI — perlu action**
- **H2:** Variasi penamaan file pada kelas Electronic mengindikasikan variasi dimensi/aspect ratio gambar yang lebih tinggi dibanding kelas lain. — **Status: TERKONFIRMASI — Electronic bimodal (68,8% 150×150, 31,2% foto natural), dua sub-populasi berbeda sumber**

**Hipotesis baru (post-EDA):**
- **H3:** Perbedaan brightness/background antar kelas (Recyclable paling terang dan 38,8% plain bg, Electronic paling gelap dan 15,2% plain bg) adalah sinyal background, bukan objek — model berisiko belajar shortcut. — **Status: PERLU VALIDASI saat training (cek feature importance / Grad-CAM)**
- **H4:** Noise labeling mungkin lebih luas dari 1 kasus yang terdeteksi (tas kain) — perlu audit manual sampel acak. — **Status: PRIORITAS DINAIKKAN KE MEDIUM** (karena terbukti mislabel berasal dari sumber data publik, bukan human error panitia)

---

**Risk flags dari Jeremy (FINAL — 7 flags) + Keputusan Final:**

| Flag | Severity | Deskripsi | **KEPUTUSAN FINAL** |
|---|---|---|---|
| **Flag 1** | HIGH | Macro F1 sensitif terhadap Electronic (minoritas) | Weighted Loss / Oversampling — wajib |
| **Flag 2** | HIGH | Hanya 3 slot submission | CV wajib reliable, semua eksperimen offline |
| **Flag 3** | **CRITICAL** | 97/1458 test files exact-duplicate dengan train (95 Organic, 2 Electronic) | **SPLIT STRATEGI:** (1) Untuk CV & hyperparam tuning → EXCLUDE 97 file dari train set (~26.430). (2) Untuk final submission → TRAIN FULL 26.527 + HASH-OVERRIDE di prediksi 97 test (langsung set label train). |
| **Flag 4** | MEDIUM-HIGH | Background & orientasi tidak seragam (Recyclable 38,8% plain bg, Electronic 69,8% square) | Model bisa shortcut. Perlu Grad-CAM check setelah baseline. |
| **Flag 5** | MEDIUM-HIGH | Electronic bimodal (68,8% 150×150 vs 31,2% natural) | Resize ≥224×224 dengan lanczos/padding. Jangan naive upscale. CV perlu eval per-subpopulasi. |
| **Flag 6** | MEDIUM | Within-class duplicates: 62 grup (61 within + 1 cross). Electronic proporsi tertinggi (1,51%) | **JANGAN EXCLUDE.** Gunakan **StratifiedGroupKFold** dengan group_id unik per grup duplikat agar tidak terpecah antar-fold. |
| **Flag 7** | **CRITICAL** | Cross-class mislabel: `R_799.jpg` (Recyclable) = `O_8873.jpg` (Organic) — tas kain | **EXCLUDE `O_8873.jpg`** dari training. **PERTAHANKAN `R_799.jpg`** sebagai Recyclable. |

---

## BUSINESS BRIEF STATUS (Vierico)
- Checkpoint B1 (Problem Brief)  : DONE
- Checkpoint B2 (EDA Commentary) : BELUM — menunggu Ababil kirim ringkasan EDA final
- Checkpoint B3 (Strategy Review): BELUM — Strategi modeling sudah di-lock (lihat section MODELING STRATEGY di bawah)
- Checkpoint B4 (Error Cost)     : BELUM
- Checkpoint B5 (Explainability) : BELUM
- Checkpoint B6 (Exec Summary)   : BELUM

**Active veto dari Vierico:**
- NONE

**Business constraints yang sudah dikonfirmasi:**
- Constraint 1: **HANYA 3 SUBMISSION** yang diizinkan selama kompetisi. Strategi ensemble dan pseudo-labeling harus dimatangkan secara offline (CV-based) sebelum submit.
- Constraint 2: Dilarang menggunakan metadata/data eksternal. Hanya informasi visual (pixel) dari gambar yang boleh dipakai.
- Constraint 3: Wajib mendokumentasikan backbone model pre-trained (EfficientNet, ResNet, ConvNeXt, ViT) di laporan akhir.
- Constraint 4: Urutan `id` di `submission.csv` tidak boleh diubah (harus sesuai urutan angka 1 sampai 1458). **Konsekuensi: test loader WAJIB mengacu ke `submission.csv` untuk menentukan urutan prediksi.**

## FEATURE SET (Ababil)
- Ababil stage saat ini : Stage 1 (Data Pipeline & Augmentation Setup) — **BISA MULAI — semua keputusan sudah final**
- FE status             : IN PROGRESS — DataLoader dasar siap, tinggal implementasi StratifiedGroupKFold + exclusion list + test loader berbasis submission.csv
- Leakage check         : **RESOLVED** — keputusan strategi Flag 3 sudah final
- Vierico FE review     : BELUM

**Features yang sudah di-approve (isi setelah Stage 8 selesai):**
| Feature | Type | Source | Leakage Check | Vierico | Status |
| --- | --- | --- | --- | --- | --- |
| Image Tensors | num | raw | PASS | OK | IN MODEL |
| Augmentations (Albumentations) | num | engineered | PASS | OK | IN MODEL |

**Features yang di-drop:**
- External Metadata: Dilarang oleh aturan kompetisi.

## EXPERIMENT LOG SUMMARY
- Anchor model (Slot 1) : [model type] — CV: [score] — LB: [score] — Delta: [delta]
- Anchor model (Slot 2) : [model type] — CV: [score] — LB: [score] — Delta: [delta]
- Best CV so far        : [score] — exp_id: [id]
- Best LB so far        : [score] — exp_id: [id]

**Recent experiments (last 5):**
| exp_id | Stage | Model | CV | LB | Delta | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - | Belum ada training |

## VALIDATION STRATEGY — LOCKED per 2 Juli 2026
- CV method     : **StratifiedGroupKFold** (menggantikan StratifiedKFold biasa)
- n_folds       : 5
- Seed          : 42
- Group column  : `duplicate_group_id` — setiap grup duplikat (62 grup) mendapat ID unik. File unik mendapat ID unik per file. Dipastikan tidak ada 2 file dari grup yang sama terpisah antar-fold.
- Locked        : **YES** — sudah final dan tidak boleh diubah lagi

**Tambahan evaluasi CV (wajib):**
- Selain Macro F1 keseluruhan, **hitung F1 Electronic secara terpisah** untuk dua subpopulasi: (1) icon 150×150 px, (2) foto natural (>150×150). Ini untuk mendeteksi jika model hanya kuat di satu subpopulasi — indikasi overfitting/ shortcut.

**Catatan implementasi untuk Ababil:**
- **Exclusion list untuk CV split:** 97 file train-test overlap + `O_8873.jpg` — total 98 file di-exclude dari train set saat split CV (sehingga ukuran train untuk CV = 26.429).
- **Final training:** gunakan FULL 26.527 (kecuali `O_8873.jpg` tetap di-exclude) → total 26.526 untuk final model.
- **Hash-override final:** saat prediksi test, 97 file yang overlap di-set langsung ke label train-nya (`95` Organic, `2` Electronic). Sisanya 1.361 file pakai prediksi model.
- **WAJIB! Test Loader:** Saat membuat pipeline prediksi, **load `submission.csv`** terlebih dahulu untuk mengambil urutan `id` (1-1458). Iterasi prediksi harus mengikuti urutan ini. Setelah prediksi didapat, tulis ke kolom `predicted` pada baris yang sesuai. **Jangan pernah mengubah urutan baris CSV.**

---

## MODELING STRATEGY (LOCKED per Sonnet Revision, 2 Juli 2026)

- **Anchor Model (wajib)**: **ConvNeXt V2-Tiny** (28.6M params), pre-trained ImageNet-21K. Dipilih karena paling forgiving untuk hardware tak-terkonfirmasi & budget 3 submission.
- **Partner Ensemble (kondisional, hanya Submission 3)**: **EfficientNetV2-S** (21.5M params) — hanya jika CV ensemble terbukti unggul dari single model.
- **Model Ditolak**: ConvNeXt V2-Base / EfficientNetV2-M / Swin V2-Tiny sebagai ANCHOR (terlalu berat/rewel untuk fase awal). Hanya dipertimbangkan sebagai kandidat sekunder jika baseline ConvNeXt Tiny solid & hardware memadai.

### Resep Training (Locked)
| Komponen | Keputusan Final |
| :--- | :--- |
| **Loss Function** | **Class-Balanced Loss** (effective number of samples) — lebih stabil untuk deep net dibanding invers-frekuensi naif. Weighted Random Sampler sebagai pelengkap. |
| **Optimizer** | AdamW (weight decay decoupled) |
| **Scheduler** | Cosine Decay + Linear Warmup (5–10% steps) |
| **Learning Rate Strategy** | **Discriminative LR**: Lapisan *head* (classifier) diberi LR 10× lebih besar dari backbone. Backbone pakai LR sangat kecil (misal 1e-5), head pakai 1e-4. Ini menjaga fitur ImageNet tetap utuh. |
| **Regularisasi** | Label Smoothing (ε=0.1) dimulai dari fine-tuning kedua. EMA (Exponential Moving Average) pada SEMUA run setelah baseline. |
| **Resolusi Input** | **Baseline 224×224**. Naikkan ke ≥288×288 **hanya jika** hardware terkonfirmasi aman (khusus untuk menangkap detail foto natural Electronic). |

### Strategi Augmentasi (Kondisional — JANGAN PAKAI SEMUA SEKALIGUS)
- **Wajib**: RandAugment (ringan–sedang) + Horizontal Flip + Color Jitter terbatas (brightness/warna adalah sinyal diskriminatif, jangan dirusak).
- **CutMix**: **TIDAK AKTIF di baseline.** Hanya diaktifkan jika hasil **Grad-CAM** pasca-baseline mengonfirmasi model belajar shortcut background (Recyclable putih / Electronic square).
- **Mixup**: Diuji TERPISAH setelah baseline+CutMix. Drop jika F1 Electronic turun.

### Evaluasi Model Selection
- Kriteria utama: **Macro F1 rata-rata 5-fold CV.**
- **Tie-breaker**: F1-Score kelas Electronic (karena variabel penentu leaderboard).
- **WAJIB** evaluasi tambahan: F1 per-subpopulasi Electronic (icon 150×150 vs foto natural) — untuk memastikan model tidak overfit ke satu subpopulasi saja.

---

## EXPERIMENT ROADMAP (5 Fase — Sonnet Revision)

| Fase | Fokus | Gerbang Keputusan |
| :--- | :--- | :--- |
| **1. Fondasi** | DataLoader final + StratifiedGroupKFold + Baseline ConvNeXt Tiny polos (tanpa trik, tanpa CutMix, tanpa Class-Balanced Loss) | Baseline harus convergen sebelum lanjut ke Fase 2 |
| **2. Diagnostik** | Grad-CAM shortcut check pada baseline | Menentukan aktivasi CutMix untuk seluruh fase berikutnya |
| **3. Optimasi Electronic** | Class-Balanced Loss, CutMix-kondisional, Resize ≥288px (jika hardware aman) | Wajib konfirmasi hardware sebelum naik resolusi |
| **4. Stabilisasi** | EMA, Label Smoothing, Mixup uji-banding | Hanya lanjut ke Fase 5 jika CV stabil dan F1 Electronic tidak turun |
| **5. Finalisasi** | Ensemble kondisional + TTA + Alokasi 3 submission | Ensemble hanya jika CV membuktikan unggul dari single model |

---

## PENDING ACTIONS

### ✅ DECISIONS MADE & CLOSED (Ababil — Eksekusi Sekarang)
- [x] **Flag 3 (train-test dupe)**: Keputusan final = Exclude untuk CV, Full data + Hash-override untuk final submission. **CLOSED.**
- [x] **Flag 7 (mislabel)**: Keputusan final = Exclude `O_8873.jpg`, keep `R_799.jpg`. **CLOSED.**
- [x] **Flag 6 (within-class dupe)**: Keputusan final = Gunakan StratifiedGroupKFold, JANGAN exclude. **CLOSED.**
- [x] **Strategi Modeling**: ConvNeXt V2-Tiny sebagai anchor, Class-Balanced Loss, CutMix kondisional. **CLOSED per Sonnet Revision.**

### 🔴 HIGH — NEXT ACTIONS (Ababil)
- [ ] **Ababil** implementasikan DataLoader final dengan:
  - StratifiedGroupKFold (n=5, seed=42, group_col=duplicate_group_id)
  - Exclude list untuk CV: 97 overlap + `O_8873.jpg`
  - Converter `.convert("RGB")` untuk handle RGBA (2 file) & Palette (17 file)
  - Resize strategy: minimal 224×224, pakai lanczos/padding (hindari naive upscale untuk 150×150)
  - **Test Loader WAJIB** membaca `submission.csv` sebagai source of truth urutan ID (1-1458). Prediksi harus ditulis kembali ke CSV dengan urutan yang persis sama (Constraint 4).
- [ ] **Ababil** jalankan **Baseline Training ConvNeXt V2-Tiny** (resolusi 224, tanpa CutMix/Mixup, tanpa Class-Balanced Loss dulu) sebagai titik referensi bersih.
- [ ] **Ababil/Jeremy** jalankan **Grad-CAM check** segera setelah baseline convergen — hasilnya menentukan apakah CutMix masuk roadmap atau tidak.
- [ ] **Ababil** implementasikan **Class-Balanced Loss** + Weighted Random Sampler di eksperimen kedua (setelah baseline & Grad-CAM selesai).
- [ ] **Ababil** konfirmasi **spesifikasi hardware riil** (VRAM, kecepatan training per epoch) SEBELUM menaikkan resolusi ke ≥288px.
- [ ] **Ababil** kirim ringkasan EDA final + keputusan strategi ke Vierico untuk Checkpoint B2

### 🟡 NEW — PAPER DATASET OVERLAP INVESTIGATION (Jeremy / Vierico)
- [ ] **Jeremy** jalankan MD5-matching antara **24.705 file paper dataset** (yang ditemukan) vs **1.458 test file** kompetisi.
  - Tujuan: Cek apakah test set kompetisi memiliki overlap dengan dataset publik berlabel di luar.
  - Ini **BEDA** dengan Flag 3 — ini narik informasi dari data EKSTERNAL (area abu-abu).
- [ ] **Vierico** berikan fatwa setelah hasil MD5 diketahui:
  - Opsi A: Boleh digunakan sebagai "prior knowledge" (sama seperti pretrained model)?
  - Opsi B: Dilarang karena dianggap "cheating" / melanggar semangat kompetisi?
  - Opsi C: Wajib dilaporkan ke panitia karena peserta lain berpotensi menemukan hal yang sama?
- [ ] **Sampai ada fatwa Vierico, hasil MD5 ini WAJIB DI-HOLD dan TIDAK boleh digunakan untuk memengaruhi model/submission.**

### 🟢 LOW — FUTURE TASKS
- [ ] **Jeremy** (opsional): Audit manual sampel acak untuk estimasi noise labeling (H4) — prioritas diturunkan karena sudah ada bukti sistemik dari paper.

## CONTEXT RESET PROTOCOL
Saat ganti akun Claude (limit habis), lakukan urutan ini:

**Untuk Ababil:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_ababil.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari Fase 1 (Fondasi) — DataLoader, Baseline ConvNeXt Tiny polos, lalu Grad-CAM. Strategi modeling sudah di-lock di Project State. Jangan pakai CutMix sebelum Grad-CAM."

**Untuk Jeremy:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_jeremy.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari E8 (EDA Complete). Tugas baru: jalankan MD5 paper vs test, tapi HOLD hasilnya sampai fatwa Vierico."

**Untuk Vierico:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_vierico.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari checkpoint B2 (EDA Commentary) — semua keputusan strategi sudah dibuat dan di-lock di section MODELING STRATEGY. Mohon fatwa untuk paper-overlap."

*Jangan skip langkah ini. Tanpa project_state, Claude mulai dari nol.*

## CATATAN BEBAS
Gunakan bagian ini untuk hal-hal yang tidak masuk kategori di atas:

- **[2 Juli 2026] [System]:** **Strategi Sonnet 5 (Revisi) secara resmi diadopsi** sebagai pedoman modeling. Semua keputusan di `## MODELING STRATEGY` dan `## EXPERIMENT ROADMAP` adalah LOCKED dan tidak boleh diubah tanpa konsensus tim.
- **[2 Juli 2026] [Ababil/Vierico]:** SEMUA KEPUTUSAN STRATEGI FINAL. Ringkasan:
  - Flag 3 (train-test dupe): Split strategy — exclude utk CV, full + override utk final.
  - Flag 6 (within dupe): StratifiedGroupKFold, jangan exclude.
  - Flag 7 (mislabel): Exclude `O_8873.jpg`, keep `R_799.jpg`.
  - Paper overlap: MD5 check ke test dijalankan oleh Jeremy, hasilnya DI-HOLD sampai fatwa Vierico.
- **[2 Juli 2026] [System]:** Validation Strategy resmi di-lock menjadi StratifiedGroupKFold. Tidak boleh diubah lagi.
- **[2 Juli 2026] [System]:** Ababil sekarang tidak punya blocker — bisa langsung coding DataLoader. **TAMBAHAN:** Test loader WAJIB pakai `submission.csv` sebagai template urutan.
- **[2 Juli 2026] [System]:** Resource compute ditetapkan: Vast.ai (4080S/5070/5080 16GB) atau Kaggle gratis. Ababil wajib pakai AMP dan menyesuaikan batch size dengan VRAM 16GB. **PERINGATAN:** Jangan coba ConvNeXt V2-Base di 16GB untuk eksperimen awal.
- **[1 Juli 2026] [System]:** Aturan kompetisi sangat ketat terkait submission (maks 3x). Pastikan CV score sangat reliable (StratifiedGroupKFold) karena kita tidak bisa mengandalkan LB probing.
- **[1 Juli 2026] [System]:** Kelas Electronic hanya 3.961 gambar (~15%). Metrik Macro F1-Score akan sangat menghukum model yang lemah di kelas ini. Pertimbangkan Weighted CrossEntropyLoss atau Class-Balanced Sampling sejak awal.
- **[1 Juli 2026] [Jeremy — EDA Complete]:**
  - Exact-hash duplicate check (train vs test) selesai. 97 file overlap. File pairing lengkap tersedia di variabel `train_test_duplicates` (dict) di notebook eksplorasi.
  - Cross-class duplicate check selesai. 1 kasus mislabel terkonfirmasi: `R_799.jpg` = `O_8873.jpg` (tas kain — seharusnya Recyclable). Hash: `95bd2693fd68b87d40601c3002ebdf21`.
  - Within-class duplicate check selesai: 62 grup total (1 cross + 61 within). Electronic proporsi tertinggi (1,51%). 5 contoh pasangan spesifik sudah dicatat di Poin 9.
  - Semua statistik (brightness, warna, dimensi, aspect ratio, foreground ratio, background variance, orientasi) sudah dihitung dan tersedia di notebook eksplorasi.
  - File RGBA di Electronic: 2 file dengan alpha channel — perlu di-handle saat loading (konversi ke RGB).
  - File mode P (palette/indexed): total 17 file — perlu di-handle saat loading (konversi ke RGB).