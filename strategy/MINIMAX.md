# Strategic Modeling Analysis — BDC Satria Data 2026, Problem 1 (Waste Classification)

> **Author**: Mavis (root session) | **Date**: 2 Juli 2026
> **Context**: EDA complete (Jeremy), keputusan Flag 3/7 masih pending (Ababil), submission budget 3/3
> **Temuan baru**: Kompetisi dataset untuk Organic + Recyclable = sourced from paper's Mendeley dataset (Yasin & Koklu 2023). Electronic = grafted kelas tambahan.
> **Tujuan dokumen**: Strategic blueprint untuk seluruh tim menuju submission final 30 Juli 2026.

---

## TL;DR (Verdict)

| Pertanyaan | Jawaban Singkat |
|---|---|
| Replikasi arsitektur paper (InceptionV3 + SVM)? | **TIDAK.** Paper pakai frozen-feature + classic ML yang obsolete; kompetisi 3-class butuh end-to-end fine-tuning. Paper = **benchmark / calibration point**, bukan strategi. |
| Model rekomendasi utama? | **ConvNeXt-V2 Tiny** (atau ConvNeXt-Tiny) @ 224×224 sebagai **anchor**. **EfficientNetV2-S** sebagai **diversity partner** untuk ensemble. |
| Transfer learning strategy? | **2-stage fine-tuning**: (1) frozen backbone + train head 2 epoch, (2) full fine-tune dengan discriminative LR + cosine + EMA. |
| Handling Electronic? | **Weighted CE Loss + WeightedRandomSampler + augmentation lebih agresif**. Electronic = bottleneck yang paling probable. |
| Prioritas eksperimen? | Resolve Flag 3/7 → Baseline ConvNeXt-Tiny → class imbalance fix → EfficientNetV2-S → 5-fold ensemble → TTA + pseudo-label (opsional). |
| Submission plan? | Slot 1 (~10 Juli): single best model. Slot 2 (~20 Juli): 5-fold ensemble. Slot 3 (~28 Juli): final ensemble + TTA. |
| **Confidence** | **HIGH** untuk rekomendasi teknis. **MEDIUM-HIGH** untuk skor absolut (ada uncertainty di test distribution & CV-LB gap). |

---

## 1. Implikasi Dataset

### 1.1 Konsekuensi "Sama Persis dengan Paper"

**Sisi positif:**

- **Pre-built priors**: Paper sudah prove bahwa 2-class Organic/Recyclable dapat ~96% dengan InceptionV3 frozen features. Artinya visual signal di kelas ini **cukup kaya dan learnable** — kita tidak berhadapan dengan data yang fundamentally noisy.
- **Pre-trained model ImageNet transfer akan sangat efektif** — fitur low/mid-level (tekstur, shape, color distribution) sangat transferable dari ImageNet ke waste imagery.
- **Hyperparameter intuition yang murah**: input size ~224×224, augmentasi standar, optimizer AdamW — semua sudah divalidasi di paper untuk dataset source yang sama.
- **Sample efficiency tinggi**: Paper pakai ~24K gambar dan achieve 96% — kita 26.5K (lebih banyak) → modern model dengan 224×224 input akan push >98% pada 2-class subset.

**Sisi negatif / risiko:**

- **Distribution shift pada kelas Electronic**: Paper tidak cover Electronic. Train kita untuk Electronic sumbernya beda (mixed stock icons + kamera HP + nama file acak). CV macro F1 kita akan overestimate performa Electronic di test set karena:
  - Test set composition unknown — mungkin Electronic di-test proporsinya berbeda
  - Background / lighting Electronic lebih varied di test (yang paper tidak cover)
- **Benchmark overconfidence**: Kalau kita "kepedean" karena paper achieve 96%, kita mungkin underestimate gap arsitektur yang sebenarnya. Realita: paper pakai **frozen feature + linear-ish classifier** yang sangat terbatas. End-to-end fine-tuning modern biasanya **+3-5% F1 di atas frozen-feature baseline**.
- **Mislabel noise yang lebih luas**: Paper pakai dataset cleaned (25,077 → 24,705 = ~1.5% removed as noise). Kompetisi mungkin tidak se-aggressive itu, atau ada **additional mislabel** yang lolos (kami sudah confirm 1 cross-class mislabel `O_8873.jpg` — flag 7). Noise label = silent CV inflation.
- **Test set mungkin ada "trick" yang paper tidak anticipate**: Panitia bisa saja inject test image dari sumber yang paper tidak cover (mungkin extra-hard Electronic, atau OOD images). Macro F1 metric sangat kejam terhadap miss-class.

### 1.2 Apakah Paper Bisa Dijadikan Baseline Kuat?

**Ya, sebagai calibration point, bukan sebagai strategi.**

Cara pakai paper sebagai baseline:

1. **Replikasi InceptionV3 frozen + SVM** pada dataset kompetisi (Organic + Recyclable subset, exclude 97 leakage file + `O_8873.jpg`). Kalau hasilnya ~94-97% pada 2-class subset → dataset kompetisi confirmed "kurang lebih sama" dengan paper. Kalau hasilnya drop jauh (<90%) → ada distribution gap yang perlu diinvestigasi.
2. **Hitung lower-bound performance kita**: Kalau paper dapat 96.3% dengan pendekatan suboptimal, target minimal kita di 2-class subset harus ≥98%. Kalau tidak tercapai, ada masalah di pipeline kita.
3. **JANGAN pakai InceptionV3+SVM sebagai submission**. Hanya sebagai sanity check / benchmark.

### 1.3 Risiko Data Leakage / Distribution Bias

| Risiko | Severity | Status | Mitigasi |
|---|---|---|---|
| Train-test MD5 overlap (97 files) | CRITICAL | Flag 3 confirmed | Exclude 97 train files dari training set. Test files dengan label train-nya sudah diketahui → tetap biarkan di test (kita tetap harus predict; tapi tidak boleh "cheat" dengan submit known labels tanpa prediksi). |
| Cross-class mislabel | CRITICAL | Flag 7 confirmed (O_8873) | Exclude O_8873.jpg dari training (atau relabel jadi Recyclable). Keep R_799.jpg. |
| Wider mislabel noise (H4) | MEDIUM | Belum divalidasi | Label smoothing 0.1, Mixup/CutMix, robust loss. Iterasi berikutnya: random sample audit oleh Jeremy. |
| Within-class duplicates antar fold | MEDIUM | Flag 6 confirmed | Group-aware StratifiedKFold (duplikat masuk fold sama). |
| CV-LB gap | MEDIUM-HIGH | Unknown | Calibrate di submission pertama dengan membaca delta. Gunakan 5-fold ensemble untuk reduce variance. |

---

## 2. Strategi Modeling

### 2.1 Replikasi Model Paper sebagai Baseline? **TIDAK.**

Alasan konkret:
- Paper cuma handle **2-class**. Kita butuh **3-class**.
- Paper pakai **frozen features + classical ML** (SVM/KNN/DT). End-to-end fine-tuning modern has dominated since 2017-2018.
- InceptionV3 (2015) secara arsitektur sudah **kalah di semua benchmark modern** (ImageNet, COCO, dll.) dibanding ConvNeXt, EfficientNetV2, Swin, EVA, dll.
- SVM/KNN tidak bisa adapt ke **distribution shift Electronic**. Backbone pretrained ImageNet + end-to-end FT bisa adapt fitur specifically untuk membedakan Electronic dari Organic/Recyclable.

**Pengecualian**: Replikasi InceptionV3+SVM sebagai **sanity check 2-class baseline** saja, untuk verifikasi pipeline kita + calibration. Bukan submission strategy.

### 2.2 Model Modern: Apa yang Direkomendasikan?

**Tier 1 (Anchor) — ConvNeXt-V2 Tiny / ConvNeXt-Tiny**

- **Alasan**: Salah satu CNN modern terbaik per-FLOP. Pretrained ImageNet-22k → IN1k. Robust, well-tested, mudah di-fine-tune. ~28M params, 224×224 input.
- **Pro**: Cepat iterasi, strong baseline, banyak referensi komunitas. ConvNeXt-V2 punya FCMAE pretraining (improved feature).
- **Con**: Tidak punya inductive bias sekuat Swin Transformer.

**Tier 2 (Diversity Partner) — EfficientNetV2-S**

- **Alasan**: Arsitektur berbeda (depthwise separable + fused-MBConv) → memberikan diversity dalam ensemble. Sering outperform ConvNeXt di small-data regime. Pretrained ImageNet.
- **Pro**: Efficient training, strong baseline.
- **Con**: Bisa lebih sensitive ke augmentation choice.

**Tier 3 (Optional, jika GPU budget cukup) — Swin V2 Tiny / EVA-02 Tiny**

- **Alasan**: Transformer architecture → diversity terbesar dari CNN. EVA-02 Tiny pretrained dengan MIM (masked image modeling) → very strong features.
- **Pro**: Diversity dalam ensemble.
- **Con**: Butuh lebih banyak epoch untuk converge, butuh learning rate lebih kecil (~1e-4).

**Anti-rekomendasi**:
- **InceptionV3**: Sudah obsolete arsitektur.
- **VGG16**: Sangat obsolete.
- **ResNet-50**: Bisa dipakai sebagai quick sanity, tapi ConvNeXt-Tiny sudah lebih baik per-FLOP.
- **Vision Transformer "vanilla" (DeiT-Tiny, ViT-Tiny)**: Butuh data lebih banyak untuk fine-tune dari scratch. ConvNeXt lebih robust untuk ~26K data.

### 2.3 Strategi Terbaik untuk Menang Kompetisi Ini?

**Hybrid strategy — anchor CNN + diversity CNN/Transformer ensemble.**

Alasan:
- Single arsitektur biasanya cukup baik, tapi ensemble 2-3 arsitektur diverse meningkatkan macro F1 ~0.3-0.7% poin.
- Macro F1 metric sangat sensitif ke error pada minority class (Electronic). Ensemble reduces variance terutama di kelas sulit.
- Submission budget 3/3 → ensemble adalah "free lunch" — kita bisa lock ensemble sebagai submission akhir tanpa extra slot.

---

## 3. Transfer Learning Strategy

### 3.1 Apakah Fine-tune dari Pretrained ImageNet? **YA, strongly recommended.**

Alasan:
- ImageNet pretrained features sudah capture low/mid-level visual features (edges, textures, shapes) yang sangat transferable ke waste imagery.
- Paper proves ini bekerja (frozen feature achieve 96% pada 2-class) — end-to-end FT akan lebih baik lagi.
- Training dari scratch pada 26K gambar akan underfit relatif terhadap fine-tuned model.

### 3.2 Apakah Perlu Staged Fine-tuning? **YA, 2-stage.**

**Stage 1: Frozen backbone + train head (warm-up)**

```
- Backbone: freeze semua layer
- Head: Linear(backbone_dim, 3) atau Linear(backbone_dim, 256) → ReLU → Dropout(0.1) → Linear(256, 3)
- Loss: CrossEntropyLoss dengan class weights
- Optimizer: Adam, LR=1e-3, weight_decay=1e-4
- Epochs: 2-3 (cukup untuk warm head)
- Purpose: Stabilize classifier weight sebelum backbone mulai berubah
```

**Stage 2: Full fine-tuning dengan discriminative LR**

```
- Backbone: unfreeze semua
- Backbone LR: 2e-4 (atau 1e-4 untuk Swin/EVA)
- Head LR: 2e-3 (5-10x lebih tinggi dari backbone)
- Loss: CrossEntropyLoss(label_smoothing=0.1) + class weights
- Optimizer: AdamW, weight_decay=0.05
- Schedule: LinearWarmup (1-2 epochs) → CosineAnnealing
- Epochs: 12-18 total
- Mixed precision: AMP fp16
- EMA: True (decay=0.999)
- Purpose: Adapt backbone ke domain spesifik sambil preserve pretrained features
```

**Alternatif (advanced, opsional)**: 3-stage dengan intermediate learning rate (1e-4 backbone epoch 1-5, lalu 2e-5 epoch 6-15). Tapi 2-stage sudah cukup untuk kebanyakan kasus.

### 3.3 Bagaimana Adaptasi ke Kelas Electronic yang Tambahan?

**Strategi utama**: Weighted loss + oversampling (lihat section 4). 

**Strategi tambahan (advanced, opsional)**:
- **Selective fine-tuning**: Train head → fine-tune hanya top 30% backbone layers (skip deep layers yang mungkin overfit ke ImageNet features) → lalu full FT. Efisiensi: 30-50% faster training. Risiko: bisa miss optimal performance.
- **Domain adaptation pretraining**: Fine-tune ImageNet pretrained model pada dataset external (seperti TrashNet, yang **TIDAK BOLEH** karena aturan kompetisi — Constraint 2 Vierico). Skip ini.
- **Self-supervised pretraining pada train kita**: Misal SimCLR atau MAE pretraining pada train set, lalu fine-tune untuk klasifikasi. Meningkatkan representation learning pada domain waste. **TAPI**: ini butuh compute tambahan dan belum tentu menang. Reserve untuk iterasi akhir jika waktu memungkinkan.

**Rekomendasi praktis**: Cukup dengan 2-stage fine-tuning + class weighting + augmentation Electronic-specific. Jangan over-engineer.

---

## 4. Handling Kelas Electronic

**Verdict: YA, Electronic akan jadi bottleneck. Perlu perlakuan khusus.**

### 4.1 Apakah Electronic akan Menjadi Bottleneck?

Probabilitas tinggi, alasan:
1. **Kelas minoritas (14.9%)** — Macro F1 = average F1 across classes. Kalau Electronic F1 = 80% dan kelas lain = 99%, macro F1 = 92.7%. Drop 5% di Electronic = drop ~1.7% macro F1.
2. **Visual heterogeneity** — bimodal (150×150 stock + foto natural). Model bisa gagal handle sub-populasi tertentu.
3. **Background mixture** — paling campur dibanding kelas lain (white studio + real-world). Background shortcut bisa gagal di test.
4. **Possible semantic confusion** dengan Recyclable: kabel, charger, circuit board → apakah recyclable (bisa didaur ulang) atau electronic? Ini edge case. Label mungkin noisy di area ini.
5. **Test set unknown composition** — kalau test punya proporsi Electronic yang sangat berbeda (misal 25%), model yang trained di distribusi 15% akan underperform.

### 4.2 Apakah Perlu Oversampling / Class Weighting / Augmentasi Khusus / Curriculum Learning?

**Rekomendasi: Kombinasi weighted loss + oversampling + augmentation khusus.**

#### a) Weighted Loss (CE Loss dengan class weights)

```
weights = [1.0, 2.0, 1.2]  # untuk [Recyclable, Electronic, Organic]
# atau computed:
weights = (1 / class_counts) ** 0.5
weights = weights / weights.sum() * len(weights)
# hasilnya kira-kira [1.0, 1.8, 1.2]
```

Loss: `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)`

**Trade-off**: Weight terlalu tinggi → predict Electronic terlalu agresif, false positive naik, F1 Recyclable/Organic turun. Mulai dengan moderate weight (~1.5-2.0 untuk Electronic), tune via CV.

#### b) WeightedRandomSampler

```python
from torch.utils.data import WeightedRandomSampler
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
```

Effect: setiap batch punya distribusi kelas yang lebih balanced.

**Alternatif**: `RandomOverSampler` dari imbalanced-learn, applied di training set. Cap di 2x untuk avoid overfit pada minority.

**Trade-off**: Oversampling + label smoothing + Mixup biasanya lebih robust dari oversampling saja.

#### c) Augmentasi Khusus untuk Electronic

Electronic **bimodal** (150×150 stock icons vs foto natural). Augmentasi yang membantu:

- **Resize strategy**:
  - 150×150 stock icons: **pad-to-square** (preserve original resolution, avoid upscale noise)
  - Natural photos (>=300px): **resize short side** + **random crop** 224×224
  - Mixed strategy: pad all to square, then resize to 224 (jauh lebih simple dan biasanya works well)
- **Geometric** (lebih agresif dari kelas lain):
  - Rotation ±45° (Electronic components bisa diorientasi apapun)
  - Horizontal & vertical flip (lebih aman di Electronic karena banyak komponen simetris)
  - Random scale 0.7-1.3 (handle scale variance dari bimodal)
  - RandomAffine (translate, shear)
- **Photometric** (moderate):
  - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
  - HueSaturationValue(hue_shift=10, sat_shift=15, val_shift=10)
  - RandomBrightnessContrast
- **Occlusion**:
  - CoarseDropout / Cutout (max_holes=4, hole_height=16-32px) — Electronic banyak komponen kecil, occlusion harus jadi bagian dari training
  - Random erasing (p=0.25)

#### d) Curriculum Learning

**Verdict: Tidak perlu di iterasi pertama.** Risiko over-engineering tanpa收益 yang signifikan.

Kalau di iterasi akhir ada waktu luang, bisa eksplorasi:
- **Easy-to-hard curriculum**: Train di "easy" Electronic dulu (well-lit, single object) → lanjut "hard" (multi-object, low-light, real-world context). 
- **Self-paced learning**: Loss-based filtering, training pada sample dengan loss rendah dulu.

**Skip unless time permits.**

### 4.3 Class-Balanced Loss (effective number of samples)

Alternative weighting scheme dari Cui et al. 2019:

```
effective_num = 1 - beta^(class_count)
weights = (1 - beta) / effective_num
# beta = 0.999 biasanya
```

Lebih robust dari inverse frequency. Bisa coba sebagai variant.

---

## 5. Training Strategy

### 5.1 Learning Rate Schedule

```
Stage 1 (frozen backbone):
- Optimizer: Adam
- LR: 1e-3 (head only)
- Epochs: 2-3
- No warmup needed (head mulai dari random init)

Stage 2 (full FT):
- Optimizer: AdamW, weight_decay=0.05
- Backbone LR: 2e-4 (CNN) atau 1e-4 (Transformer)
- Head LR: 2e-3 (5-10x backbone)
- Warmup: Linear, 1-2 epochs, dari LR/10
- Schedule: CosineAnnealingLR (atau OneCycleLR untuk alternative)
- Epochs total: 12-18
- Min LR: ~1e-6
```

**Catatan**: Discriminative LR (backbone < head) adalah critical. LR backbone terlalu tinggi → kehilangan pretrained features → underfit. LR head terlalu rendah → classifier tidak converge.

### 5.2 Augmentation (Albumentations pipeline)

```python
import albumentations as A

train_transform = A.Compose([
    # Step 1: handle aspect ratio & bimodal sizes
    A.LongestMaxSize(max_size=224, interpolation=cv2.INTER_LANCZOS4),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, fill=255),
    # Step 2: geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Affine(scale=(0.85, 1.15), translate_percent=0.05, rotate=(-30, 30), shear=(-10, 10), p=0.5),
    # Step 3: photometric
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ], p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.1),
    # Step 4: occlusion
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.4),
    # Step 5: normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = A.Compose([
    A.LongestMaxSize(max_size=224, interpolation=cv2.INTER_LANCZOS4),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, fill=255),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Note**: Padding pakai `fill=255` (putih) bukan 0 (hitam) — paper dataset cenderung white-background, jadi padding putih lebih natural. Bisa coba hitam juga sebagai variant.

### 5.3 Mixup / CutMix

**Verdict: YA, keduanya bantu terutama untuk Electronic.**

```python
from timm.data import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.2,        # Mixup intensity
    cutmix_alpha=1.0,        # CutMix intensity
    prob=0.5,                # Apply 50% of batches
    switch_prob=0.5,         # Mixup vs CutMix 50/50
    mode='batch',
    label_smoothing=0.1,
    num_classes=3,
)
```

**Apply on**: Setelah data loading, sebelum model forward pass. Loss dihitung per-sample (Mixup produces soft labels).

**Why helps**:
- **Reduces overfit to minority class** (Electronic)
- **Improves calibration**
- **Acts as implicit regularization**

### 5.4 Label Smoothing

**Verdict: YA, ε=0.1 adalah sweet spot.**

```
soft_label = (1 - 0.1) * one_hot + 0.1 / num_classes
```

**Why helps**:
- Mengurangi confidence model pada label noisy
- Mengurangi overfit
- Bekerja baik dengan Mixup/CutMix

### 5.5 EMA (Exponential Moving Average)

**Verdict: YA, decay=0.999.**

```python
import timm.utils
ema = timm.utils.ModelEmaV2(model, decay=0.999)
# Update setelah setiap optimizer step:
ema.update(model)
# Eval pakai ema.module
```

**Why helps**:
- Smooth model weights → lebih stable predictions
- Typical +0.1-0.3% F1 improvement
- "Free" regularization

### 5.6 Optimizer

**AdamW** untuk stage 2, **Adam** untuk stage 1.

- AdamW dengan weight_decay=0.05 (standard untuk fine-tuning ViT/ConvNeXt).
- Betas default (0.9, 0.999).
- EPS default (1e-8).

**Alternatif**: Lion optimizer (lebih efisien memory, sering outperform AdamW). Bisa eksplorasi tapi AdamW adalah safe default.

### 5.7 Early Stopping

**Strategy**: Patience = 5 epochs pada val macro F1.

```python
best_f1 = 0
patience = 5
counter = 0

for epoch in range(max_epochs):
    train_one_epoch()
    val_f1 = evaluate()
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_checkpoint()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break
```

**Important**: Save best checkpoint based on **macro F1** (sesuai competition metric), bukan accuracy.

### 5.8 Cross-Validation

**Strategy**: StratifiedKFold(5, seed=42), group-aware (untuk handle Flag 6 within-class duplicates).

```python
from sklearn.model_selection import StratifiedGroupKFold

# group = MD5 hash dari image (duplikat = group sama)
# y = class label

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=hashes)):
    ...
```

**Pre-processing wajib sebelum CV**:
1. Exclude 97 train files yang overlap dengan test (Flag 3)
2. Exclude `O_8873.jpg` atau relabel jadi Recyclable (Flag 7)
3. Compute MD5 hash untuk semua file → group column
4. Within-class duplicate: keep best-quality sample atau semua? **Default: keep all**, biar StratifiedGroupKFold otomatis put duplikat di fold sama.

### 5.9 Ensemble

**Strategy**: Weighted average of logits dari multiple architectures + folds.

```python
# Inference ensemble:
logits_per_model = []
for fold_models_per_arch in [convnext_folds, efficientnet_folds]:
    for fold_model in fold_models_per_arch:
        logits = fold_model(test_batch)  # bisa dengan TTA
        logits_per_model.append(logits)

# Weighted average (weight by val macro F1)
final_logits = sum(w * logits for w, logits in zip(model_weights, logits_per_model))
predictions = final_logits.argmax(dim=-1)
```

**TTA (Test-Time Augmentation)**:
- Original + Horizontal Flip + Vertical Flip (untuk waste, VFlip aman)
- Average logits across TTA versions
- Typical +0.2-0.4% F1

**Ensemble candidates**:
1. **5-fold ConvNeXt-V2 Tiny** (5 models)
2. **5-fold EfficientNetV2-S** (5 models)
3. **(Optional) 5-fold Swin V2 Tiny** (5 models)
4. Final ensemble = weighted average of (1) + (2), weight by val macro F1

**Trade-off**: 15 models inference ~3x lebih lama dari single model. Manageable untuk 1458 test images.

---

## 6. Model Selection: Paper vs Modern

### 6.1 Trade-off Table

| Aspek | InceptionV3 + SVM (paper) | ConvNeXt-V2 Tiny + FT | EfficientNetV2-S + FT | Ensemble (ConvNeXt + EffNet) |
|---|---|---|---|---|
| Estimated 3-class macro F1 | ~85-88% (binary limitation + frozen feature) | ~97-99% | ~97-99% | ~98-99% |
| Adapts to Electronic | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Training time per fold | ~10 min (SVM only) | ~15-25 min (full FT) | ~15-25 min | ~3x (15 model training) |
| Iteration speed | Fast (training) | Fast | Fast | Slow (full ensemble) |
| Inference speed | Slow (SVM @ 24K features) | Fast | Fast | Slower (3x) |
| Ensemble potential | ❌ Poor | ✅ Good | ✅ Good | ✅ Best |
| Robustness to Electronic noise | ❌ Poor | ✅ Good (FT adapts) | ✅ Good | ✅✅ Best (diversity) |
| LB ceiling estimate | ~88-92% | ~98% | ~98% | ~98.5-99% |

### 6.2 Verdict

**Modern (ConvNeXt-Tiny + EfficientNetV2-S ensemble) menang di setiap dimensi:**

1. **Accuracy**: Modern fine-tuning > frozen features (empiris 2017-sekarang)
2. **Generalization**: End-to-end FT adapts ke distribusi kelas, frozen tidak
3. **Kompleksitas**: Lebih kompleks training, tapi manageable di 1 GPU (T4/V100/A100)
4. **Waktu training**: Comparable untuk single fold. Ensemble lebih lama tapi masih dalam budget

**Pengecualian untuk paper approach**: Hanya sebagai sanity check + calibration point di awal. Bukan submission strategy.

---

## 7. Competition Strategy

### 7.1 Target: Skor Setinggi Mungkin pada Leaderboard

**Constraint utama**: 3 submission slots, no LB probing.

**Prinsip utama**: 
- **CV = ground truth kita**, bukan LB. LB adalah validasi terakhir saja.
- **Setiap submission harus berdasarkan bukti CV yang solid**, bukan eksperimen liar.
- **Slot 1 = baseline anchor**, **Slot 2 = ensemble push**, **Slot 3 = final attack**.

### 7.2 Reproduksi Paper Dulu atau Langsung Model Kuat?

**Langsung model kuat.** Reasoning:

- Paper approach (InceptionV3 + SVM, 2-class) fundamentally tidak capable untuk 3-class kompetisi ini. Replikasi literally = waste of time + slot.
- "Reproduksi paper" sebagai sanity check = OK, tapi **di fase eksplorasi offline**, bukan sebagai submission.
- Timeline 28 hari dari sekarang → terlalu mepet untuk spent time di replikasi paper.

**Yang perlu kita replicate dari paper**:
- ✅ Konfirmasi dataset kompetisi sourced from paper's Mendeley dataset (cek 2-class subset perform ~94-97% dengan InceptionV3+SVM)
- ✅ Hyperparameter intuition (input size, augmentasi basic)
- ❌ Bukan arsitektur/approach

### 7.3 Roadmap Eksperimen (28 hari, 3 submission)

| Week | Tanggal | Aktivitas | Output |
|---|---|---|---|
| **W1** | 2-8 Juli | (1) Resolve Flag 3 & Flag 7 (Ababil) <br> (2) Setup DataLoader final + StratifiedGroupKFold <br> (3) Baseline InceptionV3+SVM (sanity, 2-class subset) <br> (4) Baseline ConvNeXt-Tiny end-to-end (full 3-class) <br> (5) Initial CV macro F1 number | First CV number untuk ConvNeXt-Tiny |
| **W2** | 9-15 Juli | (1) Class imbalance fix (weighted CE + sampler) <br> (2) Augmentation tuning (especially Electronic bimodal) <br> (3) Try Mixup/CutMix + label smoothing <br> (4) Try EfficientNetV2-S baseline <br> (5) **Submit Slot 1**: Best single architecture (ConvNeXt or EffNet) | Slot 1 submission. CV-LB delta calibration. |
| **W3** | 16-22 Juli | (1) 5-fold ensemble ConvNeXt <br> (2) 5-fold ensemble EfficientNetV2-S <br> (3) Heterogeneous ensemble (both archs) <br> (4) TTA implementation <br> (5) **Submit Slot 2**: Best ensemble (single arch + TTA atau 2-arch ensemble) | Slot 2 submission. CV-LB delta ensemble. |
| **W4** | 23-30 Juli | (1) Optional: try Swin V2 / EVA-02 (if time permits) <br> (2) Pseudo-labeling pada test set high-confidence (low-risk strategy) <br> (3) Final ensemble tuning <br> (4) **Submit Slot 3**: Final ensemble + TTA + (pseudo-label if implemented) | Slot 3 submission. Final attack. |

### 7.4 Risk Mitigation per Phase

| Risk | Mitigation |
|---|---|
| Slot 1 underperforms → wasted slot | Make sure Slot 1 based on at least 3 CV folds + diverse augmentations. Avoid wild hyperparameter experiments. |
| CV-LB gap besar | 5-fold ensemble reduces variance. TTA + diverse architectures reduce overfit. |
| Electronic F1 stuck low | Try oversampling caps, focal loss, class-balanced loss. If still stuck, audit random sample of misclassified Electronic. |
| Submission file format error | Validate submission.csv format early (use sample_submission.csv template). |
| Mislabel noise silently inflating CV | Label smoothing + Mixup + manual audit of random samples (Jeremy). |

---

## 8. Final Verdict

### 8.1 Model Paling Direkomendasikan

**Primary (anchor)**: **ConvNeXt-V2 Tiny** pretrained ImageNet-22k → IN1k, fine-tuned end-to-end dengan 2-stage training.

**Secondary (diversity partner)**: **EfficientNetV2-S** pretrained ImageNet, fine-tuned sama.

**Backup / optional (jika GPU budget memungkinkan)**: Swin V2 Tiny atau EVA-02 Tiny.

**Hyperparameter anchor**:
```
- Input: 224x224
- Optimizer: AdamW (stage 2), Adam (stage 1)
- LR: backbone 2e-4, head 2e-3
- Schedule: 1-2 epoch warmup + cosine annealing
- Epochs: 12-18
- Batch size: 32-64 (tergantung GPU)
- Loss: CrossEntropyLoss(weight=[1.0, 2.0, 1.2], label_smoothing=0.1)
- Mixup α=0.2, CutMix α=1.0, prob=0.5
- EMA decay=0.999
- CV: StratifiedGroupKFold(5, seed=42)
- Augmentation: Albumentations (lihat section 5.2)
```

### 8.2 Strategi Training Paling Masuk Akal

1. **2-stage fine-tuning** (frozen → full FT dengan discriminative LR)
2. **Class weighting + WeightedRandomSampler** untuk handle Electronic imbalance
3. **Mixup/CutMix + label smoothing** untuk robustness ke noise & overfit
4. **Cosine LR + Linear warmup** + EMA
5. **5-fold ensemble** + TTA (hflip + vflip + original)

### 8.3 Prioritas Eksperimen (Execution Order)

1. **[ABABIL]** Resolve Flag 3 (exclude 97 train overlap files) + Flag 7 (exclude/relabel O_8873.jpg)
2. **[ABABIL]** Setup final DataLoader + StratifiedGroupKFold(5) + augmentation pipeline
3. **[ABABIL]** Baseline ConvNeXt-Tiny end-to-end, get CV macro F1 → calibration point
4. **[ABABIL]** Apply class weighting + WeightedRandomSampler → re-evaluate CV
5. **[ABABIL]** Add Mixup/CutMix + label smoothing → CV improvement check
6. **[ABABIL]** Try EfficientNetV2-S baseline → compare CV dengan ConvNeXt
7. **[ABABIL + VIERICO]** Pick best architecture for Slot 1. Submit.
8. **[ABABIL]** 5-fold ensemble of best architecture
9. **[ABABIL]** Add 2nd architecture for diverse ensemble
10. **[ABABIL]** Add TTA (hflip + vflip)
11. **[ABABIL + VIERICO]** Submit Slot 2 (best ensemble).
12. **[JEREMY]** (optional) Audit random misclassified Electronic samples untuk noise estimation
13. **[ABABIL]** Final ensemble + (optional) pseudo-labeling on high-confidence test
14. **[ABABIL + VIERICO]** Submit Slot 3 (final).

### 8.4 Risiko Terbesar

| Rank | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **CV-LB gap besar** (CV overestimate karena leakage/mislabel noise) | Medium | High | Use reliable CV (group-aware, exclude leakage, macro F1 metric). 5-fold ensemble reduces variance. Calibrate di Slot 1. |
| 2 | **Electronic F1 stuck <85%** (model underperforms pada minority) | Medium | High | Class weighting, oversampling, aggressive augmentation, focal loss as backup. Manual mislabel audit. |
| 3 | **Electronic misclassified as Recyclable** (semantic confusion pada kabel/charger/dll) | Medium | High | Grad-CAM check, hard-example mining, class weighting, augmentasi Electronic-specific. |
| 4 | **Mislabel noise (H4) silently inflating CV** | Medium | Medium | Label smoothing 0.1, Mixup, manual audit. Exclude suspect cases. |
| 5 | **Submission format error** | Low | High (slot wasted) | Validate format early. Use sample_submission.csv as template. |
| 6 | **GPU/OOM issues** | Low | Medium | Start with smaller batch size. ConvNeXt-Tiny @ 224 fits in T4 (16GB). |
| 7 | **Wasted submission slot** (over-confident single model tanpa ensemble) | Medium | High | Follow roadmap. Slot 1 = single anchor, Slot 2 = ensemble, Slot 3 = final ensemble. No LB probing. |

### 8.5 Confidence Level

**Rekomendasi teknis (modern > paper, ConvNeXt/EffNet anchor)**: **HIGH confidence**

Alasan:
- Paper pakai frozen feature + classic ML = baseline yang obsolete secara empiris sejak 2017.
- End-to-end fine-tuning modern pretrained model dominant di semua vision benchmarks sejak 2017-2018.
- ConvNeXt & EfficientNetV2 well-validated di komunitas.
- 2-stage fine-tuning adalah best practice standard.
- Weighted CE + Mixup/CutMix + label smoothing = textbook techniques dengan bukti empiris luas.

**Confidence pada skor absolut (~98% macro F1)**: **MEDIUM-HIGH**

Alasan:
- **HIGH** bahwa kita akan achieve ≥97% macro F1 (paper's 96% adalah lower bound pada 2-class).
- **MEDIUM** bahwa kita akan achieve ≥98.5% macro F1 (ada uncertainty di test distribution & Electronic difficulty).
- **MEDIUM** bahwa kita akan achieve ≥99% macro F1 (test set composition unknown, possible tricks).

**Confidence pada keseluruhan kompetisi menang**: **MEDIUM**

Alasan:
- Secara teknis, model modern + ensemble + TTA sangat kuat.
- Risiko utama: (a) Kompetitor lain juga pakai strategi serupa, (b) Test set punya "trick" yang tidak kita anticipate, (c) Submission budget sangat ketat (3 slot) — tidak bisa trial-error.
- Edge: kualitas DataLoader (Flag 3/7 resolution) + quality of Electronic-specific handling akan jadi differentiator.

---

## Appendix A: Quick-Start Implementation Checklist untuk Ababil

```
□ Resolve Flag 3: list of 97 train files to exclude (dari Jeremy)
□ Resolve Flag 7: drop O_8873.jpg atau relabel
□ Compute MD5 hash untuk semua train files (untuk group-aware CV)
□ Setup final DataLoader:
  - Albumentations train_transform (lihat section 5.2)
  - val_transform (no augmentation, only resize+normalize)
  - WeightedRandomSampler atau class weights
□ Setup StratifiedGroupKFold(5, shuffle=True, random_state=42)
□ Define model: timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=3)
□ Define loss: nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.2]), label_smoothing=0.1)
□ Define optimizer (stage 1 vs stage 2)
□ Define scheduler: LinearWarmup + CosineAnnealing
□ Define EMA: timm.utils.ModelEmaV2(model, decay=0.999)
□ Define Mixup: timm.data.Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, ...)
□ Train 5 folds, save best checkpoint per fold (based on val macro F1)
□ Inference: logits average across 5 folds + TTA (hflip + vflip)
□ Submit Slot 1: best single architecture (5-fold average)
```

---

## Appendix B: Links & References

- **Paper**: Yasin, E. T., & Koklu, M. (2023). "Classification of Organic and Recyclable Waste based on Feature Extraction and Machine Learning Algorithms." ICISNA'23. Liverpool, UK. (PDF dilampirkan di attachment)
- **Dataset source**: Nnamoko et al. (2022). Waste Classification Dataset, Mendeley Data. https://data.mendeley.com/datasets/n3gtgm9jxj/2
- **timm library** (untuk model & Mixup/EMA utilities): https://github.com/huggingface/pytorch-image-models
- **Albumentations**: https://albumentations.ai/
- **ConvNeXt-V2 paper**: Woo et al. 2023, "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
- **EfficientNetV2 paper**: Tan & Le 2021, "EfficientNetV2: Smaller Models and Faster Training"

---

*End of strategic analysis. Update file ini setelah setiap milestone / submission.*