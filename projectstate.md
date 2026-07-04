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
- Last updated          : 3 Juli 2026
- Updated by            : Ababil (DataLoader implementation session)

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
- Active phase          : FASE 1 — FONDASI (DataLoader + Baseline) — dalam progress
- Last completed stage  : StratifiedGroupKFold split SELESAI & tervalidasi (5 fold, group integrity PASSED, class balance deviation maksimum 0.02pp — sangat baik). Test loader (submission.csv sebagai source of truth) SELESAI & tervalidasi (1458/1458 file match). RGB/Palette/RGBA/Grayscale converter (`load_image_as_rgb`) SELESAI & tervalidasi untuk semua mode ditemukan.
- Next action           : Ababil lanjut ke PyTorch Dataset/DataLoader class (pakai `train_master_with_folds.csv` + `load_image_as_rgb` converter + resize strategy), lalu baseline training ConvNeXt V2-Tiny (224px, tanpa augmentasi kompleks dulu).
- Blocker (if any)      : ✅ **RESOLVED** — Environment error (numpy/scipy/sklearn version mismatch) sudah diperbaiki oleh Ababil. StratifiedGroupKFold berjalan normal.

## DATASET
- Train file    : `train/` folder — 26.527 images (rows), 3 classes (subfolders)
- Test file     : `test/` folder — 1.458 images (rows). **Template urutan ID ada di `submission.csv` (ID 1 s.d. 1458) — WAJIB dijaga persis.** (Tervalidasi: 1458/1458 file match, urutan aman.)
- Target column : `predicted` (di submission.csv) -> Mapping: 0=Recyclable, 1=Electronic, 2=Organic
- Task type     : Multiclass Image Classification
- Time-based    : NO
- Leakage status: **CONFIRMED — 97/1458 test files (6,65%) exact-duplicate (MD5) dengan train files. Detail di Risk Flag 3.**
- **🆕 UPDATE Image Mode Audit (3 Juli 2026, full scan — koreksi dari estimasi EDA sebelumnya):**
  - Train non-RGB: **312 file total** (bukan ~19 seperti estimasi lama — EDA sebelumnya kemungkinan hanya sampling): 293 Palette (`P`), 17 RGBA, 2 Grayscale (`L`)
  - Test non-RGB: **5 file**, semua Palette (`P`)
  - Semua mode berhasil dikonversi ke RGB dengan aman via fungsi `load_image_as_rgb()` (RGBA di-composite ke background putih untuk hindari halo hitam di pixel transparan; P/L pakai `.convert("RGB")` standard)
  - **Outlier baru ditemukan**: `614.jpg` (train, grayscale) resolusi **3856×3856** — jauh di atas mayoritas file lain. Perlu diperhatikan saat resize/preprocessing strategy (kandidat untuk dicek lebih lanjut, worth di-flag ke Jeremy kalau ada waktu).

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
- [x] **Ababil** — Group ID assignment untuk StratifiedGroupKFold: SELESAI. `train_master_with_groups.csv` tersimpan (26.527 baris, 26.463 unique groups setelah exact-dup + 2 near-dup Electronic pairs `627(1)/627.jpeg` & `629(1)/629.jpeg` digabung manual). exclude_from_cv=97 (match), exclude_from_training=1 (`O_8873.jpg`, match). Semua assertion PASSED termasuk cross-class group check.
- [x] **Ababil** — StratifiedGroupKFold split: SELESAI. `train_master_with_folds.csv` tersimpan. 5 fold (5285-5286 tiap fold), group integrity PASSED (tidak ada grup terbelah), class balance deviation maksimum 0.02pp dari global ratio (Recyclable 37.83%, Electronic 14.98%, Organic 47.19%) — jauh di bawah threshold 2pp, tidak ada tindakan lanjutan diperlukan.
- [x] **Ababil** — Test loader: SELESAI. `submission.csv` dipakai sebagai source of truth urutan ID, semua 1458 file test tervalidasi ada di disk & urutan match.
- [x] **Ababil** — Image mode converter (`load_image_as_rgb`): SELESAI & tervalidasi untuk P/RGBA/L (lihat update di DATASET section untuk angka detail).
- [ ] **Ababil** — bangun PyTorch Dataset/DataLoader class yang menggabungkan: `train_master_with_folds.csv` (untuk train/val split per fold) + `load_image_as_rgb()` converter + resize strategy (minimal 224×224, perhatikan outlier `614.jpg` 3856×3856 — hindari naive upscale/downscale ekstrem tanpa pertimbangan)
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

- **[3 Juli 2026] [Ababil]:** Environment fixed (numpy/scipy/sklearn version mismatch resolved). StratifiedGroupKFold split berjalan sempurna — class balance deviation maksimum hanya 0.02pp, jauh lebih baik dari perkiraan awal (khawatir grup Electronic yang besar bikin timpang, ternyata tidak terjadi).
- **[3 Juli 2026] [Ababil]:** Test loader & image mode converter selesai. **Koreksi penting**: full scan menunjukkan non-RGB train file jauh lebih banyak dari estimasi EDA lama (312 vs ~19) — didominasi Palette mode (293 file), bukan cuma RGBA. Test set juga punya 5 file Palette. Semua berhasil dikonversi aman via composite-to-white (RGBA) / standard convert (P, L). Ditemukan outlier baru: `614.jpg` grayscale resolusi 3856×3856 — jauh di atas mayoritas, perlu perhatian khusus di resize strategy.
- **[3 Juli 2026] [Ababil]:** DataLoader Fase 1 — bagian group-assignment SELESAI dan tervalidasi (`train_master_with_groups.csv`). Ditemukan 2 pasang near-duplicate baru di Electronic (`627(1).jpeg`/`627.jpeg`, `629(1).jpeg`/`629.jpeg`) dari `near_duplicate_candidates.csv` — bukan exact MD5, tapi disepakati treatment: gabung jadi 1 duplicate_group_id sama seperti exact-dup lain (aman & murah, tidak perlu delegasi Jeremy). Sudah masuk ke group logic.
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

---

## SOURCE CODE REFERENCE (Notebook: `satria_data_bda.ipynb` — nomor cell asli notebook)
*Ini index/kamus, BUKAN copy-paste kode lengkap. Untuk implementasi detail, cek notebook langsung. Update section ini hanya kalau ada perubahan struktur besar (cell baru/dihapus/nama variabel berubah) — jangan sinkronkan tiap kali edit kecil, supaya tidak jadi beban maintenance ganda.*

### Peta Cell → Fungsi (ringkas)
| Cell | Fungsi | Kategori |
|---|---|---|
| 0 | Setup `data_dir`, `train_dir`, `test_dir`, `submission_path` — auto-detect folder root project | Setup |
| 1 | `print_tree()`, `summarize_structure()` — utilitas print struktur folder | Utility |
| 2 | `find_corrupt_images()` — scan corrupt file train & test (hasil: 0 corrupt) | EDA |
| 3 | `compute_file_hash()` (MD5, chunked) — dasar deteksi train-test overlap (hasil: 97 file, lihat Flag 3) | EDA |
| 4 | `sample_files_per_class()` — generate `sampled_files` (seed=42, 500/kelas) untuk Cell 5-9 | EDA (support) |
| 5 | `show_sample_grid()` — visual grid sample per kelas | EDA |
| 6 | `compute_background_complexity()` — variance 4 corner patch (30px) untuk estimasi kompleksitas background. Threshold=50 untuk flag `is_plain_bg`. Hasil: Recyclable 38,8% plain bg (tertinggi), Electronic 15,2%, Organic 17,0% — lihat Flag 4 | EDA |
| 7-24 | EDA eksploratif lanjutan lainnya (dimension/aspect ratio, orientasi, within-class & cross-class dupe, Electronic subgroup 150×150 vs natural, dll). **Temuan & angka final ada di section `EXPLORATION REPORT STATUS (Jeremy)` di atas — TIDAK didobel di sini.** Detail cell-by-cell akan diisi menyusul jika diperlukan. | EDA |
| 25 | `get_image_mode()` — audit mode gambar full-scan. Hasil final: 312 non-RGB train (293 P + 17 RGBA + 2 L), 5 non-RGB test (P) | Pipeline |
| 26 | `load_image_as_rgb()` — loader robust P/RGBA/L → RGB (RGBA di-composite ke background putih) | Pipeline (fungsi kunci) |
| 27 | Verifikasi `load_image_as_rgb()` di semua mode non-RGB (train & test) — semua PASSED | Pipeline (validasi) |
| 28 | `WasteDataset` (class) — PyTorch Dataset, return (img, label) atau (img, filename) jika `is_test=True` | Pipeline (class kunci) |
| 29 | `train_transform`, `eval_transform` — pipeline Albumentations/torchvision (RandAugment + HFlip + ColorJitter terbatas untuk train; deterministic untuk eval) | Pipeline |
| 30 | Build `train_df`/`val_df` dari `train_master_with_folds.csv` (fold 0), buat `train_dataset`/`val_dataset`/`train_loader`/`val_loader` | Pipeline |
| 31 | Sanity check batch shape/dtype/value range untuk train & test loader | Pipeline (validasi) |

*(Cell 7-24 di README yang diupload berisi StratifiedGroupKFold split + test loader — ini sudah dieksekusi dan hasilnya tervalidasi, tapi di README numbering-nya beda dari notebook asli. Perlu diklarifikasi ulang nomor cell asli untuk bagian ini saat notebook di-refactor final.)*

### Artifact Files (disimpan ke disk)
| File | Dihasilkan di Cell (approx) | Isi |
|---|---|---|
| `train_master_with_groups.csv` | ~Cell 15-16 (area fold-building, perlu konfirmasi nomor asli) | Semua train file + label + `duplicate_group_id` + `exclude_from_cv`/`exclude_from_training` flags. 26.527 baris. |
| `train_master_with_folds.csv` | ~Cell 18-19 (perlu konfirmasi nomor asli) | Sama seperti atas + kolom `fold` (0-4, -1 = excluded dari CV) |

### Key Variables (Global, dipakai lintas cell)
| Variabel | Tipe | Isi |
|---|---|---|
| `data_dir`, `train_dir`, `test_dir`, `submission_path` | Path | Root direktori project, auto-detect (Cell 0) |
| `class_folders` | list | `["0_Recyclable", "1_Electronic", "2_Organic"]` |
| `sampled_files` | dict | `{class: [file_paths]}`, seed=42, 500 sample/kelas — HANYA untuk EDA visual (Cell 5-9), TIDAK dipakai pipeline training |
| `train_test_duplicates` | dict | `{train_path: test_path}` — 97 pasang exact-dup (Flag 3) |
| `train_master` | DataFrame | Master train file list + label + img_mode + group/exclude flags, 26.527 baris |
| `submission_df` | DataFrame | Test file list, kolom `id`/`filename`/`filepath`/`img_mode` — source of truth urutan test (1458 baris) |
| `fold_df` | DataFrame | Load dari `train_master_with_folds.csv`, dipakai untuk build train/val split |
| `train_df`, `val_df` | DataFrame | Subset `fold_df` untuk fold aktif (`FOLD_TO_VALIDATE`) |
| `train_dataset`, `val_dataset`, `test_dataset` | WasteDataset | PyTorch Dataset instances |
| `train_loader`, `val_loader`, `test_loader` | DataLoader | PyTorch DataLoader instances |

### Key Functions/Classes
| Nama | Cell | Fungsi |
|---|---|---|
| `compute_file_hash(file_path, chunk_size=8192)` | 3 | MD5 hash, chunked read (efisien untuk file besar) |
| `sample_files_per_class(data_dir, class_folders, n_samples=500)` | 4 | Generate sample EDA per kelas, seeded |
| `compute_background_complexity(img, patch_size=30)` | 6 | Rata-rata variance 4 corner patch — proxy kompleksitas background (low=plain/studio, high=natural) |
| `get_image_mode(filepath)` | 25 | Return `img.mode` (RGB/P/RGBA/L), error-safe (return string error kalau corrupt) |
| `load_image_as_rgb(filepath)` | 26 | **Fungsi kunci pipeline.** Convert P/RGBA/L → RGB aman. RGBA di-composite ke background putih (hindari halo hitam di pixel transparan) |
| `WasteDataset(df, transform, is_test)` | 28 | **Class kunci pipeline.** Custom PyTorch Dataset |

### Constants (Locked — jangan ubah tanpa approval)
| Nama | Nilai | Cell | Alasan |
|---|---|---|---|
| `IMG_SIZE` | 224 | 29 | Baseline resolution, lock di `MODELING STRATEGY` |
| `N_SPLITS` | 5 | (fold-building) | StratifiedGroupKFold, lock di `VALIDATION STRATEGY` |
| `SEED` | 42 | multiple | Reproducibility (sampling EDA, fold split) |
| `BATCH_SIZE` | 64 | 30 | ConvNeXt V2-Tiny @ 224px, baseline utk 16GB VRAM. **Catatan: kemungkinan upgrade ke RTX 5090 32GB — belum confirmed, JANGAN naikkan angka ini sampai hardware fix dikonfirmasi.** |
| `NUM_WORKERS` | 4 | 30 | Sesuaikan ke 0 kalau ada masalah multiprocessing di Windows |
| `FOLD_TO_VALIDATE` | 0 | 30 | Fold pertama dipakai untuk baseline sanity check |
| `threshold` (bg_variance) | 50 | 6 | Ambang batas plain-vs-complex background — **kalibrasi manual** setelah inspeksi `describe()`, bukan angka baku/teruji statistik. Worth di-review lagi kalau threshold ini dipakai untuk keputusan modeling (saat ini hanya EDA deskriptif). |

### 🔴 TODO — Belum lengkap, isi menyusul
- [ ] Nomor cell asli untuk StratifiedGroupKFold split (Cell 7-24 di README ini numbering-nya kemungkinan tidak match notebook asli — perlu diklarifikasi ulang setelah refactor selesai)
- [ ] Detail cell-by-cell untuk Cell 6-24 (EDA eksploratif) — akan diisi menyusul kalau perlu direferensikan langsung dari kode, saat ini cukup rujuk ke `EXPLORATION REPORT STATUS`