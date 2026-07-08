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
- Last updated          : 8 Juli 2026
- Updated by             : Ababil (architecture comparison ConvNeXt V2-Tiny vs EfficientNetV2-S CLOSED — champion: ConvNeXt V2-Tiny)

## COMPUTING RESOURCES
- **Status aktual (3 Juli 2026)**: Development/testing SEMENTARA di **Kaggle (Tesla T4, 15.6GB VRAM)** — dipakai karena belum masuk fase training berat & belum ada konfirmasi hardware final dari dosen. **Rencana pindah ke RTX 4090 (kemungkinan, belum fix)** setelah konfirmasi. Sebelumnya sempat dibahas opsi 4080S/5070/5080/5090 (Vast.ai) — itu SUPERSEDED, treat sebagai historical context saja, bukan rencana aktif.
- **Preferred platform**: 
  - **Kaggle** (gratis, T4 16GB) — dipakai SEKARANG untuk development pipeline & baseline testing ringan.
  - **Vast.ai** (bayar per jam, fleksibel) — untuk training berat, GPU spesifik TBD (kemungkinan RTX 4090, belum konfirmasi dosen).
  - Kaggle terbatas 30 jam/minggu & session timeout 9 jam.
- **Coding constraints for Ababil (berlaku umum untuk GPU ~16GB, termasuk T4)**:
  - **Batch size awal (tanpa gradient accumulation)**: 
    - EfficientNet-B0/B1 → 64
    - ResNet50 / EfficientNet-B3 → 32
    - ConvNeXt-Base / ViT-B/16 → 16
    - ConvNeXt V2-Tiny → 64 (tervalidasi jalan di Tesla T4 15.6GB, lihat CURRENT STATUS)
  - **WAJIB** aktifkan `torch.cuda.amp` (mixed precision) untuk menghemat VRAM dan mempercepat training ~2x. **Catatan teknis**: `torch.cuda.amp.GradScaler` sudah deprecated di PyTorch 2.10+, gunakan `torch.amp.GradScaler('cuda')` sebagai gantinya.
  - Jika pakai **Kaggle**: simpan checkpoint ke `/kaggle/working/` (`output_dir`) dan download sebelum session timeout (9 jam). Jangan andalkan penyimpanan permanen. **PENTING: `/kaggle/input/` READ-ONLY — semua operasi `.to_csv()`/checkpoint WAJIB ke `output_dir`, bukan `data_dir`.**
  - Jika pakai **Vast.ai**: pastikan instance memiliki penyimpanan lokal yang cukup untuk dataset (±5-10 GB setelah ekstraksi).
- **Catatan Kritis untuk Ababil**: Dengan ~16GB VRAM (T4 atau RTX 16GB-class), **ConvNeXt V2-Tiny** (28.6M) aman dengan batch size 64 (AMP ON) — **tervalidasi jalan di Tesla T4**. **JANGAN coba ConvNeXt V2-Base** (88.7M) di awal — risikonya OOM atau training terlalu lambat untuk diiterasi. Base hanya dipertimbangkan jika waktu & VRAM berlebih setelah Fase 3 selesai, ATAU jika sudah pindah ke hardware lebih besar (RTX 4090 dsb.) dan dikonfirmasi aman.

## CURRENT STATUS
- Active phase          : Fase 3 in-progress — architecture comparison ConvNeXt V2-Tiny vs EfficientNetV2-S **CLOSED** (8 Juli 2026), kembali ke roadmap Fase 3 utama berbasis ConvNeXt V2-Tiny.
- Last completed stage  : 7 training run total. Champion arsitektur: **exp003 (ConvNeXt V2-Tiny, plain CE, data bersih, CV 0.9815)**. EfficientNetV2-S didiagnosis 3x (exp004, exp004b, exp004c — LR head/warmup, normalisasi+resolusi native) — semua gagal menutup gap performa (best CV 0.9390 vs 0.9815 ConvNeXt). Root cause pasti tidak terpecahkan sepenuhnya; dihentikan atas dasar ROI, bukan karena masalah selesai.
- Next action           : **Architecture comparison ditutup — ConvNeXt V2-Tiny satu-satunya arsitektur aktif mulai sekarang.** Lanjut ke opsi roadmap Fase 3 yang tersisa (berbasis ConvNeXt V2-Tiny): EMA, Label Smoothing, atau training full 5-fold untuk K-fold ensemble Layer 1. **Belum diputuskan mana duluan — perlu keputusan berikutnya.**
- Blocker (if any)      : ⚠️ Checkpoint file exp002 & exp003 (.pt) hilang dari sesi Kaggle — perlu diputuskan (re-train ulang atau cari Notebook Version lama) sebelum lanjut ke tahap yang butuh checkpoint tersebut (mis. ensemble/inference).

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
- Anchor model (Slot 1) : ConvNeXt V2-Tiny — Best CV: **0.9815** (exp003, data BERSIH, plain CE) — LB: belum submit
- Anchor model (Slot 2) : **DITUTUP.** EfficientNetV2-S dieksplorasi (exp004/4b/4c) tapi tidak pernah menutup gap performa vs ConvNeXt V2-Tiny — architecture comparison CLOSED 8 Juli 2026, ConvNeXt V2-Tiny jadi satu-satunya arsitektur aktif.
- Best CV so far (data BERSIH, valid) : **0.9815** — exp_id: exp003 (plain CE, data bersih) — ✅ **champion arsitektur, basis semua eksperimen lanjutan**
- Best LB so far        : belum ada submission (0/3)

**Recent experiments (last 7):**
| exp_id | Stage | Model | Data | Resolusi | CV | LB | Delta vs prior | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | Fase 1 Baseline | ConvNeXt V2-Tiny, plain CE, no aug/EMA/CutMix, fold 0, 15 epoch | KOTOR (pre-relabel) | 224 | 0.9823 | - | - | F1 Electronic=0.994. Ini yang di-Grad-CAM, menemukan 4 mislabel. Checkpoint TERTIMPA oleh exp001-rerun (nama file sama). |
| exp001-rerun | Fase 1 Baseline (re-run tidak sengaja) | ConvNeXt V2-Tiny, plain CE, identik exp001 | **KOTOR juga** (loader tidak di-reload pasca-relabel) | 224 | 0.9827 | - | +0.0004 vs exp001 (stokastisitas training, BUKAN efek relabel) | F1 Electronic=0.995. Checkpoint: `model_s9_baseline_cv0.9827.pt` |
| exp002 | Fase 3 | ConvNeXt V2-Tiny, Class-Balanced Loss (beta=0.9999) + Weighted Random Sampler | **BERSIH** (pertama kali pakai data pasca-relabel) | 224 | 0.9809 | - | -0.0018 vs exp001-rerun (**TIDAK VALID dibandingkan — beda data DAN beda loss sekaligus**) | F1 Electronic=0.992. Checkpoint: `model_exp002_cbloss_cv0.9809.pt`. **⚠️ Checkpoint file .pt sekarang hilang dari sesi Kaggle — belum diputuskan re-train atau cari Notebook Version lama.** |
| exp003 | Fase 3 | ConvNeXt V2-Tiny, plain CE (TANPA CB-Loss/Sampler), **data bersih** | **BERSIH** | 224 | **0.9815** ✅ | - | +0.0006 vs exp002 (**VALID — satu variabel beda: loss/sampler**) | F1[Recy=0.972, Elec=0.995, Org=0.978]. Best epoch 14/15. **BEST — champion arsitektur, basis semua eksperimen lanjutan.** Checkpoint: `model_exp003_cleanbaseline_cv0.9815.pt`. **⚠️ Checkpoint file .pt sekarang hilang dari sesi Kaggle.** History: `training_history_exp003_cleanbaseline.csv`. |
| exp004 | Fase 3 (architecture comparison) | EfficientNetV2-S, plain CE (LR asli) | BERSIH | 224 (❌ salah — bukan resolusi native) | 0.9368 | - | jauh di bawah exp003 | Diagnosis #1: resolusi 224 bukan native EfficientNetV2-S. Selesai, gagal. |
| exp004b | Fase 3 (architecture comparison) | EfficientNetV2-S, plain CE (LR tuned/head+warmup) | BERSIH | 224 (❌ masih salah) | 0.6528* | - | jauh di bawah exp004 | *Baru 1 epoch, dihentikan lebih awal — LR tuning tidak menutup gap, malah lebih buruk di epoch awal. Diagnosis #2, gagal. |
| exp004c | Fase 3 (architecture comparison) | EfficientNetV2-S, plain CE (LR asli, kembali ke config awal) | BERSIH | 300 (✅ native, fixed) | 0.9390 | - | masih jauh di bawah exp003 (-0.0425) | Diagnosis #3: normalisasi + resolusi native dikoreksi, tetap tidak menutup gap. **Architecture comparison ditutup atas dasar ROI setelah exp004c** — root cause pasti tidak terpecahkan sepenuhnya (kemungkinan backbone LR arsitektur-spesifik atau training recipe fundamental berbeda), bukan karena EfficientNetV2-S inheren buruk. |

**Kesimpulan ablasi exp002 vs exp003 (VALID, apple-to-apple):**
- Plain CE (exp003): CV=0.9815, F1 Electronic=0.9950
- CB-Loss+Sampler (exp002): CV=0.9809, F1 Electronic=0.9919
- **Verdict: CB-Loss+Sampler TIDAK memberi keuntungan** pada dataset ini (imbalance 3,2:1 tidak cukup ekstrem, Electronic sudah near-ceiling). Lanjutkan iterasi Fase 3 tanpa CB-Loss sebagai default.

**Kesimpulan architecture comparison ConvNeXt V2-Tiny vs EfficientNetV2-S (CLOSED, 8 Juli 2026):**
- Pemenang: **ConvNeXt V2-Tiny** (exp003, CV 0.9815) vs EfficientNetV2-S terbaik (exp004c, CV 0.9390) — gap 0.0425.
- EfficientNetV2-S didiagnosis 3x (LR head/warmup di exp004b, normalisasi+resolusi native di exp004c) — semua gagal menutup gap. Root cause pasti **tidak terpecahkan sepenuhnya**; keputusan dihentikan atas dasar **ROI eksperimen lanjutan yang menurun tajam**, bukan karena masalah sudah selesai dianalisis.
- **Untuk laporan akhir, dokumentasikan sebagai**: "EfficientNetV2-S dieksplorasi dengan 3 iterasi debugging sistematis (LR tuning, normalisasi, resolusi native); tetap underperform signifikan vs ConvNeXt V2-Tiny meski sudah dikoreksi ke pretrained config yang benar — kemungkinan root cause di luar scope komponen yang diuji (mis. backbone LR arsitektur-spesifik atau training recipe yang berbeda secara fundamental)." **Jangan** simpulkan "EfficientNetV2-S secara inheren buruk" — klaim itu tidak didukung evidence yang cukup.
- **Dampak ke MODELING STRATEGY / ENSEMBLE STRATEGY**: Partner Model EfficientNetV2-S (wajib per revisi 4 Juli 2026) dan Layer 2 ensemble multi-arsitektur **tidak lagi applicable** dengan hasil ini — perlu direvisi/diputuskan ulang saat masuk Fase 5 (lihat catatan di bagian ENSEMBLE STRATEGY).

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
- **Partner Model (wajib, bukan lagi kondisional — REVISI 4 Juli 2026)**: **EfficientNetV2-S** (21.5M params) — dilatih dengan CV setup identik (fold & seed sama) untuk comparison eksplisit DAN sebagai komponen ensemble wajib (lihat ENSEMBLE STRATEGY di bawah).
- **Model Ditolak**: ConvNeXt V2-Base / EfficientNetV2-M / Swin V2-Tiny sebagai ANCHOR (terlalu berat/rewel untuk fase awal). ResNet50 sebagai "classic baseline" tambahan DITOLAK untuk re-training — cukup dirujuk dari literatur (nilai tambah marginal rendah dibanding compute cost). Model 3+ hanya dipertimbangkan sebagai kandidat sekunder jika baseline+partner solid & hardware memadai (RTX 4090 approved).

### ENSEMBLE STRATEGY (REVISI 4 Juli 2026 — dari kondisional jadi minimal-wajib)
**Alasan revisi**: Perlu jawaban defensible untuk pertanyaan "kenapa tidak ensemble" tanpa memerlukan compute besar. Strategi dipecah 2 layer:

| Layer | Komposisi | Compute Cost | Status |
|---|---|---|---|
| **Layer 1 — K-fold ensembling** | 5 model dari 5-fold CV, arsitektur SAMA (ConvNeXt V2-Tiny) | Hampir gratis — model sudah ada dari CV, cuma tambahan inference | **WAJIB, minimum viable ensemble** |
| **Layer 2 — Multi-architecture ensemble** | 5 fold ConvNeXt V2-Tiny + 5 fold EfficientNetV2-S = 10 model total | Tinggi — butuh training EfficientNetV2-S penuh | **❌ TIDAK APPLICABLE (8 Juli 2026)** — syarat (a) CV score sebanding GAGAL: EfficientNetV2-S terbaik (exp004c, CV 0.9390) tertinggal 0.0425 dari ConvNeXt V2-Tiny (exp003, CV 0.9815) setelah 3x diagnosis. Architecture comparison ditutup atas dasar ROI. Perlu keputusan ulang di Fase 5 apakah dicari arsitektur partner lain atau cukup Layer 1 saja. |

- **Bobot ensemble**: **Simple average (equal weight)** dulu. Weighted/OOF-based weighting adalah eksplorasi **opsional** di Fase 5, hanya kalau ada waktu & compute sisa.
- **Validasi**: Murni CV lokal (StratifiedGroupKFold) — TIDAK perlu submit ke leaderboard untuk evaluasi eksperimen ensemble. Submission budget (3x) dijaga hanya untuk kandidat final.
- **Rationale "sama-sama kuat" untuk ensemble**: Kondisi ideal ensemble adalah performa sebanding + error tidak berkorelasi tinggi (beda arsitektur → cenderung salah di kasus berbeda). Kalau satu model jauh lebih kuat, ensemble berisiko menurunkan performa (model lemah menarik turun prediksi model kuat).

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

**Sensitivity Analysis / Ablation Table (WAJIB, REVISI 4 Juli 2026):**
Setiap komponen augmentasi di atas WAJIB dicatat sebagai baris terpisah di tabel ablation berikut, dengan CV score (fold & seed identik) sebagai bukti kuantitatif kontribusi tiap komponen — bukan cuma keputusan on/off tanpa data:

| Konfigurasi | Macro F1 (CV) | F1 Electronic (CV) | Delta vs Baseline | Keputusan |
|---|---|---|---|---|
| Baseline (no aug, plain CE) | TBD | TBD | - | exp001 |
| +RandAugment+Flip+ColorJitter | TBD | TBD | TBD | TBD |
| +CutMix (kondisional Grad-CAM) | TBD | TBD | TBD | TBD |
| +Mixup | TBD | TBD | TBD | TBD |

Diisi progresif seiring Fase 2-4 berjalan. Setiap baris WAJIB pakai StratifiedGroupKFold identik supaya perbandingan apple-to-apple.

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
- [x] **Architecture comparison ConvNeXt V2-Tiny vs EfficientNetV2-S**: Keputusan final = ConvNeXt V2-Tiny menang (CV 0.9815 vs 0.9390), EfficientNetV2-S dihentikan setelah 3x diagnosis (exp004/4b/4c) atas dasar ROI. **CLOSED 8 Juli 2026.** Lihat EXPERIMENT LOG SUMMARY untuk detail dan template dokumentasi laporan akhir.

### 🔴 HIGH — NEXT ACTIONS (Ababil)
- [x] **Ababil** — Group ID assignment untuk StratifiedGroupKFold: SELESAI. `train_master_with_groups.csv` tersimpan (26.527 baris, 26.463 unique groups setelah exact-dup + 2 near-dup Electronic pairs `627(1)/627.jpeg` & `629(1)/629.jpeg` digabung manual). exclude_from_cv=97 (match), exclude_from_training=1 (`O_8873.jpg`, match). Semua assertion PASSED termasuk cross-class group check.
- [x] **Ababil** — StratifiedGroupKFold split: SELESAI. `train_master_with_folds.csv` tersimpan. 5 fold (5285-5286 tiap fold), group integrity PASSED (tidak ada grup terbelah), class balance deviation maksimum 0.02pp dari global ratio (Recyclable 37.83%, Electronic 14.98%, Organic 47.19%) — jauh di bawah threshold 2pp, tidak ada tindakan lanjutan diperlukan.
- [x] **Ababil** — Test loader: SELESAI. `submission.csv` dipakai sebagai source of truth urutan ID, semua 1458 file test tervalidasi ada di disk & urutan match.
- [x] **Ababil** — Image mode converter (`load_image_as_rgb`): SELESAI & tervalidasi untuk P/RGBA/L (lihat update di DATASET section untuk angka detail).
- [x] **Ababil** — PyTorch Dataset/DataLoader class: SELESAI & tervalidasi penuh (Cell 28-31). Batch shape [64,3,224,224], label unique [0,1,2], test filename order match submission.csv persis.
- [x] **Ababil** — Environment porting Local ↔ Kaggle: SELESAI. Notebook sekarang jalan di kedua environment dengan pola `data_dir` (read-only)/`output_dir` (write) konsisten.
- [x] **Ababil** — Model architecture setup: SELESAI. ConvNeXt V2-Tiny (timm) loaded, discriminative LR split (backbone 1e-5 / head 1e-4) berhasil, terverifikasi jalan di Tesla T4.
- [x] **Ababil** — Baseline training loop: SELESAI (exp001). Plain CrossEntropyLoss, cosine+warmup, AMP (`torch.amp.GradScaler('cuda')` — fixed dari deprecated call), tanpa EMA. Best val_macro_f1=0.9823 (epoch 10). Checkpoint tersimpan.
- [x] **⚠️ Ababil — Investigasi kecurigaan shortcut learning (subpopulation check)**: SELESAI. Recall Icon(150×150)=0.9927, Natural=0.9918, gap=0.0009. **Hipotesis shortcut ukuran/format DITOLAK** — model generalize sama baik di kedua subpopulasi Electronic.
- [x] **Ababil/Jeremy — Grad-CAM check**: SELESAI. 24 sample (16 true-positive dengan kontrol plain-bg/complex-bg + icon/natural, 10 error case Electronic FP/FN). **Hasil: TIDAK ADA shortcut learning terdeteksi.** CutMix TIDAK diaktifkan.
- [x] **🆕 Ababil — Verifikasi manual kandidat mislabel baru**: SELESAI. Semua 4 kandidat terkonfirmasi mislabel (2 laptop, 1 panel, 1 botol). Relabel dieksekusi.
- [x] **Ababil** — Reload DataLoader (Cell 30) dari `train_master_with_folds.csv` yang sudah direlabel, lalu retrain baseline sebagai exp002/exp003 untuk dapat CV score dengan data bersih. **SELESAI — exp003 (CV 0.9815) jadi champion.**
- [x] **Ababil** mulai **Fase 3 — Optimasi Electronic** tahap loss/sampler: Class-Balanced Loss + Weighted Random Sampler (exp002) vs plain CE (exp003). **SELESAI — plain CE menang, CB-Loss tidak dipakai sebagai default.**
- [x] **Ababil** training **EfficientNetV2-S** untuk model comparison eksplisit (exp004/4b/4c). **SELESAI & CLOSED 8 Juli 2026 — EfficientNetV2-S kalah signifikan, architecture comparison ditutup atas dasar ROI. Layer 2 ensemble multi-arsitektur jadi tidak applicable (lihat ENSEMBLE STRATEGY).**
- [ ] **Ababil** — Putuskan nasib **checkpoint exp002 & exp003 yang hilang** dari sesi Kaggle: re-train ulang, atau cari Notebook Version lama di Kaggle. **Prioritas tinggi — blocking untuk tahap ensemble/inference berikutnya.**
- [ ] **Ababil** lanjut **Fase 3 sisa roadmap (berbasis ConvNeXt V2-Tiny saja)** — pilih salah satu/urutan: EMA, Label Smoothing, atau training full 5-fold untuk K-fold ensemble Layer 1. **Keputusan arah berikutnya belum diambil.**
- [ ] **Ababil** isi **Ablation Table** (baseline → +RandAugment → +CutMix → +Mixup) di MODELING STRATEGY seiring eksperimen berjalan — dengan catatan CutMix baris ini kemungkinan tetap "not activated" karena Grad-CAM sudah clear (kecuali ada perubahan bukti baru).
- [ ] **Ababil** jalankan **Baseline Training ConvNeXt V2-Tiny** (resolusi 224, tanpa CutMix/Mixup, tanpa Class-Balanced Loss dulu) sebagai titik referensi bersih. **[SELESAI — lihat exp001 di atas, tapi hasil masih perlu divalidasi sebelum dianggap "clean baseline"]**
- [ ] **Ababil/Jeremy** jalankan **Grad-CAM check** segera setelah baseline convergen — hasilnya menentukan apakah CutMix masuk roadmap atau tidak. **[STATUS: NOW BLOCKING — baseline sudah convergen, Grad-CAM harus jalan sebelum keputusan lain apapun]**
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

- **[4 Juli 2026] [Ababil]:** **REVISI STRATEGI — Ensemble & Model Comparison.** Latar belakang: perlu jawaban defensible untuk pertanyaan "kenapa tidak ensemble" mengingat compute terbatas (T4 5-fold CV single model estimasi 10.5-16.5 jam, vs RTX 4090 ~2.5-4.2 jam — masih pending approval dosen). Keputusan:
  1. **Ensemble** berubah dari kondisional ("hanya jika CV membuktikan unggul") jadi **minimal-wajib** 2 layer: (a) K-fold ensembling (5 model ConvNeXt dari 5-fold CV, hampir gratis) — WAJIB; (b) Multi-architecture ensemble (5 ConvNeXt + 5 EfficientNetV2-S = 10 model) — diaktifkan jika CV sebanding & OOF correlation rendah.
  2. Bobot ensemble: **simple average dulu**, weighted/OOF-based jadi eksplorasi opsional Fase 5 kalau ada compute/waktu sisa.
  3. **Model comparison** dipatok di 2 model (ConvNeXt V2-Tiny vs EfficientNetV2-S) — usulan tambah ResNet50 sebagai "classic baseline" DITOLAK (nilai tambah marginal rendah vs compute cost, hasil sudah predictable kalah dari arsitektur modern; cukup dirujuk dari literatur di laporan).
  4. **Sensitivity analysis** diformalkan jadi ablation table augmentasi eksplisit (baseline → +RandAugment → +CutMix → +Mixup), masing-masing dengan CV score tercatat — bukan cuma keputusan on/off tanpa data kuantitatif.
  5. Validasi semua eksperimen di atas **murni CV lokal**, tidak perlu submit ke leaderboard — submission budget (3x) dijaga untuk kandidat final saja.
  6. Compute strategy: tetap Kaggle untuk sekarang, RTX 4090 masih pending approval dosen — EfficientNetV2-S training dijadwalkan setelah baseline ConvNeXt selesai & sesuai kuota yang tersisa.
- **[4 Juli 2026] [Ababil]:** **Manual verifikasi 4 kandidat mislabel dari Grad-CAM — SEMUA TERKONFIRMASI MISLABEL.** R_3825.jpg & R_3733.jpg (dilabel Recyclable) ternyata jelas foto laptop (Surface Book) → seharusnya Electronic. O_7776.jpg (dilabel Organic) ternyata panel kontrol elektronik → seharusnya Electronic. battery_61.jpg (dilabel Electronic) ternyata foto botol minuman berjejer → seharusnya Recyclable. Model baseline SEBENARNYA BENAR pada 4 kasus ini — yang salah adalah ground truth. Implikasi: F1 0.9823 kemungkinan sedikit underestimate performa asli model.
- **[4 Juli 2026] [Ababil]:** **Investigasi filename non-standar Electronic (1.039 file, 26,2% dari total Electronic) — CLEAR, bukan mislabel.** Debug regex mengonfirmasi ini bukan bug: Recyclable & Organic 100% pakai format seragam (R_####.jpg / O_####.jpg), sementara Electronic satu-satunya kelas dengan format beragam — indikasi kelas ini dikumpulkan dari sumber campuran/scraping (bukan foto internal terstruktur). Dari 333 file caption-like (nama berupa kalimat/deskriptif), sample stratified 10 file diverifikasi visual: SEMUA 10 benar berlabel Electronic (tumpukan e-waste, PCB, komputer rusak, dst — beberapa dari stock photo site seperti Shutterstock/PNGTree). Kesimpulan: filename non-standar ini bukan noise, kemungkinan besar augmentasi sumber data yang sah untuk memperkaya variasi visual kelas Electronic. Tidak menambah temuan mislabel baru — total tetap 4 file dari investigasi Grad-CAM.
- **[4 Juli 2026] [Ababil]:** **KEPUTUSAN PENDING**: 4 file mislabel terkonfirmasi (R_3825, R_3733, O_7776, battery_61) — perlu keputusan exclude vs relabel sebelum training Fase 3 dimulai.
- **[4 Juli 2026] [Ababil]:** **Relabel 4 file mislabel — SELESAI DIEKSEKUSI.** Keputusan: relabel manual (bukan exclude), karena hanya 4 file dan sudah diverifikasi visual langsung. R_3825.jpg: 0→1, R_3733.jpg: 0→1, O_7776.jpg: 2→1, battery_61.jpg: 1→0. Backup pre-relabel tersimpan: `train_master_with_folds_PRE_RELABEL_BACKUP.csv`. Distribusi kelas pasca-relabel nyaris tidak berubah (Recyclable 37,83%→37,83%, Electronic 14,97%→14,99%, Organic 47,18%→47,18%) — dampak signifikan tidak diharapkan dari 4 file, tapi tetap dicatat untuk audit trail. `train_master_with_folds.csv` (versi kerja/output) sudah ter-update; siap untuk training ulang.
- **[4 Juli 2026] [Ababil]:** **CATATAN caption-like investigation**: 10/333 sample diverifikasi visual, semua benar (tidak ada mislabel tambahan ditemukan). Keputusan: 10-sample dianggap cukup representatif, TIDAK cek 333 file satu-satu — trade-off waktu vs kepastian marjinal disetujui secara sadar (bukan diabaikan begitu saja).
- **[8 Juli 2026] [Ababil]:** **exp003 (true clean baseline) SELESAI.** ConvNeXt V2-Tiny, plain CE, data bersih, 15 epoch, fold 0. Best val_macro_f1=**0.9815** di epoch 14. F1[Recy=0.9720, Elec=0.9950, Org=0.9776]. Training stabil: val loss plateau di ~0.074 setelah epoch 9, tidak ada tanda overfitting signifikan meski train_acc mencapai 0.9989 di epoch 15. Checkpoint: `model_exp003_cleanbaseline_cv0.9815.pt`. **Perbandingan valid exp002 vs exp003 (satu variabel beda)**: CB-Loss+Sampler (0.9809) vs plain CE (0.9815) → **CB-Loss TIDAK membantu** (+0.0006 untuk plain CE). Verdict: plain CE tetap jadi default untuk iterasi Fase 3 selanjutnya.
- **[4 Juli 2026] [Ababil]:** **SOURCE CODE REFERENCE disinkronkan penuh** dari source code asli (35 cell: 0, 0b, 0c, 1-34, termasuk 32b) yang diupload Ababil — section sebelumnya berisi banyak nomor cell perkiraan/belum terkonfirmasi, sekarang akurat 1:1 dengan kode berjalan. Perubahan struktural penting: pipeline EDA+cleaning+modeling ternyata linear dalam satu notebook (bukan terpisah seperti draft lama), Grad-CAM+mislabel-cleaning masuk sebagai Cell 28-34 (bukan tahap terpisah di luar notebook). Lihat section SOURCE CODE REFERENCE di bawah untuk detail lengkap.
- **[4 Juli 2026] [Ababil]:** **KOREKSI PENOMORAN EKSPERIMEN — penting untuk audit trail.** 3 training run sudah dilakukan, TERNYATA baru 1 yang pakai data bersih:
  1. **exp001** (training pertama): plain CE, **data KOTOR** (sebelum relabel Cell 34). CV macro F1=**0.9823**. Ini yang di-Grad-CAM & menemukan 4 mislabel.
  2. **exp001-rerun** (training kedua, TIDAK direncanakan sebagai eksperimen baru, sekadar re-run Cell 27 biasa): plain CE, **DATA MASIH KOTOR juga** (loader tidak di-rebuild dari CSV yang sudah direlabel — relabel Cell 34 baru dieksekusi TAPI loader di memori/run ini belum reload). CV macro F1=**0.9827** — beda dari exp001 murni karena stokastisitas training (RandAugment, dst.), BUKAN karena data beda. **File checkpoint/CSV-nya menimpa nama yang sama dengan exp001** (`model_s9_baseline_cv{score}.pt`, `training_history_baseline.csv`) — jadi angka 0.9823 asli sudah tertimpa di disk, cuma tercatat di CATATAN BEBAS entry sebelumnya.
  3. **exp002** (Cell 35 baru — Class-Balanced Loss + Weighted Random Sampler): ini eksperimen PERTAMA yang benar-benar pakai **data bersih** (loader di-rebuild dari `train_master_with_folds.csv` pasca-relabel di dalam Cell 35 itu sendiri). CV macro F1=**0.9809**.
  - **⚠️ GAP PENTING**: belum ada baseline plain CE dengan data bersih untuk pembanding apple-to-apple terhadap exp002. Perbandingan 0.9809 (exp002) vs 0.9827 (exp001-rerun, data kotor) **TIDAK VALID** karena 2 variabel berubah sekaligus (loss function DAN kebersihan data). **exp003 yang seharusnya dijalankan dulu**: plain CE + data bersih (pakai Cell 21 rebuild loader dari CSV bersih, TANPA CB-Loss/Sampler) sebagai true baseline pembanding.
- **[4 Juli 2026] [Ababil]:** **exp002 (CB-Loss+Sampler) hasil awal — perlu direvalidasi ulang setelah true clean baseline (exp003) ada.** Sementara ini (dibandingkan dengan data kotor, TIDAK VALID secara metodologi): F1 Electronic turun dari 0.9950→0.9919, macro F1 turun 0.9827→0.9809. Kemungkinan penyebab (masih hipotesis, belum firm): (a) Electronic sudah near-ceiling di baseline, ruang improvement CB-Loss terbatas; (b) oversampling agresif menyebabkan overfit ke variasi terbatas Electronic (cuma 3.166 sample asli); (c) CB-Loss weight (1.56x untuk Electronic) mungkin terlalu agresif untuk imbalance yang tidak terlalu ekstrem (3,2:1). **Kesimpulan DITUNDA sampai exp003 (clean baseline) selesai untuk perbandingan yang valid.**
- **[4 Juli 2026] [Ababil]:** **Grad-CAM diagnostic SELESAI — GATE FASE 1→2 CLEARED.** Total 24 sample dianalisis dalam 2 batch:
  - **Batch 1 (16 sample, semua prediksi benar)**: Recyclable plain-bg (5) + Recyclable complex-bg kontrol (3) + Electronic icon150 (5) + Electronic natural kontrol (3). Hasil: aktivasi konsisten fokus ke objek/fitur diskriminatif (teks, logo, badan objek), TIDAK ke background kosong. 1 sample agak menyebar ke area sekitar objek kecil (karet/gelang) tapi tidak signifikan.
  - **Batch 2 (10 sample, error case Electronic — 4 FP + 6 FN)**: SEMUA aktivasi tetap fokus ke objek, bahkan pada kasus salah prediksi. Error ternyata didominasi 2 sumber: (a) ambiguitas visual genuine (logam bisa Recyclable/Electronic), (b) **kandidat mislabel ground truth baru** — minimal 3-4 kasus terlihat sangat jelas salah label secara visual (2 gambar laptop dilabel Recyclable, 1 panel/tombol elektronik dilabel Organic, 1 foto botol-botol dilabel Electronic).
  - **Kesimpulan final**: TIDAK ADA bukti shortcut learning (background maupun ukuran icon — icon check sudah ditolak sebelumnya, background check sekarang juga ditolak). F1 0.9823 kemungkinan genuine, bahkan mungkin sedikit underestimate karena sebagian "error" adalah noise label, bukan model salah. **Keputusan**: CutMix TIDAK diaktifkan di Fase 3. Kandidat mislabel baru di atas akan diverifikasi manual oleh Ababil (bukan didelegasikan ke Jeremy, karena scope kecil — 3-4 kasus) sebelum diputuskan exclude/relabel. Gambar Grad-CAM tersimpan: `gradcam_shortcut_check.png`, `gradcam_electronic_errors.png`.
- **[4 Juli 2026] [Ababil]:** **Baseline training (exp001) SELESAI.** ConvNeXt V2-Tiny, plain CE Loss, 15 epoch, fold 0, di Kaggle T4 (~227s/epoch, total training ~57 menit). Best val_macro_f1=0.9823 di epoch 10 (checkpoint tersimpan: `model_s9_baseline_cv0.9823.pt`). **Kecurigaan awal**: F1 Electronic=0.994 — tertinggi dari 3 kelas, padahal Electronic adalah kelas minoritas (14,9%) yang menurut Flag 1 seharusnya paling sulit/rendah F1-nya. Ini sempat dikira konsisten dengan hipotesis H3 & Flag 4/5 (Electronic 68,8% berupa icon 150×150 seragam) — dugaan model belajar shortcut ukuran/format gambar. Train-test leakage (Flag 3) dan within-class dupe leakage (Flag 6) sudah dikonfirmasi TIDAK jadi penyebab dari awal — sanity check Cell 30 semua PASSED (train size 21.143/val 5.286 sesuai ekspektasi, group integrity PASSED, class balance deviation max 0.0005).
- **[4 Juli 2026] [Ababil]:** **Subpopulation sensitivity check (icon vs natural) SELESAI — hipotesis shortcut ukuran DITOLAK.** Hasil: Total Electronic di val=793 (69,2% icon/30,8% natural — proporsi konsisten dengan EDA Jeremy 68,8%/31,2%). Recall Icon=0.9927 (545/549), Recall Natural=0.9918 (242/244), gap=0.0009 — nyaris nol. **Koreksi dari kecurigaan awal**: F1 Electronic tinggi TIDAK terbukti berasal dari shortcut ukuran icon — model generalize sama baik di kedua subpopulasi. F1 tinggi kemungkinan genuine (elektronik punya sinyal visual distinctive: brightness rendah 142 vs Recyclable 182, warna R≈G≈B metalik vs Organic R-dominan). **Namun** ini TIDAK menyingkirkan kemungkinan shortcut BACKGROUND (Flag 4) yang tidak tercakup analisis ukuran gambar — Grad-CAM tetap wajib dijalankan sebagai gate terakhir Fase 1→2, fokus sekarang murni ke cek fokus spasial model (objek vs background), bukan lagi ukuran/format gambar.
- **[3 Juli 2026] [Ababil]:** **Model architecture setup SELESAI.** ConvNeXt V2-Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k` via timm) berhasil di-load & jalan di Kaggle Tesla T4 (15.6GB VRAM). Discriminative LR split backbone (27,86M params, LR=1e-5) vs head `model.head.fc` (3.843 params, LR=1e-4) berhasil. Keputusan: **baseline pertama pakai plain CrossEntropyLoss** (bukan Class-Balanced Loss) — sesuai roadmap Fase 1 (baseline polos dulu, Class-Balanced baru masuk Fase 3). Minor: `torch.cuda.amp.GradScaler` deprecated di PyTorch 2.10+, ganti ke `torch.amp.GradScaler('cuda')`.
- **[3 Juli 2026] [Ababil]:** **Update rencana hardware — GANTIKAN rencana lama.** Development/testing SEMENTARA pakai **Kaggle (Tesla T4, 15.6GB)** karena belum masuk training berat & belum ada konfirmasi hardware final dari dosen. Kemungkinan pindah ke **RTX 4090** (bukan lagi 4080S/5070/5080/5090 yang dibahas sebelumnya) — masih belum fix, tunggu konfirmasi dosen. Opsi GPU Vast.ai lama di section COMPUTING RESOURCES sudah di-update untuk reflect ini.
- **[3 Juli 2026] [Ababil]:** **Environment porting Local → Kaggle SELESAI.** Notebook berhasil dijalankan di Kaggle dengan struktur dataset nested (`/kaggle/input/datasets/<username>/<slug>/BDC2026/`) — auto-detect `data_dir` pakai pencarian generic (cari folder yang punya `train/`+`test/`+`submission.csv` sekaligus, tidak hardcode path). 3 CSV exclude-list (`train_test_overlap.csv`, `train_duplicate_groups.csv`, `near_duplicate_candidates.csv`) di-upload sebagai dataset terpisah (`dupe-exclude-satria-data-2026`) dan dihitung ULANG di Kaggle (bukan pakai hasil lokal) — keputusan: pakai hasil hitung ulang Kaggle sebagai source of truth baru. **Pelajaran penting**: sempat beberapa kali OSError (`Read-only file system`) karena residual `data_dir` dipakai untuk operasi `.to_csv()` — Kaggle `/kaggle/input/` READ-ONLY, semua tulis wajib ke `output_dir` (`/kaggle/working/`). Pola yang sekarang dipegang: `data_dir` = read-only (baca gambar/CSV asal), `output_dir` = write (semua hasil pipeline). Fixed di Cell 15 dan Cell 20 (README numbering Kaggle, bisa beda dari notebook lokal asli).
- **[3 Juli 2026] [Ababil]:** DataLoader Fase 1 — SELESAI TOTAL & tervalidasi end-to-end (Cell 31 sanity check). Batch shape `[64,3,224,224]`, dtype float32, label unique `[0,1,2]`, pixel range post-normalize -2.12 to 2.64 (wajar ImageNet stats), test filename order `('1.jpg','2.jpg','3.jpg','4.jpg','5.jpg')` — **match persis urutan submission.csv**, bukan urutan sistem file. Train/Val split fold 0: 21.143/5.286, class balance konsisten (~37,8%/15,0%/47,2% di kedua split).
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

## SOURCE CODE REFERENCE (REVISI 4 Juli 2026 — sinkron dari source code asli yang diupload Ababil)
*Ini index/kamus, BUKAN copy-paste kode lengkap. Numbering di bawah ini mengikuti KOMENTAR di dalam kode (`# Cell N`), bukan posisi cell fisik di notebook — beberapa cell (0b, 0c, 32b) adalah sisipan non-sekuensial. Update section ini kalau ada refactor besar lagi.*

### Peta Cell → Fungsi (akurat per 4 Juli 2026)
| Cell | Fungsi | Kategori |
|---|---|---|
| 0 | Setup `data_dir`, `train_dir`, `test_dir`, `submission_path`, `output_dir` — auto-detect Kaggle vs Local | Setup |
| 0b | Generate `train_test_overlap.csv`, `train_duplicate_groups.csv`, `near_duplicate_candidates.csv` via MD5 hashing (`compute_file_hash`, `hash_all`) + regex copy-pattern near-dup detection | EDA (data prep) |
| 0c | Load 3 CSV dari Cell 0b ke `overlap_df`, `groups_df`, `near_dup_df` | Pipeline |
| 1 | Build `train_master` (DataFrame dasar): scan `train_dir`, kolom `filename`/`filepath`/`label`/`label_folder` | Pipeline |
| 2 | Merge exact duplicate `duplicate_group_id` ke `train_master` (assert group count match) | Pipeline |
| 3 | Merge near-duplicate `duplicate_group_id` (offset `NEAR_DUP_GROUP_START=900000`) | Pipeline |
| 4 | Flag `exclude_from_cv` (97 train-test overlap) & `exclude_from_training` (`MISLABEL_EXCLUDE = {"O_8873.jpg"}`) | Pipeline |
| 5 | Validasi integritas `train_master` (assert count, no dup filename, cross-class group check) → save `train_master_with_groups.csv` | Pipeline (validasi) |
| 6 | `sample_files_per_class()`, `show_sample_grid()` — visual grid sample per kelas (seed=42, n=500/kelas) → `sampled_files` | EDA |
| 7 | `compute_background_complexity()` — variance 4-corner patch (30px), threshold=50 → `df_bg`, `is_plain_bg` | EDA |
| 8 | Dimension/aspect ratio/orientation analysis → `df_dim` | EDA |
| 9 | Electronic subgroup: 150×150 icon vs natural (pakai `df_dim`) | EDA |
| 10 | Visual comparison Recyclable vs Organic | EDA |
| 11 | Brightness & RGB channel mean per kelas → `df_brightness` | EDA |
| 12 | **StratifiedGroupKFold split** (`N_SPLITS=5`, `SEED=42`) pada `cv_pool` (exclude cv+training flags) → kolom `fold` | Pipeline (KUNCI) |
| 13 | Validasi fold: group integrity (no leakage), class balance deviation (<2pp), fold size | Pipeline (validasi) |
| 14 | Merge `fold` ke `train_master` (excluded rows → fold=-1) → save `train_master_with_folds.csv` | Pipeline (KUNCI) |
| 15 | Load `submission_df` dari `submission.csv`, build `filename`/`filepath` dari `id`, verify semua file ada | Pipeline |
| 16 | `get_image_mode()` — audit mode gambar full-scan → 312 non-RGB train, 5 non-RGB test | Pipeline |
| 17 | `load_image_as_rgb()` — loader robust P/RGBA/L → RGB (RGBA composite ke putih) | Pipeline (fungsi kunci) |
| 18 | Verifikasi `load_image_as_rgb()` di semua mode non-RGB (train & test) | Pipeline (validasi) |
| 19 | `WasteDataset` (class) — PyTorch Dataset | Pipeline (class kunci) |
| 20 | `train_transform`, `eval_transform` (`IMG_SIZE=224`, RandAugment+HFlip+ColorJitter utk train) | Pipeline |
| 21 | Build `train_df`/`val_df`/loaders dari `train_master_with_folds.csv` (`FOLD_TO_VALIDATE=0`, `BATCH_SIZE=64`, `NUM_WORKERS=4`) | Pipeline |
| 22 | Sanity check batch shape/dtype/value range (train & test loader) | Pipeline (validasi) |
| 23 | Install `timm`, import training utils (`torch.amp.GradScaler`/`autocast` — non-deprecated) | Setup |
| 24 | Build model: `timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=3)` → `model`, `DEVICE` | Model |
| 25 | Discriminative LR: `BACKBONE_LR=1e-5`, `HEAD_LR=1e-4`, `WEIGHT_DECAY=0.05` → `optimizer` (AdamW, 2 param groups) | Model |
| 26 | Loss/scheduler/scaler: `criterion` (plain CE), `NUM_EPOCHS=15`, cosine+warmup (`WARMUP_STEPS_RATIO=0.075`) → `scheduler`, `scaler` | Model |
| 27 | **Training loop** (dengan fallback checkpoint resume dari `/kaggle/input/datasets/*/phase0/*.pt` atau `output_dir`) → `history`, checkpoint `model_s9_baseline_cv{score}.pt`, `training_history_baseline.csv` | Model (KUNCI) |
| 28 | Subpopulation sensitivity check (icon 150×150 vs natural) pada Electronic — hasil: gap 0.0009, hipotesis shortcut ukuran DITOLAK | Diagnostik |
| 36b | **exp003 — true clean baseline**: plain CE + data bersih (TANPA CB-Loss/Sampler). Rebuild loader dari `train_master_with_folds.csv`, fresh model, identik setup exp002 kecuali loss & sampler. Best CV=0.9815 (epoch 14). Output: `model_exp003_cleanbaseline_cv0.9815.pt`, `training_history_exp003_cleanbaseline.csv` | Model (Fase 3) |
| 29 | Grad-CAM shortcut check (16 sample: plain/complex bg × icon/natural, dengan kontrol) → `gradcam_shortcut_check.png` | Diagnostik (KUNCI) |
| 30 | Grad-CAM Electronic failure mode (10 sample: 4 FP + 6 FN) → `gradcam_electronic_errors.png`. Ditemukan: Recyclable↔Organic confusion 104 kasus (belum diinvestigasi) | Diagnostik |
| 31 | Manual mislabel candidate verification (4 file, visual side-by-side) → `mislabel_verification_batch1.png` | Diagnostik/Cleaning |
| 32 | `is_standard_pattern()`, `looks_like_caption()` — filename pattern analysis → `suspicious` (1039 non-standard, semua di Electronic) | Cleaning |
| 32b | Debug regex match rate per class — konfirmasi 1039 non-standard BUKAN bug, murni karakteristik data (Electronic sumber campuran) | Cleaning (validasi) |
| 33 | Caption-like candidate visual inspection (10/333 sample) → `caption_visual_check_sample.png`, `caption_like_candidates.csv` — hasil: semua benar, tidak ada mislabel tambahan | Cleaning |
| 34 | **Relabel 4 file mislabel terkonfirmasi** (`RELABEL_MAP`) → overwrite `train_master_with_folds.csv` + backup `train_master_with_folds_PRE_RELABEL_BACKUP.csv` | Cleaning (KUNCI, SELESAI dieksekusi) |

### Artifact Files (disimpan ke `output_dir`)
| File | Dihasilkan di Cell | Isi |
|---|---|---|
| `train_test_overlap.csv` | 0b | 97 pasang exact-dup train-test (MD5) |
| `train_duplicate_groups.csv` | 0b | Exact duplicate groups dalam train |
| `near_duplicate_candidates.csv` | 0b | Near-dup candidates dari pattern `file(1).jpg` |
| `train_master_with_groups.csv` | 5 | Semua train file + label + `duplicate_group_id` + exclude flags. 26.527 baris |
| `train_master_with_folds.csv` | 14 (awal) → **direlabel di Cell 34 (4 Juli 2026)** | Sama seperti atas + kolom `fold` (0-4, -1=excluded). **VERSI TERKINI sudah bersih dari 4 mislabel.** |
| `train_master_with_folds_PRE_RELABEL_BACKUP.csv` | 34 | Backup sebelum relabel dieksekusi (untuk rollback kalau perlu) |
| `model_s9_baseline_cv{score}.pt` | 27 | Checkpoint terbaik per val_macro_f1. exp001 (data lama): `cv0.9823`. Checkpoint fold0-conv dataset lama cuma sempat simpan `cv0.9397` (epoch 1) sebelum sesi terputus. |
| `training_history_baseline.csv` | 27 | Full epoch history (train/val loss, F1 per kelas, LR, waktu) |
| `gradcam_shortcut_check.png` | 29 | 16-sample Grad-CAM (kontrol shortcut) |
| `gradcam_electronic_errors.png` | 30 | 10-sample Grad-CAM (FP/FN Electronic) |
| `mislabel_verification_batch1.png` | 31 | 4-sample visual verification mislabel |
| `caption_like_candidates.csv` | 33 | 333 file caption-like filename |
| `caption_visual_check_sample.png` | 33 | 10-sample visual check caption-like |
| `model_exp003_cleanbaseline_cv0.9815.pt` | 36b | Checkpoint terbaik exp003 (plain CE, data bersih) — epoch 14, val_macro_f1=0.9815 |
| `training_history_exp003_cleanbaseline.csv` | 36b | Full epoch history exp003 (15 epoch) |

### Key Variables (Global, dipakai lintas cell)
| Variabel | Tipe | Isi |
|---|---|---|
| `data_dir`, `train_dir`, `test_dir`, `submission_path`, `output_dir` | Path | Root direktori project, auto-detect Kaggle/Local (Cell 0) |
| `IS_KAGGLE` | bool | Flag environment |
| `class_folders` | list | `["0_Recyclable", "1_Electronic", "2_Organic"]` |
| `label_map` | dict | `{"0_Recyclable": 0, "1_Electronic": 1, "2_Organic": 2}` (Cell 1) |
| `overlap_df`, `groups_df`, `near_dup_df` | DataFrame | Hasil Cell 0b/0c — dasar exclude flags |
| `sampled_files` | dict | `{class: [file_paths]}`, seed=42, 500/kelas — HANYA EDA visual (Cell 6-11), TIDAK dipakai pipeline training |
| `train_master` | DataFrame | Master train file list + label + group/exclude flags, 26.527 baris (Cell 1-5) |
| `submission_df` | DataFrame | Test file list — source of truth urutan test (1458 baris, Cell 15) |
| `fold_df` | DataFrame | Load dari `train_master_with_folds.csv` (Cell 21) — **REBUILD ULANG setelah relabel Cell 34** |
| `train_df`, `val_df` | DataFrame | Subset `fold_df` untuk `FOLD_TO_VALIDATE` (Cell 21) |
| `train_dataset`, `val_dataset`, `test_dataset` | WasteDataset | PyTorch Dataset instances |
| `train_loader`, `val_loader`, `test_loader` | DataLoader | PyTorch DataLoader instances (val/test WAJIB `shuffle=False`) |
| `model` | timm model | ConvNeXt V2-Tiny, 3-class head (Cell 24) |
| `optimizer`, `scheduler`, `scaler`, `criterion` | torch objects | Setup training (Cell 25-26) |
| `history` | list of dict | Epoch-by-epoch metrics (Cell 27) |
| `val_labels_np`, `val_preds_np`, `val_filepaths_all` | np.array/list | Hasil inference val set — dipakai lintas Cell 28-31 untuk diagnostik |
| `cam` | GradCAM object | Instance Grad-CAM (Cell 29), reused di Cell 30 |
| `suspicious`, `caption_subset` | DataFrame | Hasil filename pattern analysis (Cell 32-33) |

### Key Functions/Classes
| Nama | Cell | Fungsi |
|---|---|---|
| `compute_file_hash(file_path, chunk_size=8192)` | 0b | MD5 hash, chunked read |
| `hash_all(root_dirs)` | 0b | Hash semua file di beberapa folder sekaligus |
| `sample_files_per_class(train_dir, class_folders, n_samples=500)` | 6 | Generate sample EDA per kelas, seeded |
| `compute_background_complexity(img, patch_size=30)` | 7 | Proxy kompleksitas background (4-corner variance) |
| `get_image_mode(filepath)` | 16 | Return `img.mode`, error-safe |
| `load_image_as_rgb(filepath)` | 17 | **Fungsi kunci pipeline.** Convert P/RGBA/L → RGB aman |
| `WasteDataset(df, transform, is_test)` | 19 | **Class kunci pipeline.** Custom PyTorch Dataset |
| `lr_lambda(current_step)` | 26 | Cosine+warmup LR schedule function |
| `get_image_dims(filepath)` | 28 | Return `(width, height)` — dasar deteksi icon 150×150 |
| `load_and_preprocess(filepath)` | 29 | Load+resize+normalize+tensor untuk Grad-CAM input |
| `is_standard_pattern(filename)` | 32 | Regex check nama file standar (`R_####`, `O_####`, dll) |
| `looks_like_caption(filename)` | 32 | Heuristik deteksi nama file berupa kalimat/caption |
| `resolve_filepath(filename)` | 33 | Cari filepath asli by loop `class_folders` (robust, tidak asumsi struktur folder label) |

### Constants (Locked — jangan ubah tanpa approval)
| Nama | Nilai | Cell | Alasan |
|---|---|---|---|
| `IMG_SIZE` | 224 | 20 | Baseline resolution, lock di `MODELING STRATEGY` |
| `N_SPLITS` | 5 | 12 | StratifiedGroupKFold, lock di `VALIDATION STRATEGY` |
| `SEED` | 42 | multiple | Reproducibility (sampling EDA, fold split, sampling diagnostik) |
| `BATCH_SIZE` | 64 | 21 | ConvNeXt V2-Tiny @ 224px, baseline utk 16GB VRAM. **2080 Ti (11GB) kemungkinan perlu turun sedikit (48) — belum divalidasi langsung.** |
| `NUM_WORKERS` | 4 | 21 | Sesuaikan kalau ada masalah multiprocessing |
| `FOLD_TO_VALIDATE` | 0 | 21 | Fold pertama dipakai untuk baseline sanity check |
| `NUM_EPOCHS` | 15 | 26 | Baseline sanity run |
| `WARMUP_STEPS_RATIO` | 0.075 | 26 | 7,5% dari total steps, dalam range locked 5-10% |
| `BACKBONE_LR` / `HEAD_LR` | 1e-5 / 1e-4 | 25 | Discriminative LR, backbone vs head |
| `WEIGHT_DECAY` | 0.05 | 25 | AdamW, standar untuk ConvNeXt-family |
| `threshold` (bg_variance) | 50 | 7 | Ambang batas plain-vs-complex background — kalibrasi manual, bukan angka baku |
| `RELABEL_MAP` | 4 file (lihat Cell 34) | 34 | Mapping relabel mislabel terkonfirmasi — SUDAH DIEKSEKUSI, jangan jalankan ulang tanpa cek (idempotent tapi redundant) |

### ⚠️ Catatan Penting Struktural
- **`train_master_with_folds.csv` SEKARANG PUNYA 2 VERSI SECARA KONSEPTUAL**: versi asli (dipakai exp001, CV=0.9823) dan versi ter-relabel (Cell 34, 4 Juli 2026). File fisik di `output_dir` SUDAH di-overwrite ke versi bersih — kalau butuh versi lama, pakai `train_master_with_folds_PRE_RELABEL_BACKUP.csv`.
- **Checkpoint path fallback di Cell 27 berubah** dari asumsi lama (`fold0-conv`) menjadi pencarian pola `/kaggle/input/datasets/*/phase0/*.pt` — sesuaikan pattern ini kalau nama dataset Kaggle kamu berubah lagi.
- **Setelah relabel Cell 34, Cell 21 (DataLoader) WAJIB di-rerun** sebelum training ulang — `train_df`/`val_df`/`train_loader`/`val_loader` yang sudah di-load ke memori masih pakai label lama sampai di-reload.
- Cell 27 (training loop) **exp001 CV=0.9823 tidak lagi representasi data terkini** — retrain dengan data bersih akan jadi **exp002**, dicatat terpisah di EXPERIMENT LOG SUMMARY begitu selesai.

### 🔴 TODO — Belum lengkap, isi menyusul
- [x] ~~Retrain exp003 (clean baseline, plain CE + data bersih) — **SELESAI 8 Juli 2026**, CV=0.9815~~
- [x] ~~Architecture comparison ConvNeXt V2-Tiny vs EfficientNetV2-S (exp004/4b/4c) — **CLOSED 8 Juli 2026**, ConvNeXt V2-Tiny menang, dihentikan atas dasar ROI~~
- [ ] Investigasi Recyclable↔Organic confusion (104 kasus dari Cell 30) — ditemukan tapi belum dianalisis, prioritas lebih rendah dari Electronic tapi worth di-follow-up
- [ ] Putuskan nasib checkpoint exp002 & exp003 (.pt) yang hilang dari sesi Kaggle — re-train ulang atau cari Notebook Version lama
- [ ] Putuskan arah lanjutan Fase 3 (berbasis ConvNeXt V2-Tiny): EMA vs Label Smoothing vs training full 5-fold K-fold ensemble Layer 1 — belum ada keputusan urutan