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
- Last updated          : 1 Juli 2026
- Updated by            : System / Post-EDA Update

## CURRENT STATUS
- Active phase          : FASE 1 EDA & Data Loading — EDA SELESAI, menunggu keputusan Ababil
- Last completed stage  : EDA Complete (Jeremy) — seluruh Checklist A, B, C selesai
- Next action           : Ababil memutuskan penanganan leakage & mislabel, lalu setup DataLoader final
- Blocker (if any)      : Keputusan Ababil terkait 97 file train-test overlap (Flag 3) dan cross-class mislabel (Flag 7) BELUM dibuat

## DATASET
- Train file    : `train/` folder — 26.527 images (rows), 3 classes (subfolders)
- Test file     : `test/` folder — 1.458 images (rows)
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

### 3. Integritas File (sample 1.500/26.527 train)
- Corrupt files: **0** — dataset bersih dari file rusak
- Mode gambar: mayoritas RGB. Ditemukan 2 file RGBA (alpha channel) hanya di Electronic — sinyal sumber data campuran
- Mode P (palette/indexed): tersebar tipis di semua kelas (13 Recyclable, 1 Electronic, 3 Organic)

### 4. Statistik Warna & Brightness per Kelas
| Kelas | Brightness (mean) | Pola Warna | Interpretasi |
|---|---|---|---|
| Recyclable | 181.8 (paling terang) | R≈G≈B, seimbang | Background putih/studio dominan |
| Electronic | 142.3 (paling gelap) | R≈G≈B, gelap | Objek metalik/hitam + background campuran |
| Organic | 145.0 | R dominan (167), B rendah (118) | Warna alami (hijau/coklat), sedikit biru |

**Risk:** Perbedaan brightness antar kelas signifikan. Kemungkinan didorong oleh background foto, bukan murni karakteristik objek — model bisa belajar shortcut "background putih = bukan Electronic" (lihat Flag 4).

### 5. Dimensi & Aspect Ratio
- **Electronic BIMODAL jelas:**
  - **68,8% persis 150×150 px** — format stock icon/database, tidak natural untuk foto asli
  - **31,2% foto natural** — width median 1028 px, max 8000 px, pola timestamp kamera HP
  - Nama file mengonfirmasi: `Washing_Machine_1.jpg` (kelompok 150×150) vs `IMG_20200328_214915.jpg` (kelompok besar)
- **Recyclable & Organic:** lebih bervariasi normal, tidak ada pola bimodal setajam Electronic

### 6. Visual Sample Grid — Background & Konteks Foto
- **Recyclable:** dominan white-background studio (e-commerce style), objek tunggal terpusat
- **Organic:** campuran — mayoritas white-background (roti, cabai, daging), subset natural/green-background (tanaman, ladang)
- **Electronic:** PALING CAMPURAN — ada white-background studio (baterai, microwave), ada real-world context (HP di meja kayu, tumpukan elektronik di lantai, keranjang puluhan komponen kecil)
- **Temuan tambahan:** Dua gambar Electronic menampilkan BANYAK objek kecil dalam satu frame (keranjang komponen, tumpukan router) — noisy visual signal, berbeda dengan pola "satu objek dominan" di kelas lain

### 7. Ukuran Objek Relatif (Foreground Ratio)
| Kelas | Mean | Median | 25% | 75% |
|---|---|---|---|---|
| Recyclable | 0,589 | 0,576 | 0,322 | 0,885 |
| Electronic | 0,718 | 0,776 | 0,542 | 0,915 |
| Organic | 0,761 | 0,871 | 0,587 | 0,952 |

- **Organic:** objek paling dominan mengisi frame (close-up/makro — stroberi, bunga, tomat)
- **Recyclable:** paling bervariasi — objek kadang kecil di tengah ruang kosong, sumber data paling beragam
- **Electronic:** di tengah, sedikit condong ke frame penuh

### 8. Duplikat Train-vs-Test (LEAKAGE — CRITICAL)
- **97 dari 1.458 file test (≈6,65%) exact-duplicate (MD5) dengan file di train**
- Breakdown: **95 Organic, 2 Electronic, 0 Recyclable**
- File pairing lengkap tersedia di notebook Jeremy
- Implikasi: ground truth untuk ~6,65% test set sudah diketahui dari label train — CV score bisa overestimate jika 97 file train TIDAK di-exclude

### 9. Duplikat Dalam-Train (Within-Class)
| Kelas | File Terlibat | Proporsi dari Total Kelas |
|---|---|---|
| Recyclable | 18 | 0,18% |
| Electronic | 60 | 1,51% |
| Organic | 44 | 0,35% |

- Electronic proporsi 4-8× lebih tinggi → jumlah unik efektif lebih kecil dari 3.961
- Relevan untuk CV split: duplikat yang terpecah antar-fold = leakage antar-fold → perlu dipastikan duplikat tetap dalam fold yang sama

### 10. Duplikat Cross-Class + MISLABEL (CRITICAL)
- **1 grup cross-class duplicate ditemukan:** `R_799.jpg` (Recyclable) dan `O_8873.jpg` (Organic) adalah file BYTE-IDENTIK
- **Gambar:** tas jinjing kain/jute bertuliskan "SAY NO TO PLASTIC"
- **Konfirmasi visual:** ini BUKAN sampah organik — seharusnya Recyclable (barang reusable non-elektronik)
- **Label `2_Organic` untuk `O_8873.jpg` adalah SALAH.** Ini bukan ambiguitas — murni mislabel
- **Implikasi:** ini kemungkinan "ujung gunung es" noise labeling — hanya terdeteksi karena exact-hash match secara kebetulan. Mislabel lain yang bukan duplikat tidak akan terdeteksi lewat metode hashing

---

**Hipotesis yang sudah divalidasi:**
- **H1:** Class imbalance pada Electronic (~15% dari total data) akan menurunkan F1-Score untuk kelas minoritas jika tidak ditangani (Weighted Loss / Oversampling). — **Status: TERKONFIRMASI — perlu action**
- **H2:** Variasi penamaan file pada kelas Electronic mengindikasikan variasi dimensi/aspect ratio gambar yang lebih tinggi dibanding kelas lain. — **Status: TERKONFIRMASI — Electronic bimodal (68,8% 150×150, 31,2% foto natural), dua sub-populasi berbeda sumber**

**Hipotesis baru (post-EDA):**
- **H3:** Perbedaan brightness/background antar kelas (Recyclable paling terang, Electronic paling gelap) adalah sinyal background, bukan objek — model berisiko belajar shortcut. — **Status: PERLU VALIDASI saat training (cek feature importance / Grad-CAM)**
- **H4:** Noise labeling mungkin lebih luas dari 1 kasus yang terdeteksi (tas kain) — perlu audit manual sampel acak. — **Status: BELUM TERVALIDASI (di luar scope hashing)**

---

**Risk flags dari Jeremy (FINAL — 7 flags):**

| Flag | Severity | Deskripsi | Action Required |
|---|---|---|---|
| **Flag 1** | HIGH | Macro F1 sangat sensitif terhadap performa Electronic (minoritas). Model tidak boleh bias ke kelas mayoritas | Weighted Loss / Oversampling / Class-Balanced Sampling |
| **Flag 2** | HIGH | Hanya 3 slot submission. Dilarang trial-and-error atau LB probing sembarangan | CV harus sangat reliable (StratifiedKFold), semua eksperimen offline |
| **Flag 3** | **CRITICAL** | 97/1458 test files (6,65%) exact-duplicate (MD5) dengan train. 95 Organic, 2 Electronic, 0 Recyclable | **BLOCKER:** Ababil harus memutuskan — exclude 97 file train dari training? Biarkan tapi catat sebagai caveat CV overestimate? |
| **Flag 4** | MEDIUM-HIGH | Background tidak seragam antar kelas. Recyclable dominan white-background, Electronic paling campuran (white studio + real-world context) | Model bisa belajar shortcut dari background. Perlu Grad-CAM check setelah baseline training |
| **Flag 5** | MEDIUM-HIGH | Electronic bimodal: 68,8% 150×150 px (stock icon), 31,2% foto natural (width median 1028). Dua sub-populasi berbeda sumber | Resize strategy perlu hati-hati. 150×150 di-upscale ke input size model berisiko degradasi kualitas. Pertimbangkan input size ≥ 224×224 dengan padding/lanczos |
| **Flag 6** | MEDIUM | Within-class duplicates: Recyclable 18 (0,18%), Electronic 60 (1,51%), Organic 44 (0,35%). Electronic proporsi tertinggi | Pastikan duplikat tidak terpecah antar-fold (CV leakage). Pertimbangkan exclude duplikat untuk dapat jumlah unik sebenarnya |
| **Flag 7** | **CRITICAL** | Cross-class duplicate + MISLABEL: `R_799.jpg` (Recyclable) = `O_8873.jpg` (Organic) — tas kain "Say No To Plastic". Label Organic salah | **BLOCKER:** Ababil harus memutuskan — exclude `O_8873.jpg` dari training? Relabel jadi Recyclable? Atau exclude kedua file? |

---

## BUSINESS BRIEF STATUS (Vierico)
- Checkpoint B1 (Problem Brief)  : DONE
- Checkpoint B2 (EDA Commentary) : BELUM — menunggu Ababil kirim ringkasan EDA final
- Checkpoint B3 (Strategy Review): BELUM
- Checkpoint B4 (Error Cost)     : BELUM
- Checkpoint B5 (Explainability) : BELUM
- Checkpoint B6 (Exec Summary)   : BELUM

**Active veto dari Vierico:**
- NONE

**Business constraints yang sudah dikonfirmasi:**
- Constraint 1: **HANYA 3 SUBMISSION** yang diizinkan selama kompetisi. Strategi ensemble dan pseudo-labeling harus dimatangkan secara offline (CV-based) sebelum submit.
- Constraint 2: Dilarang menggunakan metadata/data eksternal. Hanya informasi visual (pixel) dari gambar yang boleh dipakai.
- Constraint 3: Wajib mendokumentasikan backbone model pre-trained (EfficientNet, ResNet, ConvNeXt, ViT) di laporan akhir.
- Constraint 4: Urutan `id` di `submission.csv` tidak boleh diubah (harus sesuai urutan angka 1 sampai 1458).

## FEATURE SET (Ababil)
- Ababil stage saat ini : Stage 1 (Data Pipeline & Augmentation Setup) — MENUNGGU KEPUTUSAN
- FE status             : IN PROGRESS — DataLoader dasar sudah siap, perlu disesuaikan dengan keputusan leakage/mislabel
- Leakage check         : **FAIL — 97 file overlap confirmed via exact MD5 hash antara train dan test (Flag 3). Cross-class mislabel confirmed (Flag 7). Perlu action dari Ababil sebelum training dimulai.**
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

## VALIDATION STRATEGY (locked after Stage 7)
- CV method     : StratifiedKFold (Wajib stratified karena class imbalance)
- n_folds       : 5
- Seed          : 42
- Group column  : N/A
- Locked        : NO

**Catatan tambahan untuk CV (post-EDA):**
- Duplikat within-class (Flag 6) harus dipastikan tidak terpecah antar-fold → gunakan group-aware split atau exclude duplikat
- 97 file train-test overlap (Flag 3) jika di-exclude → ukuran train berkurang dari 26.527 menjadi 26.430
- Cross-class mislabel (Flag 7): minimal 1 file `O_8873.jpg` harus di-exclude atau direlabel sebelum split

## PENDING ACTIONS
Isi ini setiap sesi sebelum tutup chat.

### CRITICAL — BLOCKER UNTUK TRAINING (Ababil)
- [ ] **Ababil** memutuskan penanganan Flag 3: Apakah 97 file train yang exact-duplicate dengan test di-exclude dari training set? (Rekomendasi Jeremy: EXCLUDE, untuk menjaga validitas CV score internal)
- [ ] **Ababil** memutuskan penanganan Flag 7: Apakah file `O_8873.jpg` (mislabel, seharusnya Recyclable) di-exclude atau direlabel? (Rekomendasi Jeremy: EXCLUDE file `O_8873.jpg`, pertahankan `R_799.jpg`)
- [ ] **Ababil** memutuskan penanganan Flag 6: Apakah within-class duplicates di-exclude, atau cukup dipastikan tidak terpecah antar-fold?

### HIGH — NEXT STEPS SETELAH KEPUTUSAN
- [ ] **Ababil** membuat Stratified K-Fold splitter dengan group constraint (duplikat dalam fold sama) dan DataLoader final — sudah memperhitungkan file yang di-exclude
- [ ] **Ababil** menentukan input size model dengan mempertimbangkan Flag 5 (Electronic 68,8% adalah 150×150 px) — rekomendasi: minimum 224×224 dengan padding/lanczos, atau 150×150 langsung dengan pertimbangan info loss
- [ ] **Ababil** mengirim ringkasan EDA final ke Vierico untuk Checkpoint B2 (EDA Commentary)
- [ ] **Vierico** merancang "3-Submission Masterplan": Kapan slot 1 (baseline), slot 2 (ensemble/pseudo-label), dan slot 3 (final tweak) akan digunakan
- [ ] **Jeremy** (jika diperlukan): Audit manual sampel acak untuk estimasi noise labeling di luar exact-hash (H4) — prioritas LOW, bisa ditunda setelah baseline

## CONTEXT RESET PROTOCOL
Saat ganti akun Claude (limit habis), lakukan urutan ini:

**Untuk Ababil:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_ababil.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari Stage 1 (Data Pipeline) — keputusan leakage/mislabel belum dibuat. Semua context ada di atas."

**Untuk Jeremy:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_jeremy.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari E8 (EDA Complete). Semua context ada di atas."

**Untuk Vierico:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_vierico.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari checkpoint B2 (EDA Commentary). Semua context ada di atas."

*Jangan skip langkah ini. Tanpa project_state, Claude mulai dari nol.*

## CATATAN BEBAS
Gunakan bagian ini untuk hal-hal yang tidak masuk kategori di atas:

- **[1 Juli 2026] [System]:** Aturan kompetisi sangat ketat terkait submission (maks 3x). Pastikan CV score sangat reliable (Stratified K-Fold) karena kita tidak bisa mengandalkan LB probing.
- **[1 Juli 2026] [System]:** Kelas Electronic hanya 3.961 gambar (~15%). Metrik Macro F1-Score akan sangat menghukum model yang lemah di kelas ini. Pertimbangkan Weighted CrossEntropyLoss atau Class-Balanced Sampling sejak awal.
- **[1 Juli 2026] [Jeremy — EDA Complete]:**
  - Exact-hash duplicate check (train vs test) selesai. 97 file overlap. File pairing lengkap tersedia di variabel `train_test_duplicates` (dict) di notebook eksplorasi.
  - Cross-class duplicate check selesai. 1 kasus mislabel terkonfirmasi: `R_799.jpg` = `O_8873.jpg` (tas kain — seharusnya Recyclable).
  - Within-class duplicate check selesai: 61 grup (122 file), Electronic proporsi tertinggi (1,51%).
  - Semua statistik (brightness, warna, dimensi, aspect ratio, foreground ratio) sudah dihitung dan tersedia di notebook eksplorasi.
  - File RGBA di Electronic: 2 file dengan alpha channel — perlu di-handle saat loading (konversi ke RGB).
  - File mode P (palette/indexed): total 17 file — perlu di-handle saat loading (konversi ke RGB).