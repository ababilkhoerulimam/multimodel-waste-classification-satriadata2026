Analisis Strategis: **Kompetisi ini akan dimenangkan atau dikalahkan oleh satu variabel tunggal: performa kelas Electronic.** Bukti dari dataset paper publik (Yasin & Koklu, 2023/2024) menunjukkan bahwa sub-problem Organic-vs-Recyclable sudah mendekati saturasi bahkan dengan metode usang sekalipun (frozen InceptionV3 + SVM mencapai F1 ~96,6% pada 2 kelas ini). Electronic tidak punya preseden ilmiah sama sekali di paper, bersifat bimodal ekstrem (68,8% stock-icon 150×150 vs 31,2% foto natural), dan merupakan kelas minoritas (14,9%) — kombinasi ini menjadikannya satu-satunya bottleneck nyata terhadap Macro F1. Strategi final adalah **fine-tuning end-to-end backbone CNN modern skala Tiny (ConvNeXt V2-Tiny sebagai anchor)**, bukan replikasi feature-extraction ala paper dan bukan model raksasa yang belum teruji terhadap hardware tim, dengan seluruh keputusan desain diarahkan secara eksplisit untuk menaikkan F1 Electronic tanpa merusak F1 dua kelas lain yang sudah kuat. Validasi StratifiedGroupKFold yang sudah di-lock dipertahankan tanpa perubahan, dan anggaran 3 submission diperlakukan sebagai constraint keras yang menuntut CV 100% dipercaya.

---

# Revisi Strategi Modeling BDC Satria Data 2026: Waste Classification

**Tanggal:** 2 Juli 2026
**Status dokumen:** Revisi final — sintesis dari dua strategi awal (Kimi/generik-luas vs Sonnet/presisi-Project-State), dikoreksi pada titik risiko model utama.

Dokumen ini merevisi rekomendasi sebelumnya dengan mengoreksi dua kesalahan mendasar: (1) target performa yang overclaim tanpa dasar empiris, dan (2) pemilihan model utama yang tidak mempertimbangkan ketidakpastian hardware dan ketatnya anggaran submission. Seluruh rekomendasi di bawah ditautkan langsung ke bukti spesifik dari Project State — bukan katalog teknik CV generik yang "diarahkan ke semua kelas" tanpa penajaman.

## 1. Implikasi Dataset: Apa yang Benar-benar Diketahui

### 1.1. Dataset Kompetisi = Turunan Langsung Dataset Paper, Kecuali Electronic

Bukti dari Project State jauh lebih kuat daripada sekadar "mirip secara statistik" — ini adalah bukti populasi sumber yang sama. Selisih brightness Organic antara kompetisi (145,05) dan sampel paper (140,58) berada di bawah 5 poin; Recyclable serupa (181,76 vs 180,52). Foreground ratio kedua kelas juga berdekatan (0,761/0,589 kompetisi vs 0,786/0,600 paper). Bukti paling definitif: pasangan cross-class duplicate `O_8873.jpg`/`R_799.jpg` di kompetisi adalah file **byte-identik** (MD5 sama) dengan pasangan mislabel yang juga ditemukan di dataset paper publik. Probabilitas dua sumber data independen kebetulan memiliki file identik dengan mislabel identik pada pasangan yang sama mendekati nol.

Konsekuensinya jelas: dataset Organic dan Recyclable pada kompetisi ini **bukan** kumpulan foto baru yang dikurasi khusus panitia — ia adalah subset atau turunan dari lineage dataset publik yang sama. **Electronic tidak punya padanan di paper sama sekali.** Ini bukan detail kecil; ini adalah fakta yang harus mengarahkan seluruh alokasi waktu riset tim.

### 1.2. Mengapa Paper Bukan Baseline Angka, Tapi Tetap Berguna sebagai Sanity-Check

Angka 96,3% akurasi dari kombinasi InceptionV3+SVM di paper **tidak bisa dipakai sebagai target skor kompetisi**, karena tiga alasan konkret: metrik berbeda (akurasi vs Macro F1 3-kelas dengan imbalance), tidak ada kelas Electronic di paper yang justru menjadi penentu skor kompetisi, dan metodologi paper (frozen feature extraction + classifier klasik) bukan pendekatan kompetitif untuk deep learning modern. Namun paper tetap punya nilai strategis besar: ia membuktikan bahwa **problem Organic-vs-Recyclable sudah "mudah" secara ilmiah**, bahkan tanpa fine-tuning sama sekali. Jika model modern kompetisi tidak mencapai F1 tinggi (>95%) pada sub-masalah dua kelas ini, itu adalah sinyal bug pipeline — bukan sinyal bahwa problem ini sulit.

Implikasi paling penting dari fakta ini: **tim tidak perlu menghabiskan waktu riset signifikan untuk Organic dan Recyclable.** Seluruh energi eksperimen harus dialihkan ke Electronic.

### 1.3. Risiko Data yang Sudah Closed — Jangan Dibuka Kembali

Dua isu data integrity sudah closed dan tidak boleh direvisi ulang tanpa alasan baru: leakage train-test (97/1.458 file, MD5-exact, sudah dimitigasi via exclude-untuk-CV dan hash-override-untuk-final), dan mislabel `O_8873.jpg`=`R_799.jpg` (exclude dari training). Within-class duplicate (62 grup, tertinggi di Electronic 1,51%) sudah ditangani via StratifiedGroupKFold dan berstatus locked. Satu isu yang **masih terbuka** dan harus tetap begitu: kemungkinan overlap MD5 antara dataset paper dan test set kompetisi — ini di-hold menunggu fatwa Vierico, dan hasil MD5-matching **tidak boleh** memengaruhi keputusan model atau submission sampai fatwa keluar secara eksplisit.

## 2. Strategi Modeling: Kenapa Bukan Replikasi Paper, dan Kenapa Bukan Model Raksasa

### 2.1. Replikasi Paper sebagai Floor, Bukan Ceiling

Pendekatan feature-extraction beku (frozen InceptionV3 + SVM/KNN) punya satu kegunaan legitim dalam strategi ini: sebagai **sanity-check kualitatif** untuk sub-problem Organic/Recyclable, bukan sebagai model kompetisi. Jika hasil fine-tuning model modern tim tidak jauh melampaui ~96% pada dua kelas ini, itu tanda ada yang salah di pipeline, bukan tanda bahwa target sudah tercapai. Replikasi penuh paper sebagai strategi utama akan berisiko menghasilkan skor Macro F1 rendah, karena arsitektur dan metodologi paper sama sekali tidak dirancang menangani kelas minoritas heterogen seperti Electronic.

### 2.2. Kenapa Model Raksasa (ConvNeXt V2-Base, EfficientNetV2-M) Ditolak sebagai Anchor

Ada godaan kuat untuk langsung memilih model berkapasitas tertinggi yang tersedia — semakin besar model, semakin tinggi representasi yang bisa ditangkap. Godaan ini harus ditolak untuk kompetisi ini secara spesifik, karena dua constraint keras yang membatasi ruang eksperimen: **spesifikasi hardware training tim tidak disebutkan eksplisit di Project State**, dan **hanya ada 3 submission untuk seluruh kompetisi**. Kombinasi model besar + ensemble besar + budget submission ketat adalah resep pemborosan waktu eksperimen jika ternyata hardware tim terbatas — setiap epoch yang gagal karena OOM atau terlalu lambat untuk diiterasi adalah waktu riset yang tidak bisa dikembalikan, jauh lebih mahal dibanding potensi kenaikan F1 marjinal dari model yang lebih besar.

Target performa seperti "Macro F1 > 96%" yang kadang muncul dalam diskusi strategi juga harus ditolak sebagai klaim resmi — ini adalah overclaim tanpa dasar empiris untuk kompetisi 3-kelas dengan minoritas 14,9% dan sumber data Electronic yang belum pernah dibuktikan performanya di literatur manapun. Target performa yang tidak dikalibrasi terhadap ketidakpastian nyata (hardware belum terkonfirmasi, fatwa Vierico belum keluar, noise label residual belum diaudit) berisiko menyesatkan ekspektasi tim dan mendorong keputusan desain yang terlalu agresif di awal.

### 2.3. Fine-Tuning End-to-End Skala Tiny sebagai Jalan Tengah yang Disiplin

Jalan yang dipilih adalah **fine-tuning penuh backbone CNN modern skala Tiny**, bukan feature-extraction beku (terlalu lemah untuk Electronic) dan bukan model skala Base/M ke atas (terlalu berisiko untuk hardware-tak-terkonfirmasi + budget submission ketat). Fine-tuning penuh memungkinkan filter konvolusi awal beradaptasi dengan domain spesifik sampah — misalnya tekstur sirkuit elektronik atau permukaan logam — yang tidak mungkin didapat dari fitur ImageNet generik yang dibekukan.

## 3. Electronic: Satu-satunya Variabel Penentu Leaderboard

### 3.1. Dua Lapis Masalah yang Harus Ditangani Terpisah

Electronic bukan sekadar "kelas minoritas" — ia adalah dua masalah berbeda yang bercampur dalam satu label:

**Lapis pertama — imbalance klasik.** Electronic hanya 14,9% dari total data, jauh di bawah Organic (47,4%) dan Recyclable (37,7%). Ini ditangani dengan pendekatan imbalance standar: Class-Balanced Loss berbasis effective number of samples (bukan invers-frekuensi naif yang cenderung tidak stabil untuk deep net), dikombinasikan weighted-random sampler.

**Lapis kedua — domain-shift internal.** 68,8% gambar Electronic berupa stock-icon persegi 150×150, sementara 31,2% sisanya adalah foto natural resolusi tinggi. Ini bukan variasi biasa dalam satu distribusi — ini adalah dua distribusi visual yang berbeda dipaksa masuk ke satu label. Model yang hanya dioptimasi untuk imbalance tanpa menangani bimodalitas ini berisiko overfit ke subpopulasi mayoritas (stock-icon) dan gagal total di foto natural, atau sebaliknya.

Kedua lapis ini **harus ditangani terpisah** — memperbaiki imbalance saja tidak menyelesaikan bimodalitas, dan sebaliknya. Solusi: resize aspect-ratio-preserving + padding (bukan stretch atau naive upscale, yang akan merusak proporsi objek pada subset 150×150), resolusi input dinaikkan ke ≥288px khusus untuk menangkap detail foto natural (kondisional pada konfirmasi hardware), dan evaluasi CV yang wajib dipecah per-subpopulasi (icon vs foto) agar model selection tidak "buta" terhadap salah satu sisi.

### 3.2. Risiko Shortcut Learning

Ada indikasi kuat dari EDA granular bahwa background putih berkorelasi dengan Recyclable dan crop persegi berkorelasi dengan Electronic — korelasi ini berasal dari cara data dikumpulkan (studio vs foto natural), bukan sinyal objek yang pasti muncul konsisten di test set. Jika model belajar shortcut ini alih-alih fitur objek asli, ia akan tampak bagus di CV tapi rapuh terhadap variasi nyata. Mitigasi: jalankan **Grad-CAM check** sedini mungkin setelah baseline pertama convergen, sebagai gerbang keputusan — jika attention map terkonsentrasi pada background alih-alih objek, itu konfirmasi shortcut, dan CutMix diaktifkan untuk seluruh eksperimen berikutnya. Jika tidak, compute dialihkan ke tuning Class-Balanced Loss dan resolusi.

## 4. Pemilihan Arsitektur: ConvNeXt V2-Tiny sebagai Anchor

### 4.1. Kenapa CNN, Bukan Transformer, yang Jadi Taruhan Pertama

Model Transformer modern skala Tiny (Swin V2 Tiny, EVA-02 Tiny) memiliki representasi semantik yang menarik secara teoretis dan berpotensi unggul memisahkan subpopulasi icon-vs-foto-vs-studio karena representasinya lebih kontekstual. Namun arsitektur ini secara umum **lebih sensitif terhadap hyperparameter dan lebih rentan gagal konvergen stabil** dibanding CNN modern, terutama pada budget eksperimen yang ketat. Dengan hardware tim yang belum terkonfirmasi dan hanya 3 submission untuk seluruh kompetisi, risiko menaruh model paling "finicky" sebagai taruhan pertama tidak sepadan dengan potensi keunggulan marjinalnya.

**ConvNeXt V2-Tiny** dipilih sebagai anchor karena kompetitif dengan Swin V2 pada skala Tiny namun jauh lebih toleran terhadap resep training yang belum sepenuhnya di-tuning — resep fine-tuning yang paling *forgiving* dan cepat konvergen lebih bernilai dalam kondisi ketidakpastian ini daripada potensi keunggulan arsitektur yang lebih rewel.

### 4.2. Struktur Model: Anchor, Partner Ensemble, dan Kandidat Sekunder

| Peran | Model | Justifikasi |
|---|---|---|
| **Anchor (submission 1–2)** | ConvNeXt V2-Tiny, ImageNet-pretrained, fine-tune penuh | Paling forgiving untuk hardware tak-terkonfirmasi + budget ketat; kompetitif dengan Transformer pada skala Tiny |
| **Partner ensemble (submission 3, kondisional)** | EfficientNetV2-S | Inductive bias berbeda (compound scaling vs modernized-conv) → diversitas nyata untuk ensemble; tetap ringan secara compute |
| **Kandidat sekunder (bukan anchor)** | Swin V2 Tiny / EVA-02 Tiny | Dipertimbangkan **hanya jika** hardware terkonfirmasi memadai dan baseline ConvNeXt sudah solid; dipromosikan ke ensemble hanya jika unggul jelas pada F1 Electronic di CV yang sama |
| **Bukan model kompetisi** | Frozen InceptionV3 + SVM (paper) | Hanya sanity-check floor kualitatif untuk Organic/Recyclable |

Model besar (ConvNeXt V2-Base, EfficientNetV2-M) dan ensemble 3–5 model ditolak sebagai rekomendasi utama — bukan karena tidak berpotensi kuat secara teori, tapi karena kombinasinya dengan hardware-tak-terkonfirmasi dan budget-3-submission menciptakan risiko pemborosan waktu eksperimen yang tidak proporsional terhadap potensi kenaikan skor.

## 5. Resep Fine-Tuning dan Augmentasi

| Komponen | Keputusan | Alasan |
|---|---|---|
| Optimizer | AdamW, weight decay decoupled | Standar modern, stabil untuk fine-tuning backbone pretrained |
| Scheduler | Cosine decay + linear warmup (5–10% steps) | Konvergensi halus, mengurangi risiko instabilitas awal training |
| Augmentasi dasar | RandAugment ringan–sedang + horizontal flip + color jitter terbatas | Brightness/warna adalah sinyal diskriminatif untuk Organic/Recyclable — augmentasi warna agresif berisiko merusak sinyal ini |
| Loss | Class-Balanced Loss (effective number of samples) | Lebih stabil untuk deep net dibanding invers-frekuensi naif |
| Oversampling | Weighted-random sampler + augmentasi (bukan duplikasi mentah) | Duplikasi mentah akan memperparah within-class duplicate Electronic yang sudah 1,51% |
| CutMix | Kondisional — hanya jika Grad-CAM konfirmasi shortcut background | Jangan aktifkan berdasarkan asumsi; aktifkan berdasarkan bukti diagnostik |
| Mixup | Diuji terpisah setelah baseline, evaluasi dampak per-kelas | Drop jika F1 Electronic turun dibanding baseline+CutMix |
| Label Smoothing | ε=0,1, mulai dari fine-tuning kedua | Membantu kalibrasi terhadap noise label residual yang belum sepenuhnya teraudit |
| EMA | Digunakan pada semua run setelah baseline | Stabilisasi bobot, biaya rendah |
| Resolusi input | Baseline 224px; naikkan ke ≥288px untuk eksperimen Electronic **setelah** hardware terkonfirmasi | Resolusi lebih tinggi krusial untuk subset foto natural Electronic, tapi menambah beban compute yang belum bisa dipastikan aman |
| TTA | Flip + multi-crop ringan, submission final saja | Kenaikan skor murah-risiko-rendah |
| Pseudo-labeling | Tidak digunakan di fase awal; hanya dipertimbangkan di akhir pada 1.361 file test non-overlap | Risiko tinggi jika confidence CV tidak sangat tinggi |

## 6. Strategi Validasi

Skema validasi **StratifiedGroupKFold, 5-fold, seed 42, group by duplicate_group_id tidak diubah** — sudah locked dan divalidasi ulang tanpa ditemukan celah baru. Yang perlu ditambahkan bukan mengganti skema ini, melainkan menajamkan kriteria *model selection* di atasnya:

- Model selection berdasarkan Macro F1 per-fold CV, dengan **F1 Electronic sebagai kriteria pembeda utama** ketika macro-F1 rata-rata antar-kandidat setara — karena Electronic adalah variabel yang benar-benar menentukan posisi leaderboard, bukan rata-rata makro yang bisa menyembunyikan kelemahan di kelas minoritas.
- Checkpoint disimpan berdasarkan bobot EMA terbaik menurut macro-F1 validasi, bukan loss atau accuracy.
- Evaluasi tambahan wajib: F1 terpisah per-subpopulasi Electronic (icon 150×150 vs foto natural), agar model tidak lolos seleksi hanya karena kuat di satu subpopulasi.
- Anggaran 3 submission diperlakukan sebagai constraint keras: CV harus menjadi satu-satunya sumber kepercayaan, tidak ada ruang untuk LB-probing. Sisakan minimal 1 submission cadangan sampai mendekati deadline, untuk berjaga-jaga terhadap CV-LB gap yang tidak terduga.

## 7. Rencana Eksperimen dan Alokasi Submission

Urutan diprioritaskan agar keputusan diagnostik (Grad-CAM) dijalankan sedini mungkin, karena hasilnya mengubah desain augmentasi untuk seluruh eksperimen berikutnya.

| # | Eksperimen | Prioritas | Catatan |
|---|---|---|---|
| 1 | Finalisasi DataLoader + StratifiedGroupKFold | Tertinggi (in progress) | Fondasi — CV tidak bisa dipercaya tanpa ini |
| 2 | Baseline ConvNeXt V2-Tiny, fine-tune polos, resolusi 224 | Tertinggi | Titik referensi bersih, tanpa trik eksotis |
| 3 | Grad-CAM shortcut check pasca-baseline | Tertinggi (segera setelah #2) | Menentukan apakah CutMix diaktifkan |
| 4 | Class-Balanced Loss untuk Electronic | Tinggi | Area dampak terbesar terhadap Macro F1 |
| 5 | CutMix (kondisional pada hasil #3) | Tinggi (kondisional) | Hanya jika shortcut terkonfirmasi |
| 6 | Resize/resolusi ≥288px untuk Electronic bimodal | Tinggi | Butuh konfirmasi hardware dulu |
| 7 | EMA + Label Smoothing | Tinggi | Biaya rendah, stabilisasi konsisten |
| 8 | Mixup (uji-banding vs CutMix) | Sedang | Drop jika F1 Electronic turun |
| 9 | Kandidat sekunder (Swin V2 Tiny / EVA-02 Tiny) | Sedang | Hanya jika hardware memadai & baseline solid |
| 10 | Ensemble ConvNeXt V2-Tiny + EfficientNetV2-S | Sedang (dekat akhir) | Submission ke-3, kondisional CV mendukung |
| 11 | TTA pada submission final | Sedang | Biaya rendah, risiko rendah |
| 12 | Pseudo-labeling (opsional) | Rendah | Hanya jika waktu & confidence sangat tinggi |

### Alokasi 3 Submission

1. **Submission 1**: ConvNeXt V2-Tiny anchor terbaik dari CV (dengan Class-Balanced Loss, CutMix-kondisional, EMA, Label Smoothing) — titik referensi kompetitif pertama sekaligus validasi bahwa CV selaras dengan LB.
2. **Submission 2**: Iterasi terbaik setelah tuning resolusi Electronic dan/atau kandidat sekunder terbukti unggul di CV — jangan gunakan hanya untuk eksperimen coba-coba yang belum divalidasi CV.
3. **Submission 3**: Ensemble ConvNeXt V2-Tiny + EfficientNetV2-S dengan TTA, kondisional CV ensemble > CV model tunggal terbaik. Jika ensemble tidak terbukti unggul di CV, submit model tunggal terbaik + TTA saja — jangan spekulasi di submission terakhir.

## 8. Final Verdict dan Rekomendasi Akhir

### 8.1. Model Paling Direkomendasikan

**ConvNeXt V2-Tiny** adalah model anchor yang direkomendasikan, bukan varian Base atau EfficientNetV2-M. Alasannya bukan karena kapasitas representasinya lebih rendah tidak masalah, tapi karena kombinasi hardware-tak-terkonfirmasi dan budget-3-submission menuntut resep training paling forgiving dan cepat konvergen. **EfficientNetV2-S** direkomendasikan sebagai partner ensemble di submission ketiga karena inductive bias yang berbeda memberi diversitas nyata dengan biaya compute rendah.

### 8.2. Strategi Training Paling Masuk Akal

1. **Pre-trained weights**: ImageNet-pretrained standar (bukan otomatis ImageNet-21K jika belum ada konfirmasi kapasitas compute untuk fine-tuning penuh model yang lebih berat).
2. **Fine-tuning**: end-to-end penuh, bukan feature-extraction beku.
3. **Augmentasi**: RandAugment ringan–sedang, CutMix kondisional pada bukti Grad-CAM, Mixup sebagai uji-banding terpisah — bukan ditumpuk semua sekaligus tanpa validasi dampak per-kelas.
4. **Loss function**: Class-Balanced Loss (effective number of samples), bukan Focal Loss agresif tanpa dasar tuning yang jelas.
5. **Regularisasi**: Label Smoothing ε=0,1, dimulai dari fine-tuning kedua.
6. **Optimasi**: AdamW + Cosine Annealing + Linear Warmup.
7. **Stabilisasi**: EMA pada semua run setelah baseline.

### 8.3. Prioritas Eksperimen dari Tahap Awal hingga Final

| Fase | Fokus | Gerbang Keputusan |
|---|---|---|
| **1. Fondasi** | DataLoader final + StratifiedGroupKFold + baseline ConvNeXt V2-Tiny bersih | Baseline harus convergen sebelum lanjut |
| **2. Diagnostik** | Grad-CAM shortcut check | Menentukan aktivasi CutMix untuk seluruh fase berikutnya |
| **3. Optimasi Electronic** | Class-Balanced Loss, CutMix-kondisional, resize/resolusi ≥288px | Wajib konfirmasi hardware sebelum menaikkan resolusi |
| **4. Stabilisasi & sekunder** | EMA, Label Smoothing, Mixup uji-banding, kandidat sekunder (Swin V2/EVA-02) | Kandidat sekunder hanya lanjut jika baseline solid |
| **5. Finalisasi & submission** | Ensemble kondisional, TTA, alokasi 3 submission | Ensemble hanya jika CV membuktikan unggul dari model tunggal |

### 8.4. Risiko Terbesar yang Mungkin Muncul

1. **Class imbalance + domain-shift internal Electronic menekan Macro F1.** Mitigasi: Class-Balanced Loss + oversampling-dengan-augmentasi, model selection berbasis F1 Electronic, evaluasi CV per-subpopulasi.
2. **Shortcut learning pada background/shape.** Mitigasi: Grad-CAM check dini sebagai gerbang keputusan, CutMix kondisional.
3. **Hardware tim belum terkonfirmasi.** Mitigasi: konfirmasi segera sebelum commit ke resolusi ≥288px atau ensemble; jika hardware ternyata terbatas, turunkan ke model tunggal ConvNeXt V2-Tiny + TTA saja tanpa ensemble.
4. **Budget submission (3x) tidak cukup jika CV-LB gap tak terduga.** Mitigasi: CV sebagai satu-satunya sumber kepercayaan, sisakan minimal 1 submission cadangan sampai mendekati deadline.
5. **Benchmark leakage via paper overlap.** Mitigasi: tetap hold hasil MD5 paper-vs-test sampai fatwa Vierico eksplisit; tidak boleh memengaruhi model atau submission sebelum itu.
6. **Noise label residual di luar 1 kasus terdeteksi (H4).** Mitigasi: Label Smoothing sudah masuk resep training; audit manual sampel tetap prioritas rendah, dijalankan jika waktu memungkinkan.
7. **Model utama gagal konvergen stabil.** Risiko ini sudah diturunkan secara sengaja dengan memilih ConvNeXt V2-Tiny (CNN forgiving) sebagai anchor, bukan Transformer yang lebih sensitif hyperparameter.

### 8.5. Tingkat Keyakinan dan Alasannya

**Tingkat keyakinan: 75%** — dikalibrasi secara eksplisit, bukan klaim "tinggi" tanpa dasar.

Keyakinan tinggi pada bagian yang didukung bukti langsung dan ganda dari Project State: penanganan leakage dan mislabel yang sudah closed, arah fokus total ke Electronic (tervalidasi silang oleh statistik EDA internal dan bukti dataset paper), serta pemilihan ConvNeXt V2-Tiny sebagai anchor yang paling forgiving untuk kondisi ketidakpastian saat ini. Keyakinan tidak mencapai 90%+ karena tiga ketidakpastian nyata yang masih di luar kendali analisis ini: (1) spesifikasi hardware training riil tim belum dikonfirmasi, sehingga keputusan resolusi ≥288px dan kelayakan ensemble bisa perlu direvisi; (2) fatwa Vierico terkait status paper-overlap belum keluar; dan (3) sejauh mana noise label residual (H4) yang tak terdeteksi metode exact-hash bisa memengaruhi training tidak dapat dipastikan tanpa audit manual yang belum dijalankan.

---

## Immediate Next Actions

- [ ] **Ababil**: selesaikan DataLoader final (StratifiedGroupKFold, exclusion list 98 file, RGB conversion RGBA/Palette, resize aspect-ratio-preserving).
- [ ] **Ababil**: mulai baseline training ConvNeXt V2-Tiny (resolusi 224, tanpa CutMix/Mixup, tanpa Class-Balanced Loss dulu) sebagai titik referensi bersih.
- [ ] **Ababil/Jeremy**: jalankan Grad-CAM check segera setelah baseline convergen — hasilnya menentukan apakah CutMix masuk roadmap.
- [ ] **Tim (di luar Ababil, segera)**: konfirmasi spesifikasi hardware training riil (GPU, VRAM) — memblokir keputusan resolusi Electronic dan kelayakan ensemble.
- [ ] **Jeremy**: lanjutkan MD5-matching paper-vs-test, tapi hasilnya tetap **hold**, tidak boleh memengaruhi model/submission sampai fatwa Vierico keluar.
- [ ] **Vierico**: terima ringkasan EDA final dari Ababil untuk Checkpoint B2; siapkan fatwa untuk isu paper-overlap begitu hasil MD5 Jeremy tersedia.
