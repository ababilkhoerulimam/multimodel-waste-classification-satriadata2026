# BLUEPRINT STRATEGI — BDC SATRIA DATA 2026 (Problem 1: Waste Classification)
**Sintesis: Paper Analysis (Yasin & Koklu, 2024) × Competition Project State**
Disusun: 2 Juli 2026 | Role: Principal ML Engineer Review

---

## 1. EXECUTIVE SUMMARY

### Fakta yang sudah pasti
- Kompetisi: klasifikasi 3 kelas (Organic, Recyclable, Electronic), metrik **Macro F1**, deadline 30 Juli 2026, **hanya 3 submission** untuk seluruh kompetisi.
- Train 26.527 gambar (Organic 47,4% / Recyclable 37,7% / Electronic 14,9%), test 1.458 gambar.
- **Leakage terkonfirmasi**: 97/1.458 file test (6,65%) exact-duplicate MD5 dengan train (95 Organic, 2 Electronic, 0 Recyclable).
- **Mislabel terkonfirmasi**: `O_8873.jpg` = `R_799.jpg` (tas kain "Say No To Plastic"), dan file byte-identik ini **juga ditemukan di dataset paper publik** (`organic_012716` vs `recyclable_008879`, hash sama: `95bd2693…`). Ini bukti independen bahwa noise labeling berasal dari sumber data asli, bukan human error panitia.
- Statistik warna/brightness/background/foreground-ratio kelas Organic & Recyclable kompetisi **sangat dekat** dengan dataset paper (selisih brightness <5 poin, foreground ratio <0,03) → populasi sumbernya sama atau sangat berdekatan.
- Electronic **tidak ada** di paper — sumbernya campuran (68,8% stock icon 150×150 + 31,2% foto natural), bimodal tajam.
- Keputusan CV (StratifiedGroupKFold, 5-fold, seed 42) sudah **locked**, dan Flag 3/6/7 sudah **closed**.
- Aturan kompetisi melarang data/metadata eksternal (Constraint 2) — ini relevan langsung ke isu overlap paper.

### Asumsi (belum terverifikasi eksplisit di dokumen)
- Hardware training tidak disebutkan eksplisit di Project State — diasumsikan **single-GPU kelas menengah** (mis. Colab/Kaggle T4-P100 atau RTX 30/40-series 8–16GB VRAM). Semua rekomendasi model/resolusi di bawah dibangun dengan asumsi ini; **perlu dikonfirmasi ke tim**, karena ini mengubah pilihan model/resolusi secara material.
- Test set diasumsikan **tidak** mengandung file dari paper dataset di luar 97 yang sudah terdeteksi sebagai train-test dupe — MD5 matching paper-vs-test belum selesai/di-hold.
- Distribusi kelas di test set diasumsikan mengikuti proporsi train (tidak ada info sebaliknya).

### Ketidakpastian
- Hasil fatwa Vierico soal penggunaan paper overlap (boleh/tidak sebagai prior knowledge).
- Sejauh mana noise labeling meluas di luar 1 kasus yang terdeteksi (H4 — hanya terdeteksi via exact-hash, mislabel non-duplikat tidak akan terdeteksi metode ini).
- Kapasitas hardware/compute riil tim.

### Risiko utama
1. Macro F1 sangat sensitif ke performa **Electronic** (kelas minoritas, ~15%, sumber data paling heterogen/noisy).
2. Model belajar **shortcut** background-putih=Recyclable / square-crop=Electronic, alih-alih fitur objek asli — kombinasi dari cara data dikumpulkan (studio vs natural), bukan sinyal kausal yang pasti muncul di test set.
3. Hanya 3 submission → tidak ada ruang untuk LB-probing; CV *harus* dipercaya 100%.
4. Risiko benchmark leakage via paper overlap jika nanti ternyata test set juga bersumber dari data publik yang sama — area abu-abu aturan.

### Peluang
- Paper (meski usang metodenya) secara tidak langsung **mengonfirmasi** bahwa Organic-vs-Recyclable adalah masalah "mudah" secara statistik: bahkan InceptionV3-frozen-feature + SVM klasik tanpa fine-tuning mencapai F1 96,6% pada 2-kelas ini. Artinya, dengan backbone modern yang di-fine-tune penuh, sub-problem Organic vs Recyclable kemungkinan besar akan **dekat saturasi/ceiling** dengan cepat.
- Konsekuensinya: **kompetisi ini akan dimenangkan atau dikalahkan oleh seberapa baik model menangani Electronic** — bukan oleh Organic/Recyclable. Ini harus jadi pusat gravitasi seluruh strategi eksperimen.

---

## 2. DATASET ANALYSIS

### Identik atau hanya mirip?
Bukti mengarah ke **populasi sumber yang sama**, bukan sekadar mirip secara statistik:
1. Statistik distribusi (brightness, warna, background variance, foreground ratio, dimensi) untuk Organic & Recyclable kompetisi vs paper **konsisten pada level yang tidak mungkin kebetulan** — selisih brightness Organic 145,05 (kompetisi) vs 140,58 (paper sample), Recyclable 181,76 vs 180,52; foreground ratio 0,761/0,589 (kompetisi) vs 0,786/0,600 (paper).
2. **Bukti definitif**: pasangan cross-class duplicate `O_8873.jpg`/`R_799.jpg` di kompetisi adalah file **byte-identik** dengan pasangan duplicate di paper dataset. Probabilitas ini terjadi secara independen (dua sumber data berbeda kebetulan punya file identik dengan mislabel identik) mendekati nol.

Kesimpulan: dataset kompetisi untuk kelas Organic dan Recyclable adalah **subset atau turunan langsung** dari lineage dataset yang sama dengan paper (paper sendiri menyebut datasetnya diambil dari Mendeley, hasil restrukturisasi dataset Kaggle 25.077 gambar oleh Nnamoko et al. — lihat referensi [18] di paper). Electronic **bukan** bagian dari lineage ini.

### Implikasi ilmiah
- Hasil akurasi/F1 di paper (96,3% SVM) **valid sebagai bukti bahwa problem 2-kelas ini "mudah dipisahkan"**, tapi tidak valid sebagai benchmark angka langsung untuk kompetisi, karena: (a) metrik beda (accuracy vs macro-F1 3-kelas dengan imbalance), (b) tidak ada kelas Electronic yang justru jadi penentu skor kompetisi, (c) metodologi paper (frozen feature + classifier klasik, hyperparameter default MATLAB tanpa tuning) bukan pendekatan yang kompetitif untuk deep learning modern.
- Paper **tidak bisa dijadikan baseline angka** untuk leaderboard, tapi **bisa dijadikan sanity-check kualitatif**: jika model modern kompetisi tidak mencapai F1 tinggi (>95%) di sub-masalah Organic vs Recyclable, itu sinyal ada bug di pipeline — karena problem ini sudah terbukti "mudah" bahkan dengan metode 2016.

### Distribution shift
- Untuk Organic/Recyclable: shift minimal (populasi sumber sama). Risiko utama di sini bukan shift, tapi **shortcut learning** (lihat Flag 4/H3 di bawah) — background & brightness berkorelasi kuat dengan label karena cara pengambilan foto, bukan karena sifat fisik objek.
- Untuk Electronic: **tidak ada pembanding di paper sama sekali**. Kelas ini punya distribution shift internal (bimodal: icon 150×150 vs foto natural resolusi tinggi) — ini adalah domain-shift *di dalam kelasnya sendiri*, bukan cuma train-test shift. Ini risiko yang lebih besar daripada train-test shift konvensional.

### Potensi benchmark leakage
- Karena Organic/Recyclable kompetisi = subset dataset publik berlabel, **ada kemungkinan test set kompetisi juga overlap dengan dataset publik ini** (di luar 97 yang sudah terdeteksi sebagai train-test dupe internal). Jika MD5 matching paper-vs-test (sedang di-hold) menemukan overlap, label ground truth sebagian test bisa "bocor" dari sumber eksternal.
- Ini **PERSIS** kasus benchmark/data leakage klasik di kompetisi ML publik (dataset kompetisi diam-diam berasal dari dataset publik berlabel). Keputusan **hold** yang sudah diambil tim adalah keputusan yang benar — memakai informasi ini tanpa fatwa melanggar Constraint 2 ("dilarang menggunakan data eksternal") dan berisiko diskualifikasi.

### Risiko overfitting terhadap karakteristik dataset
- Karena Organic/Recyclable dikumpulkan dengan protokol foto yang berbeda (natural vs studio), model condong belajar "background putih polos → Recyclable" alih-alih fitur objek. Ini valid *di dalam distribusi saat ini*, tapi rapuh jika test set (atau data real-world) diambil dengan protokol foto berbeda.
- Mitigasi: augmentasi background-aware (random erasing pada area background, CutMix, atau background randomization) + verifikasi Grad-CAM.

### Kelas Electronic: perlu strategi khusus?
**Ya, wajib.** Alasannya bukan cuma imbalance (14,9%), tapi:
1. **Bimodalitas ekstrem** (68,8% icon 150×150 vs 31,2% foto natural resolusi tinggi) — dua distribusi visual yang sangat berbeda digabung dalam satu label.
2. Sumber campuran (nama file bervariasi: `Player_xx`, `Washing_Machine_x`, `pcb_xxx`, `IMG_20200328…`) — noise sumber data lebih tinggi dari 2 kelas lain.
3. Duplikat within-class proporsional tertinggi (1,51%) — jumlah gambar unik efektif lebih kecil dari 3.961.
4. RGBA & palette-mode image terkonsentrasi di kelas ini — sinyal tambahan sumber data campuran/kualitas kontrol lemah.
5. Tidak ada preseden ilmiah dari paper sama sekali untuk kelas ini — tim harus membangun strategi dari nol, bukan mentransfer dari paper.

Strategi khusus yang diperlukan: resize/padding aspect-ratio-preserving (bukan naive upscale untuk subset 150×150), evaluasi CV terpisah per-subpopulasi (icon vs foto natural) untuk memastikan model tidak hanya kuat di satu subpopulasi, dan pembobotan loss/oversampling untuk mengatasi imbalance.

---

## 3. KNOWLEDGE TRANSFER

### Dapat langsung digunakan
- **Prinsip transfer learning dari backbone pretrained bekerja sangat baik di domain ini.** Paper membuktikan bahkan *frozen* feature InceptionV3 (2015, tanpa fine-tuning sama sekali) + classifier klasik sudah mencapai F1 96,6% pada 2 kelas. Ini validasi kuat bahwa arsitektur modern yang di-*fine-tune penuh* akan sangat mumpuni untuk sub-masalah Organic/Recyclable — prior ini menaikkan keyakinan bahwa effort riset sebaiknya difokuskan ke Electronic, bukan disebar rata ke 3 kelas.
- **Kerangka evaluasi metrik** (precision/recall/F1 per kelas, confusion matrix) — logikanya sama persis, tinggal disesuaikan ke macro-averaging 3-kelas.
- **Insight kualitatif domain**: organic waste secara visual didominasi warna hangat (R dominan) dan tekstur natural; recyclable cenderung studio-lit, warna netral/terang. Insight ini berguna untuk desain augmentasi (jangan terlalu agresif mengubah color jitter pada channel yang justru jadi sinyal utama, tapi tetap harus dirandomisasi cukup untuk cegah shortcut).

### Perlu dimodifikasi
- **Pendekatan feature-extraction beku + classifier klasik → ganti total menjadi fine-tuning end-to-end** backbone modern. Paper memvalidasi *konsep* transfer learning, bukan *implementasinya* — implementasi mereka (2048 fitur InceptionV3 dibekukan, lalu SVM/DT/KNN) sudah usang untuk standar 2026.
- **Protokol validasi paper lemah** — mereka menyebut 10-fold CV tapi implementasi sebenarnya tampak seperti hold-out sederhana via MATLAB Classification Learner dengan parameter default (Table I), tanpa pembahasan leakage/duplikasi sama sekali. **Jangan tiru rigor ini** — StratifiedGroupKFold yang sudah di-lock tim jauh lebih ketat dan harus dipertahankan.
- **Asumsi 2-kelas seimbang** di paper (56%/44%) tidak transferable ke setting 3-kelas dengan minoritas 14,9% — teknik penanganan imbalance harus dibangun sendiri, paper tidak memberi panduan di area ini.

### Sebaiknya tidak digunakan
- **Dataset mentah paper (24.705 gambar) sebagai data training tambahan.** Ini melanggar Constraint 2 (dilarang data eksternal) dan sedang di-hold menunggu fatwa Vierico. Rekomendasi tegas: **jangan gunakan sampai ada fatwa eksplisit**, walau secara teknis akan mudah dan menggiurkan (menambah data Organic/Recyclable yang sudah kuat, padahal yang dibutuhkan justru data Electronic yang tidak ada di paper sama sekali).
- **Pilihan classifier paper (SVM cubic/DT/KNN pada fitur beku)** — tidak kompetitif dibanding fine-tuning CNN/Transformer modern untuk macro-F1 3-kelas dengan kelas heterogen seperti Electronic; classifier klasik pada fitur generik tidak akan menangani bimodalitas visual Electronic dengan baik.
- **Disiplin hyperparameter paper** (default MATLAB, tidak ada tuning terdokumentasi) — bertentangan dengan kebutuhan kompetisi yang hanya punya 3 submission, sehingga *harus* ada tuning offline yang disiplin via CV.

---

## 4. STRATEGY REVIEW (Project State 2)

| Strategi/Keputusan | Masih valid? | Catatan kritis |
|---|---|---|
| **Flag 3** — Exclude 97 dupe untuk CV, full data + hash-override untuk final | ✅ Valid, tidak perlu revisi | Paper tidak membahas leakage sama sekali — ini justru keunggulan tim dibanding paper aslinya. Hash-override untuk 97 test file adalah keputusan tepat karena ground truth-nya memang sudah diketahui pasti (bukan diprediksi model) — pastikan implementasi tidak salah index saat mapping balik ke `submission.csv`. |
| **Flag 6** — StratifiedGroupKFold, jangan exclude within-class dupe | ✅ Valid | Best practice standar untuk cegah leakage antar-fold. Tidak ada revisi. |
| **Flag 7** — Exclude `O_8873.jpg`, keep `R_799.jpg` | ✅ Valid, **keyakinan naik** | Temuan independen di paper (hash identik) mengonfirmasi ini bukan human error panitia — memperkuat keputusan exclude. |
| **Weighted Loss/Oversampling untuk Electronic** (Flag 1, wajib) | ✅ Valid arahnya, perlu spesifikasi lebih tajam | Paper tidak beri panduan (kelasnya seimbang). Rekomendasi: gunakan **Class-Balanced Loss** (effective number of samples, Cui et al.) atau **Focal Loss** dibanding pembobotan invers-frekuensi naif, karena weighting naif pada deep net kerap mengacaukan kalibrasi & konvergensi. Kombinasikan dengan augmentasi kuat untuk minoritas, bukan oversampling duplikat mentah (bisa memperparah within-class duplicate yang sudah 1,51% di Electronic). |
| **Resize/padding untuk Electronic bimodal** (Flag 5) | ✅ Valid, perlu detail teknis | Tambahkan: resize aspect-ratio-preserving + padding (bukan stretch), resolusi input disarankan ≥288px agar foto natural (median 1028px) tidak kehilangan detail berlebihan sementara icon 150×150 tidak di-upscale secara ekstrem tanpa interpolation berkualitas (lanczos sudah tepat). |
| **Grad-CAM shortcut check** (Flag 4/H3) | ✅ Valid, **naikkan prioritas** | Sebaiknya dijalankan **segera setelah baseline pertama**, bukan sebagai eksperimen belakangan — karena jika shortcut background terkonfirmasi, ini mengubah desain augmentasi (perlu background randomization/CutMix agresif) untuk SEMUA eksperimen berikutnya. Menunda ini berisiko membuang submission budget pada model yang overfit ke shortcut. |
| **Hold paper-overlap investigation** | ✅ Valid, tepat kehati-hatiannya | Selaras dengan Constraint 2. Direkomendasikan: jangan pernah gunakan hasil MD5 paper-vs-test untuk mempengaruhi model/submission sampai ada fatwa tertulis dari Vierico — termasuk godaan implisit seperti "menambah data training dari paper untuk kelas yang overlap." |

**Kesimpulan Strategy Review**: Tidak ada strategi lama yang perlu dibatalkan. Yang perlu ditambahkan adalah **penajaman prioritas** — Grad-CAM check dan strategi Electronic-specific harus naik ke urutan paling awal roadmap eksperimen, karena paper baru saja mengonfirmasi bahwa Organic/Recyclable bukan area yang butuh inovasi.

---

## 5. MODEL SELECTION

| Kriteria | Paper (InceptionV3-frozen + SVM) | EVA-02 Tiny | Swin V2 Tiny | ConvNeXt V2 Tiny/Nano (alternatif) |
|---|---|---|---|---|
| Akurasi potensial (domain ini) | Tinggi utk 2-kelas (96,6% F1), **tidak diketahui** utk Electronic | Sangat tinggi (distilasi EVA-CLIP, representasi semantik kuat) — unggul di fine-grained & tekstur | Tinggi, sangat stabil, terbukti luas di banyak benchmark klasifikasi | Tinggi, kompetitif dgn Swin V2 di skala Tiny, sering lebih stabil |
| Generalisasi ke kelas heterogen (Electronic) | Buruk — classifier klasik pada fitur beku sulit menangani 2 sub-populasi visual berbeda | Berpotensi kuat — representasi semantik CLIP-distilled cenderung lebih robust lintas domain visual (foto vs icon) | Baik — jendela hierarkis menangkap struktur multi-skala, relevan utk objek besar (icon) maupun kecil-di-frame (foto natural) | Baik, inductive bias conv kuat utk tekstur & bentuk lokal |
| Transfer learning / fine-tuning | Tidak di-fine-tune sama sekali (fitur beku) | Perlu resep fine-tuning lebih hati-hati (layer-wise LR decay, warmup) — lebih sensitif hyperparameter | Resep fine-tuning matang & terdokumentasi luas (timm), lebih "berjalan langsung" | Resep fine-tuning sangat matang, salah satu yang paling *forgiving* di antara opsi modern |
| Kompleksitas implementasi | Rendah tapi usang | Sedang-tinggi | Sedang | Rendah-sedang |
| Risiko (given 26,5K data, 3 submission only) | Tidak relevan sbg kandidat utama | **Risiko lebih tinggi** — arsitektur lebih "finicky", kegagalan konvergensi/overfitting lebih mungkin dgn budget eksperimen ketat | **Risiko rendah** — pilihan paling dapat diprediksi | **Risiko rendah**, alternatif solid |
| Kecocokan dgn constraint 3-submission | Buruk (tidak scalable) | Sedang (butuh lebih banyak percobaan offline utk stabil) | **Baik** | **Baik** |

### Rekomendasi
- **Model utama: Swin V2 Tiny** (atau ConvNeXt V2 Tiny sebagai setara-kuat jika tim lebih nyaman dengan resep training conv yang lebih sederhana/cepat konvergen). Alasan: dengan hanya 3 submission dan CV sebagai satu-satunya sumber kepercayaan, **stabilitas & reproducibility training jauh lebih berharga** daripada potensi akurasi marjinal lebih tinggi dari arsitektur yang lebih sensitif hyperparameter.
- **Model kandidat sekunder/ensemble: EVA-02 Tiny.** Representasi semantik hasil distilasi CLIP-nya punya alasan teoretis kuat untuk membantu tepatnya di titik lemah kompetisi ini: membedakan icon-Electronic vs foto-Electronic vs Recyclable (yang sama-sama sering studio-background). Namun karena lebih sensitif dan berisiko, posisikan sebagai eksperimen **setelah** baseline Swin V2 Tiny solid — bukan taruhan pertama.
- **Paper's approach**: tidak direkomendasikan sebagai model kompetisi, hanya sebagai *sanity-check floor* untuk sub-masalah Organic/Recyclable.

---

## 6. COMPETITION STRATEGY

| Komponen | Keputusan | Alasan |
|---|---|---|
| **Baseline pertama** | Swin V2 Tiny, ImageNet-22k pretrained, fine-tune penuh, resolusi 224–288, tanpa trik eksotis (no Mixup/CutMix dulu) | Perlu titik referensi bersih untuk mengukur efek tiap trik secara terisolasi, dan untuk segera menjalankan Grad-CAM check |
| **Optimizer** | AdamW | Standar untuk fine-tuning Transformer/modern-conv, weight decay decoupled lebih stabil dibanding SGD utk skenario data terbatas |
| **Scheduler** | Cosine decay + linear warmup (5–10% steps) | Mengurangi risiko instabilitas awal fine-tuning, terbukti robust lintas arsitektur modern |
| **Augmentation dasar** | RandAugment ringan-sedang + horizontal flip + color jitter terbatas | Waspada: color jitter agresif bisa merusak sinyal brightness yang justru diskriminatif untuk Organic/Recyclable — perlu tuning intensitas, bukan preset default agresif |
| **Mixup** | Digunakan **setelah** baseline, terutama utk membantu regularisasi Electronic minoritas | Membantu generalisasi kelas kecil, tapi uji dampaknya lewat CV per-kelas (bisa mengaburkan batas objek kecil di frame khas Electronic icon) |
| **CutMix** | Prioritas lebih tinggi dari Mixup **jika Grad-CAM konfirmasi shortcut background** | CutMix memaksa model memakai sinyal lokal dari objek, bukan hanya konteks background global — mitigasi langsung utk Flag 4/H3 |
| **Label Smoothing** | Digunakan (ε=0.1) sejak fine-tuning kedua | Membantu kalibrasi & mengurangi overconfidence pada label yang diketahui noisy (Flag 7 kasus, dan potensi noise tak terdeteksi H4) |
| **EMA (Exponential Moving Average)** | Digunakan pada semua run setelah baseline | Praktik standar modern yang murah secara komputasi, konsisten menstabilkan hasil evaluasi akhir — penting krn budget submission sangat terbatas |
| **Class weighting** | Class-Balanced Loss (effective number of samples), bukan invers-frekuensi naif | Lebih stabil untuk deep net dibanding weighting naif; ditargetkan langsung untuk menaikkan recall Electronic tanpa merusak kalibrasi kelas lain |
| **Oversampling** | Sampler berbasis weighted-random per epoch untuk Electronic, dikombinasi augmentasi kuat (bukan duplikasi murni) | Duplikasi murni akan memperparah within-class duplicate Electronic yang sudah 1,51% — kombinasi dgn augmentasi mengurangi risiko ini |
| **Cross-validation** | StratifiedGroupKFold locked (5-fold, seed 42) — dipertahankan, tidak diubah | Sudah dibahas di Strategy Review — tetap sumber kepercayaan utama |
| **Pseudo labeling** | **Tidak digunakan pada fase awal**; pertimbangkan hanya di akhir jika waktu & submission budget masih ada, dan hanya pada 1.361 file test non-overlap (jangan pernah pseudo-label 97 file overlap — sudah ada ground truth pasti) | Risiko tinggi mengingat budget submission hanya 3 dan tidak ada cara memvalidasi pseudo-label secara online berulang |
| **Test time augmentation (TTA)** | Digunakan pada submission final (flip + multi-crop ringan) | Biaya rendah, potensi kenaikan macro-F1 kecil-konsisten, tidak menambah risiko training |
| **Ensemble** | Swin V2 Tiny + ConvNeXt V2 Tiny (dua backbone dengan inductive bias berbeda) di submission ke-3 jika CV mendukung | Diversitas arsitektur lebih efektif daripada ensemble seed-only pada dataset skala menengah ini |
| **Model selection** | Berdasarkan **macro-F1 per-fold CV**, khususnya F1 Electronic — bukan hanya rata-rata makro | Karena Electronic adalah penentu utama skor, model dgn macro-F1 rerata sama tapi F1-Electronic lebih tinggi harus dipilih |
| **Checkpoint strategy** | Simpan checkpoint EMA-weights terbaik berdasarkan **macro-F1 validasi**, bukan loss atau accuracy | Selaras langsung dgn metrik kompetisi |

---

## 7. EXPERIMENT ROADMAP

| # | Eksperimen | Tujuan | Hipotesis | Expected Improvement | Biaya Komputasi | Prioritas | Kriteria Berhasil/Gagal |
|---|---|---|---|---|---|---|---|
| 1 | DataLoader final + StratifiedGroupKFold implementation | Fondasi pipeline yang benar | Pipeline bebas leakage akan memberi CV yang bisa dipercaya | N/A (enabler) | Rendah | **Tertinggi (sudah in progress)** | CV score antar-fold stabil (std rendah), tidak ada file duplikat lintas fold |
| 2 | Baseline Swin V2 Tiny, fine-tune polos, resolusi 224 | Titik referensi bersih | Macro F1 baseline akan tinggi di Organic/Recyclable (>0.9x per class F1), lebih rendah di Electronic | Baseline reference | Sedang | **Tertinggi** | Model konvergen stabil; F1 per kelas terekam sbg baseline |
| 3 | Grad-CAM shortcut check pasca-baseline | Validasi H3 (shortcut background/shape) | Model kemungkinan memakai background sbg sinyal dominan | N/A (diagnostik) | Rendah | **Tertinggi (segera setelah #2)** | Jika attention map fokus di background bukan objek → shortcut terkonfirmasi, augmentasi strategi berubah |
| 4 | Class-Balanced Loss / Focal Loss utk Electronic | Naikkan recall/F1 Electronic | Weighting yg tepat akan menaikkan F1 Electronic tanpa menurunkan Organic/Recyclable signifikan | +Macro F1 sedang-tinggi (area dampak terbesar) | Sedang | **Tinggi** | F1 Electronic naik ≥ beberapa poin, F1 kelas lain tidak turun signifikan |
| 5 | CutMix (jika shortcut terkonfirmasi di #3) | Paksa model pakai fitur lokal objek | Mengurangi ketergantungan model pada background global | +Macro F1 sedang, +robustness | Sedang | **Tinggi (kondisional pada #3)** | Grad-CAM ulang menunjukkan fokus bergeser ke objek; F1 stabil/naik |
| 6 | Resize/padding aspect-ratio-preserving + resolusi ≥288 utk subset Electronic bimodal | Tangani domain shift internal Electronic (icon vs foto) | Resolusi lebih tinggi + padding proporsional akan menaikkan F1 subpopulasi foto natural tanpa merusak subpopulasi icon | +F1 Electronic | Sedang-tinggi (resolusi lebih besar = compute lebih besar) | Tinggi | Evaluasi CV terpisah per-subpopulasi (icon vs foto) menunjukkan kenaikan di kedua sisi atau minimal tidak ada yg turun |
| 7 | EMA + Label Smoothing | Stabilisasi & kalibrasi | Mengurangi overfitting ke label noise & varians antar-run | +stabilitas, +Macro F1 kecil-sedang | Rendah | Tinggi | CV antar-seed lebih konsisten |
| 8 | Mixup (uji terpisah dari CutMix, bandingkan) | Regularisasi tambahan Electronic minoritas | Mixup membantu margin decision Electronic vs kelas lain | +Macro F1 kecil | Sedang | Sedang | F1 Electronic naik dibanding baseline+CutMix; jika turun, drop |
| 9 | EVA-02 Tiny fine-tuning (kandidat kedua) | Uji apakah representasi semantik CLIP-distilled lebih unggul khusus utk Electronic | EVA-02 lebih baik memisahkan icon-vs-foto-vs-recyclable-studio karena representasi lebih semantik | +Macro F1 jika hipotesis benar | Tinggi (lebih sensitif, butuh lebih banyak percobaan tuning) | Sedang | Bandingkan F1 Electronic head-to-head vs Swin V2 Tiny di CV yang sama; adopsi hanya jika unggul jelas |
| 10 | Ensemble Swin V2 Tiny + ConvNeXt V2 Tiny (atau + EVA-02 jika menang di #9) | Maksimalkan skor submission final | Diversitas arsitektur menaikkan macro-F1 lebih dari model tunggal terbaik | +Macro F1 kecil-sedang | Tinggi (2× training + inference) | Sedang (dekat akhir) | Ensemble CV score > model tunggal terbaik; jika tidak, submit model tunggal saja |
| 11 | TTA pada submission final | Kenaikan skor murah-risiko-rendah | TTA (flip+crop ringan) menaikkan macro-F1 sedikit tanpa risiko tambahan | +Macro F1 kecil | Rendah (inference only) | Sedang | CV dgn TTA ≥ CV tanpa TTA |
| 12 (opsional) | Pseudo-labeling pada 1.361 test non-overlap | Data augmentation tambahan jika waktu tersisa | Pseudo-label berkualitas tinggi bisa menambah sinyal training, khususnya Electronic | +Macro F1 kecil, **risiko tinggi** | Sedang-tinggi | **Rendah (hanya jika waktu & submission budget sisa)** | Hanya lanjut jika confidence threshold sangat tinggi & validasi CV internal mendukung; jika ragu, skip |

---

## 8. RISK ASSESSMENT

| Risiko | Severity | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|---|
| **Class imbalance Electronic menekan Macro F1** | Tinggi | Pasti terjadi (sudah terkonfirmasi dari EDA) | Skor kompetisi turun signifikan jika tidak ditangani | Class-Balanced Loss, oversampling+augmentasi, evaluasi F1 per-kelas sbg kriteria utama model selection |
| **Shortcut learning (background/shape)** | Tinggi | Sedang-tinggi (indikasi kuat dari statistik EDA) | Model tampak bagus di CV tapi rapuh di test/real-world | Grad-CAM check dini (Roadmap #3), CutMix, augmentasi background-aware |
| **Domain shift internal Electronic (bimodal icon vs foto)** | Tinggi | Pasti terjadi (68,8%/31,2% terkonfirmasi) | Model overfit ke satu subpopulasi, lemah di subpopulasi lain | Resize aspect-ratio-preserving, evaluasi CV terpisah per-subpopulasi |
| **Overfitting akibat budget submission (3x) & CV-LB gap tak terduga** | Tinggi | Sedang | Model terbaik di CV ternyata jelek di LB, submission terbuang | StratifiedGroupKFold ketat, jangan pernah pilih model berdasar intuisi tanpa CV, sisakan minimal 1 submission cadangan sampai H-akhir |
| **Data leakage residual (train-test dupe)** | Rendah (sudah ditangani) | Rendah (mitigasi sudah diterapkan) | Overestimasi CV jika lupa exclude 97 file | Pastikan implementasi exclusion list benar-benar dieksekusi di kode DataLoader, tambahkan unit test/assertion jumlah file |
| **Benchmark/eksternal leakage (paper overlap dgn test)** | Tinggi (jika terkonfirmasi & dipakai) | Belum diketahui (di-hold) | Risiko pelanggaran aturan/diskualifikasi jika dipakai tanpa izin | Jangan gunakan hasil MD5 paper-vs-test sampai fatwa Vierico eksplisit; tetap HOLD |
| **Noisy label residual (H4 — noise di luar 1 kasus terdeteksi)** | Sedang | Sedang (sudah ada bukti sistemik noise dari sumber data) | Training terganggu label salah yang tak terdeteksi | Label smoothing, loss robust ke noise (mis. symmetric cross-entropy jika waktu tersedia), audit manual sampel acak (prioritas rendah, sesuai catatan Jeremy) |
| **Underfitting krn arsitektur terlalu "finicky" (mis. EVA-02) dgn budget eksperimen terbatas** | Sedang | Sedang jika EVA-02 dipaksa jadi model utama | Model gagal konvergen baik, submission terbuang | Posisikan EVA-02 sbg kandidat sekunder, bukan taruhan pertama; Swin V2/ConvNeXt sbg jangkar utama |
| **Hardware limitation (belum eksplisit di dokumen)** | Sedang | Tidak diketahui | Bisa memaksa turun resolusi/backbone lebih kecil dari rencana | **Perlu konfirmasi segera** dari tim spesifikasi hardware riil sebelum commit ke resolusi ≥288px & ensemble 2-model |
| **Waktu kompetisi (deadline 30 Juli, ~4 minggu dari hari ini)** | Sedang-tinggi | Pasti (hard constraint) | Roadmap 12 eksperimen bisa tidak selesai semua | Prioritaskan #1–#7 (fondasi + shortcut-fix + imbalance-fix) sbg wajib; #9–#12 sbg stretch goals kondisional waktu |

---

## 9. FINAL VERDICT

Sebagai *Principal ML Engineer* yang bertanggung jawab atas arah proyek ini, berikut keputusan final:

**Tidak semua strategi sebelumnya perlu diganti** — pekerjaan Ababil/Jeremy/Vierico di fase EDA & data-integrity (Flag 3, 6, 7, StratifiedGroupKFold) sudah solid dan divalidasi ulang secara independen oleh bukti dari paper. Yang perlu ditambahkan adalah **fokus strategis yang lebih tajam**: paper baru saja membuktikan bahwa Organic-vs-Recyclable bukan medan pertempuran sebenarnya di kompetisi ini — **Electronic adalah satu-satunya variabel yang benar-benar menentukan posisi leaderboard.**

- **Strategi modeling final**: Fine-tuning end-to-end backbone modern (bukan feature-extraction beku ala paper), dengan seluruh trik training diarahkan secara eksplisit untuk menaikkan F1 Electronic tanpa mengorbankan F1 Organic/Recyclable yang sudah dekat saturasi.
- **Model utama**: **Swin V2 Tiny**, ImageNet-22k pretrained, fine-tune penuh.
- **Model cadangan**: **ConvNeXt V2 Tiny** (setara-kuat, resep training lebih forgiving) sebagai fallback jika Swin V2 tidak stabil di hardware riil tim, dan sebagai partner ensemble di submission ke-3. **EVA-02 Tiny** sebagai kandidat eksperimen lanjutan (bukan jangkar utama) — hanya dipromosikan jika terbukti unggul jelas di F1 Electronic pada CV yang sama.
- **Strategi training**: AdamW + cosine schedule + warmup, RandAugment ringan, EMA, Label Smoothing (ε=0.1), Class-Balanced Loss untuk imbalance, CutMix diaktifkan segera jika Grad-CAM mengonfirmasi shortcut background.
- **Strategi validasi**: StratifiedGroupKFold (locked, 5-fold, seed 42) — tidak berubah. Model selection berdasarkan **F1 Electronic** sebagai kriteria pembeda utama saat macro-F1 rerata setara.
- **Strategi handling Electronic**: resize aspect-ratio-preserving + padding, resolusi input ≥288px, evaluasi CV terpisah per-subpopulasi (icon 150×150 vs foto natural), Class-Balanced Loss + oversampling-dengan-augmentasi (bukan duplikasi murni).
- **Urutan eksperimen**: (1) DataLoader & CV fix → (2) baseline bersih → (3) Grad-CAM shortcut check → (4) Class-Balanced Loss → (5) CutMix (kondisional) → (6) resize/resolusi Electronic → (7) EMA+LabelSmoothing → (8) Mixup uji-banding → (9) EVA-02 uji-banding → (10) ensemble → (11) TTA → (12) pseudo-labeling (opsional, hanya jika waktu & keyakinan tinggi).
- **Kondisi kapan strategi harus diubah**:
  - Jika Grad-CAM **tidak** menunjukkan shortcut background → skip CutMix, alihkan compute ke tuning Class-Balanced Loss & resolusi lebih lanjut.
  - Jika hardware ternyata jauh lebih terbatas dari asumsi (single GPU kecil/Colab free-tier) → turunkan ke ConvNeXt V2 Nano/Tiny sebagai model utama tunggal, drop rencana ensemble, fokuskan 3 submission pada model tunggal terbaik + TTA saja.
  - Jika fatwa Vierico mengizinkan penggunaan insight paper overlap sebagai "prior knowledge" (bukan data eksternal langsung) → tetap **jangan** menambah data training dari paper; paling jauh gunakan sebagai validasi kualitatif tambahan bahwa pipeline Organic/Recyclable sudah benar.
  - Jika CV per-fold menunjukkan std tinggi (tidak stabil) meski sudah StratifiedGroupKFold → curigai noise label residual (H4), pertimbangkan audit manual sampel sebelum lanjut ke eksperimen lanjutan.

**Confidence level: 75%**

Alasan: Keyakinan tinggi pada bagian yang didukung bukti kuat dan langsung (leakage handling, mislabel handling, arah fokus ke Electronic — semua tervalidasi ganda oleh paper + EDA internal). Confidence tidak 90%+ karena tiga ketidakpastian nyata yang belum terselesaikan di luar kendali analisis ini: (1) spesifikasi hardware riil belum dikonfirmasi, sehingga pilihan resolusi/ensemble bisa perlu direvisi, (2) fatwa Vierico terkait paper-overlap belum keluar, dan (3) sejauh mana noise label residual (H4) yang tak terdeteksi bisa mempengaruhi hasil training tidak bisa dipastikan tanpa audit manual yang belum dijalankan.
