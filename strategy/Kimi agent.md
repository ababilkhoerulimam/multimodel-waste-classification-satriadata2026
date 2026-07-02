Analisis Strategis: **Model modern seperti ConvNeXt V2 dan EfficientNetV2 memberikan performa superior dibandingkan replikasi pendekatan paper ICISNA 2023**, yang hanya menggunakan feature extraction dengan InceptionV3 + SVM. Untuk kompetisi BDC Satria Data 2026, strategi paling optimal adalah menggunakan **EfficientNetV2-M atau ConvNeXt V2-Base** sebagai backbone, fine-tuning penuh dengan **discriminative learning rate**, mengimplementasikan **Class-Balanced Sampling** untuk menangani imbalance, dan menggunakan **ensemble 3-5 model** dengan snapshot ensemble atau SWA. Pendekatan ini, dikombinasikan dengan augmentasi kuat (RandAugment, CutMix, Mixup), diharapkan mampu mencapai **Macro F1 > 96%** pada leaderboard, jauh melampaui potensi pendekatan paper yang mungkin terbatas sekitar 85-88%.

---

# Analisis Strategis Pendekatan Modeling Big Data Challenge Satria Data 2026: Waste Classification

**Tanggal:** 2026-07-02

Dokumen ini disusun sebagai panduan strategis komprehensif untuk tim Big Data Challenge (BDC) Satria Data 2026 dalam merancang, mengimplementasikan, dan mengevaluasi pendekatan modeling untuk kompetisi klasifikasi sampah. Analisis ini berdasarkan temuan kritis bahwa dataset yang digunakan identik dengan dataset penelitian sebelumnya (Yasin & Koklu, 2023) [^37^] untuk kelas Organic dan Recyclable, dengan tambahan kelas Electronic [^37^]. Dokumen ini mencakup analisis mendalam atas implikasi dataset, perbandingan strategi modeling (replikasi paper vs. pendekatan modern), strategi transfer learning, penanganan khusus untuk kelas Electronic, resep training yang optimal, seleksi model, dan perencanaan eksperimen yang efisien dalam batasan submission yang ketat. Seluruh rekomendasi didasarkan pada bukti empiris dari literatur terkini dan praktik terbaik dalam kompetisi machine learning untuk computer vision.

## 1. Implikasi Dataset: Analisis Mendalam

### 1.1. Konsekuensi Identitas Dataset dengan Paper ICISNA 2023

Temuan bahwa dataset kompetisi identik dengan penelitian yang dipublikasikan oleh Yasin dan Koklu pada konferensi ICISNA tahun 2023 membawa konsekuensi strategis yang sangat signifikan dan kompleks [^37^]. Penelitian tersebut berjudul "Classification of Organic and Recyclable Waste based on Feature Extraction and Machine Learning Algorithms" dan menggunakan dataset dari Mendeley Data yang berisi 24.705 gambar (13.880 gambar "Organic" dan 10.825 gambar "Recyclable") [^37^]. Kompetisi BDC Satria Data 2026 menambahkan kelas "Electronic" sebagai variabel baru, sehingga total data mencapai 26.527 gambar. Identitas dataset ini menciptakan dinamika unik antara memanfaatkan wawasan historis dan menghindari jebakan yang dapat merugikan performa di kompetisi.

#### 1.1.1. Potensi Baseline yang Kuat dan Risiko Overfitting

Salah satu konsekuensi paling langsung adalah adanya **potensi baseline performa yang sangat kuat dan terukur**. Paper ICISNA 2023 telah secara ekstensif mendokumentasikan performa dari berbagai kombinasi model feature extraction dan classifier machine learning. Hasil tertinggi yang dilaporkan adalah **akurasi 96.3% yang dicapai oleh kombinasi InceptionV3 dengan Support Vector Machine (SVM)** [^37^]. Angka ini memberikan target awal yang jelas bagi tim. Menggunakan pendekatan yang identik atau sangat mirip dengan yang dijelaskan dalam paper seharusnya menghasilkan performa yang konsisten dengan angka 96.3% untuk klasifikasi biner (Organic vs. Recyclable), asumsi seluruh pipeline direplikasi dengan benar. Ini memungkinkan tim untuk memvalidasi pipeline data, preprocessing, dan implementasi model mereka pada tahap awal dengan target yang jelas.

Namun, kekuatan ini juga membawa risiko besar, yaitu **risiko overfitting terhadap metodologi paper**. Karena pendekatan dalam paper sudah terbukti sangat efektif pada dataset ini, ada godaan kuat untuk mengadopsinya tanpa kritis. Tim mungkin menjadi terlalu fokus untuk menyempurnakan hyperparameter SVM atau InceptionV3, dan melewatkan peluang besar yang ditawarkan oleh arsitektur modern. Lebih penting lagi, paper tersebut menggunakan akurasi (accuracy) sebagai metrik evaluasi utama [^37^], sementara kompetisi BDC menggunakan **Macro F1-Score**. Akurasi dapat menjadi metrik yang menyesatkan pada dataset yang tidak seimbang, terutama ketika kelas Electronic, yang merupakan kelas baru, memiliki jumlah sampel yang jauh lebih sedikit (hanya 3.961 dari 26.527 total gambar). Oleh karena itu, **baseline yang kuat dari paper mungkin tidak secara langsung menerjemahkan ke performa yang kompetitif pada metrik Macro F1-Score**, yang mengharuskan model untuk memiliki performa yang seimbang di semua kelas, termasuk kelas minoritas.

#### 1.1.2. Risiko Data Leakage dan Benchmark Leakage

Identitas dataset dengan paper yang sudah dipublikasikan secara terbuka memunculkan risiko signifikan terkait **data leakage dan benchmark leakage**. Data leakage dalam konteks ini bukan berarti ada tumpang tindih antara set pelatihan dan pengujian yang disediakan panitia, melainkan adanya **"kebocoran informasi" dari paper ke dalam model tim**. Karena paper tersebut telah menganalisis distribusi data, karakteristik visual, dan bahkan kegagalan model secara detail, seluruh tim yang menemukan paper ini memiliki keuntungan informasi yang sama. Setiap tim yang mereplikasi pendekatan paper akan mencapai hasil yang sangat mirip, membuatnya sulit untuk menonjol di kompetisi.

Risiko yang lebih strategis adalah **benchmark leakage**. Jika metode atau hasil dari paper ICISNA 2023 secara luas dikutip atau digunakan sebagai titik perbandingan, maka "standar" untuk dataset ini sudah terlalu mapan. Untuk menang kompetisi, tim tidak hanya perlu mencapai 96.3% akurasi, tetapi harus **secara signifikan melampaui baseline tersebut**, terutama pada metrik Macro F1 yang lebih sulit. Ini berarti replikasi murni dari paper tidak akan pernah cukup untuk mencapai peringkat teratas. Tim harus menganggap hasil 96.3% tersebut sebagai lantai (floor), bukan langit-langit (ceiling), dan secara agresif mengeksplorasi pendekatan yang lebih canggih untuk mengekstrak performa yang lebih tinggi.

#### 1.1.3. Ancaman Generalisasi dan Inovasi Pendekatan

Ketergantungan pada pendekatan paper yang sudah usang juga mengancam **kemampuan generalisasi dan inovasi**. Paper tersebut menggunakan arsitektur InceptionV3 yang dirilis pada tahun 2015 dan pendekatan tradisional feature extraction yang sudah tidak lagi menjadi state-of-the-art [^37^]. Meskipun pendekatan ini mungkin masih efektif untuk dataset ini, kemampuannya untuk menangani variasi dan kompleksitas yang lebih tinggi, seperti penambahan kelas "Electronic" dengan visual yang sangat beragam, bisa jadi terbatas. Kelas Electronic seringkali mencakup objek seperti ponsel, baterai, keyboard, dan kabel, yang memiliki variasi bentuk, tekstur, dan konteks visual yang jauh lebih besar dibandingkan sampah organik atau daur ulang.

Pendekatan modern seperti **Vision Transformers (ViT)**, **ConvNeXt**, atau **EfficientNetV2** dirancang untuk menangkap representasi visual yang lebih kaya dan hierarkis melalui mekanisme seperti self-attention atau convolutional blocks yang dioptimalkan [^75^] [^79^]. Mengabaikan kemajuan ini berarti tidak memanfaatkan potensi terbesar model untuk memahami nuansa kelas Electronic yang baru dan kompleks. Inovasi dalam pendekatan modeling bukan hanya tentang menggunakan arsitektur terbaru, tetapi juga tentang bagaimana menangani tantangan spesifik dari kompetisi, seperti class imbalance dan batasan jumlah submission, yang tidak dibahas dalam konteks paper asli.

### 1.2. Dataset sebagai Titik Awal, Bukan Akhir

#### 1.2.1. Peluang Peningkatan Signifikan di Atas Baseline Paper

Meskipun paper ICISNA 2023 menetapkan baseline yang kuat pada 96.3% akurasi, analisis mendalam menunjukkan adanya peluang yang sangat besar untuk peningkatan signifikan di atas angka tersebut, terutama dengan beralih ke metrik Macro F1-Score. Pertama, metodologi paper menggunakan pendekatan **feature extraction yang "beku"**, di mana bobot InceptionV3 tidak diperbarui selama pelatihan classifier (SVM, KNN, Decision Tree) [^37^]. Pendekatan ini, meskipun efisien secara komputasi, secara inheren membatasi kemampuan model untuk beradaptasi sepenuhnya dengan domain spesifik dari dataset sampah. Fine-tuning, atau melatih ulang seluruh bagian dari model pre-trained, memungkinkan model untuk menyesuaikan filter konvolusi awal untuk mendeteksi fitur-fitur yang lebih relevan dengan sampah (misalnya, tekstur khusus untuk plastik, daun, atau sirkuit elektronik), yang pada akhirnya menghasilkan representasi yang lebih diskriminatif.

Kedua, paper tersebut melaporkan hasil dengan menggunakan seluruh 2.048 fitur yang diekstrak oleh InceptionV3 tanpa seleksi fitur [^37^]. Terdapat potensi besar bahwa banyak dari fitur tersebut berisi informasi yang berlebihan (redundant) atau tidak relevan untuk tugas klasifikasi sampah. Menggunakan teknik seleksi fitur atau, yang lebih umum dalam deep learning modern, menggunakan lapisan fully connected yang dapat belajar untuk memberi bobot pada fitur yang paling penting, dapat menyederhanakan model dan meningkatkan generalisasi. Dengan beralih dari pendekatan manual feature extraction + ML klasik ke **end-to-end deep learning dengan fine-tuning**, tim dapat secara drastis meningkatkan kapasitas model untuk belajar representasi yang optimal langsung dari data, yang merupakan kunci untuk melampaui baseline yang sudah mapan.

#### 1.2.2. Eksplorasi Arsitektur Modern untuk Performa Superior

Lanskap computer vision telah berkembang pesat sejak publikasi paper ICISNA 2023. Eksplorasi arsitektur modern bukan lagi sebuah pilihan, melainkan **keharusan strategis untuk mencapai performa superior**. Model-model seperti **EfficientNetV2**, **ConvNeXt V2**, dan **Swin Transformer** telah menunjukkan performa yang jauh lebih unggul pada berbagai benchmark standar seperti ImageNet, dan mereka membawa kemajuan fundamental dalam cara model memahami gambar. Misalnya, EfficientNetV2 dirancang dengan pendekatan neural architecture search (NAS) yang mengoptimalkan tidak hanya akurasi tetapi juga kecepatan pelatihan dan efisiensi parameter, menjadikannya pilihan yang sangat kuat dan praktis [^75^].

Arsitektur modern ini memiliki kapasitas untuk menangkap **dependensi jangka panjang dan konteks global** dalam gambar dengan lebih baik. Ini sangat penting untuk tugas klasifikasi sampah, di mana konteks dapat sangat membantu. Misalnya, model modern mungkin belajar bahwa keberadaan papan sirkuit hijau mengindikasikan sampah elektronik, terlepas dari sudut pandang atau pencahayaannya, dengan lebih efektif daripada InceptionV3. Selain itu, praktik terbaik modern seperti penggunaan **augmentasi data yang lebih canggih (CutMix, Mixup)**, **regularisasi (Label Smoothing, Stochastic Depth)**, dan **strategi optimasi (AdamW, Cosine Decay)** secara kolektif berkontribusi pada kemampuan model untuk generalisasi yang jauh lebih baik. Mengadopsi seluruh ekosistem praktik modern ini, bukan hanya arsitekturnya, adalah jalur paling dapat diandalkan untuk mencapai skor Macro F1 yang tinggi dan kompetitif.

#### 1.2.3. Tantangan Kelas Electronic: Variabilitas dan Kompleksitas Visual

Penambahan kelas Electronic merupakan variabel paling signifikan yang membedakan kompetisi ini dari penelitian asli dan menjadi titik fokus utama untuk inovasi. Kelas ini memperkenalkan **kompleksitas visual yang jauh lebih tinggi** dibandingkan dua kelas lainnya. Sampah organik (makanan, dedaunan) dan daur ulang (kertas, plastik, logam) memiliki karakteristik visual yang relatif lebih konsisten dan mudah dibedakan. Sebaliknya, sampah elektronik mencakup spektrum objek yang sangat luas, mulai dari perangkat besar seperti monitor dan keyboard, hingga komponen kecil seperti kabel, charger, dan baterai. Ini menciptakan tantangan signifikan untuk model:

1.  **Variabilitas Bentuk dan Ukuran (Intra-class Variance):** Dua objek yang sama-sama termasuk "Electronic", seperti sebuah laptop dan mouse, memiliki perbedaan bentuk dan ukuran yang drastis. Model harus belajar untuk mengenali fitur-fitur umum yang mendefinisikan kategori "elektronik" (misalnya, keberadaan tombol, kabel, atau permukaan plastik/metal yang halus) di balik variasi besar ini.
2.  **Kemiripan dengan Kelas Lain (Inter-class Similarity):** Banyak objek elektronik, terutama yang terbuat dari plastik atau logam, dapat secara visual menyerupai objek dari kelas daur ulang. Misalnya, casing plastik dari perangkat elektronik bisa terlihat sangat mirip dengan botol plastik. Ini meningkatkan risiko model salah mengklasifikasikan Electronic sebagai Recyclable.
3.  **Keterbatasan Data:** Seperti yang terungkap dalam analisis data, kelas Electronic hanya memiliki **3.961 sampel**, jauh lebih sedikit daripada kelas Organic (12.567) dan Recyclable (9.999). Keterbatasan data ini, dikombinasikan dengan variabilitas visual yang tinggi, membuat kelas Electronic menjadi **"bottleneck" performa** yang paling mungkin menurunkan skor Macro F1 secara keseluruhan.

Oleh karena itu, strategi modeling harus secara eksplisit dirancang untuk menangani tantangan kelas Electronic. Ini termasuk teknik untuk mengatasi ketidakseimbangan data, augmentasi yang dirancang khusus untuk meningkatkan variasi sampel elektronik, dan arsitektur model yang cukup kuat untuk membedakan nuansa visual yang halus.

## 2. Strategi Modeling: Replikasi vs. Inovasi

### 2.1. Evaluasi Replikasi Model Paper sebagai Baseline

Replikasi model dari paper ICISNA 2023 oleh Yasin dan Koklu, yang menggunakan kombinasi InceptionV3 untuk ekstraksi fitur dan SVM untuk klasifikasi, merupakan langkah strategis yang bijaksana sebagai titik awal. Paper tersebut berhasil mencapai akurasi **96.3% pada dataset biner** (Organic vs. Recyclable), yang menetapkan standar performa yang kuat dan terukur [^37^]. Meskipun demikian, penting untuk menyadari bahwa replikasi ini harus diperlakukan sebagai **baseline minimal**, bukan solusi akhir. Tantangan utama yang dihadapi dalam kompetisi ini adalah adanya kelas tambahan, yaitu Electronic, yang secara signifikan meningkatkan kompleksitas tugas klasifikasi dari biner menjadi multikelas. Kelas Electronic memperkenalkan variabilitas visual yang tinggi, mencakup berbagai macam objek mulai dari perangkat kecil seperti ponsel dan charger hingga komponen yang lebih besar seperti keyboard atau mainboard. Kompleksitas ini, ditambah dengan ketidakseimbangan data yang inheren, membuat metrik Macro F1-Score menjadi tolok ukur yang jauh lebih ketat dan relevan dibandingkan akurasi sederhana. Oleh karena itu, meskipun model dari paper dapat direplikasi untuk memvalidasi pipeline data dan memastikan implementasi yang benar, **tidak realistis untuk mengharapkan performa yang sama tingginya secara langsung pada tugas tiga kelas tanpa modifikasi signifikan**. Langkah replikasi ini lebih berfungsi sebagai *sanity check* dan titik referensi untuk mengukur peningkatan dari pendekatan yang lebih canggih.

#### 2.1.1. Kelebihan: Validasi Cepat dan Titik Referensi yang Jelas

Salah satu keuntungan utama dari replikasi model paper adalah **kecepatan dan kesederhanaan implementasinya**. Menggunakan InceptionV3 yang sudah terlatih sebelumnya sebagai *feature extractor* adalah proses yang sangat efisien secara komputasi karena tidak memerlukan pelatihan ulang pada seluruh jaringan konvolusional. Fitur yang diekstrak (2.048 vektor fitur) kemudian dapat langsung digunakan untuk melatih classifier *shallow* seperti Support Vector Machine (SVM), K-Nearest Neighbors (KNN), atau Decision Tree dalam waktu yang sangat singkat [^37^]. Proses ini memungkinkan tim untuk dengan cepat memiliki pipeline end-to-end yang berfungsi, memvalidasi bahwa data dapat dimuat, diproses, dan diumpankan ke model dengan benar. Hasil dari replikasi ini memberikan **titik referensi yang jelas dan terukur**. Jika tim berhasil mencapai akurasi yang mendekati 96.3% pada subset data biner (Organic dan Recyclable), ini adalah indikasi kuat bahwa preprocessing data, pemisahan *train/test*, dan pipeline evaluasi sudah benar. Pencapaian baseline ini memberikan kepercayaan diri sebelum beralih ke eksperimen yang lebih kompleks dan memakan waktu. Selain itu, baseline ini menjadi tolok ukur untuk mengevaluasi seberapa besar peningkatan yang dihasilkan oleh model-model modern. Sebagai contoh, jika sebuah model Transformer hanya memberikan peningkatan marginal (misalnya, <1%) dibandingkan dengan baseline SVM, maka kompleksitas tambahan tersebut mungkin tidak sepadan, terutama dengan keterbatasan *submission*.

#### 2.1.2. Keterbatasan: Ketidakmampuan Menangani Kelas Baru dan Kompleksitas Multikelas

Meskipun replikasi model paper memberikan fondasi yang baik, keterbatasannya menjadi sangat jelas ketika dihadapkan pada persyaratan kompetisi yang sebenarnya. Keterbatasan paling mendasar adalah arsitektur yang dirancang untuk klasifikasi biner (dua kelas) tidak dapat langsung diterapkan pada tugas klasifikasi tiga kelas (Organic, Recyclable, Electronic) tanpa modifikasi substansial. Meskipun SVM dapat diperluas untuk multikelas menggunakan pendekatan seperti "one-vs-one" atau "one-vs-rest", dan jaringan neural seperti InceptionV3 dapat dimodifikasi dengan mengganti lapisan klasifikasi akhir (*top layer*) untuk menghasilkan tiga output, **pendekatan ini tidak mengatasi akar masalah kompleksitas yang ditambahkan oleh kelas Electronic**. Paper asli tidak perlu menangani tantangan visual yang ekstrem dari kelas Electronic, yang dapat mencakup objek dengan tekstur logam, plastik, atau komponen yang rumit. Model InceptionV3 + SVM mungkin berhasil membedakan gambar dengan latar belakang alami (Organik) dari gambar dengan latar belakang putih/tata letak bersih (Daur Ulang), tetapi mungkin akan kesulitan dengan kelas Electronic yang bisa memiliki variasi latar belakang dan komposisi visual yang sangat besar. Selain itu, metrik evaluasi kompetisi adalah **Macro F1-Score**, yang sangat sensitif terhadap performa pada kelas minoritas. Paper asli hanya melaporkan akurasi, yang dapat menutupi performa yang buruk pada kelas dengan jumlah sampel lebih sedikit. Oleh karena itu, mengandalkan replikasi model paper sebagai strategi utama berisiko menghasilkan skor Macro F1 yang tidak kompetitif karena model kemungkinan besar akan bias terhadap kelas mayoritas (Organic dan Recyclable) dan gagal mengklasifikasikan Electronic dengan baik.

#### 2.1.3. Potensi Overfitting pada Dua Kelas Awal

Ketika mencoba untuk mengadaptasi model dari paper untuk tugas tiga kelas, ada risiko signifikan terjadinya *overfitting* pada karakteristik spesifik dari dua kelas awal, yaitu Organic dan Recyclable. Karena dataset untuk dua kelas ini berasal dari sumber yang sama dengan paper, model mungkin dengan mudah "mengingat" atau mempelajari fitur-fitur spesifik yang telah terbukti efektif untuk membedakan keduanya. Ini bisa mencakup fitur-fitur sederhana seperti warna dominan (hijau/cokelat untuk Organik, putih/transparan untuk Daur Ulang), tekstur, atau bahkan karakteristik metadata dari gambar itu sendiri. Jika model terlalu fokus pada pemisahan Organik dan Daur Ulang, ia mungkin tidak belajar representasi yang cukup general untuk mengenali kelas Electronic yang lebih beragam. Sebagai contoh, model mungkin belajar bahwa "banyak warna hijau = Organik" dan "banyak warna putih = Daur Ulang". Ketika dihadapkan dengan sebuah keyboard elektronik yang memiliki warna hitam dan abu-abu, model ini mungkin akan bingung atau salah mengklasifikasikannya. Fenomena ini, di mana model belajar fitur yang spesifik untuk dataset pelatihan tetapi tidak general untuk data baru, adalah inti dari *overfitting*. Dalam konteks kompetisi, ini berarti model mungkin mencapai akurasi tinggi pada data *training* atau *validation* yang memiliki proporsi kelas serupa, tetapi akan gagal secara dramatis pada *test set* yang mungkin memiliki distribusi atau variasi gambar yang sedikit berbeda. Untuk menghindari ini, diperlukan **strategi regularisasi yang kuat**, augmentasi data yang agresif, dan arsitektur yang dirancang untuk menangkap fitur-fitur yang lebih invarian dan hierarkis, yang umumnya merupakan kekuatan model-model modern.

### 2.2. Analisis Pendekatan Model Modern

#### 2.2.1. Kandidat Model: EVA-02, Swin V2, EfficientViT, ConvNeXt

Beralih dari pendekatan tradisional, penggunaan arsitektur deep learning modern yang telah terbukti superior dalam berbagai tugas computer vision adalah strategi yang paling masuk akal untuk mengejar peringkat teratas. Model-model ini dirancang dengan mekanisme yang lebih canggih untuk menangkap fitur-fitur visual yang kompleks dan hierarkis. Berikut adalah analisis beberapa kandidat model yang kuat:

*   **ConvNeXt V2:** Model ini merupakan hasil evolusi dari jaringan konvolusional (ConvNet) tradisional yang secara sadar mengadopsi prinsip-prinsip desain dari Vision Transformer (ViT) sambil mempertahankan efisiensi ConvNet. ConvNeXt V2, misalnya, menggunakan *depthwise convolution* dengan kernel besar (7x7) dan *inverted bottleneck* yang terinspirasi dari Swin Transformer dan MobileNetV2. Keunggulannya terletak pada **efisiensi yang sangat tinggi baik dalam hal parameter maupun FLOPs**, sambil tetap memberikan akurasi yang sangat kompetitif. Tersedia dalam berbagai varian ukuran (Atto, Femto, Pico, Nano, Tiny, Base, Large, Huge), memberikan fleksibilitas besar untuk menyesuaikan kapasitas model dengan sumber daya komputasi yang tersedia. Untuk kompetisi ini, varian seperti **ConvNeXt V2-Tiny (28.6M parameter, 82.94% ImageNet Top-1 Acc.) atau ConvNeXt V2-Base (88.7M parameter, 84.87% Acc.)** adalah pilihan yang sangat menarik karena menawarkan keseimbangan sempurna antara performa dan efisiensi [^105^].

*   **Swin Transformer V2:** Sebagai model berbasis Transformer hierarkis, Swin V2 unggul dalam menangkap hubungan jangka panjang (long-range dependencies) dalam gambar melalui mekanisme *self-attention*. Berbeda dengan ViT standar yang menerapkan attention pada seluruh gambar (yang sangat mahal secara komputasi), Swin Transformer membagi gambar menjadi *windows* dan menerapkan *shifted window attention*, yang secara drastis mengurangi kompleksitas komputasi sambil tetap memungkinkan interaksi antar wilayah gambar. Kemampuannya untuk memahami konteks global dan dependensi spasial yang kompleks menjadikannya pilihan yang sangat kuat untuk tugas di mana konteks penting, seperti membedakan objek yang tumpang tindih atau memahami hubungan antar bagian dari suatu barang elektronik.

*   **EfficientNetV2:** Merupakan penerus dari keluarga EfficientNet yang terkenal, EfficientNetV2 dirancang dengan fokus utama pada **kecepatan pelatihan dan efisiensi**. Menggunakan *training-aware Neural Architecture Search (NAS)*, model ini tidak hanya mengoptimalkan akurasi tetapi juga waktu yang dibutuhkan untuk mencapai akurasi tersebut. Ini dicapai melalui penggunaan *Fused-MBConv* dan strategi *progressive learning* yang menyesuaikan ukuran gambar dan regularisasi secara dinamis selama pelatihan [^75^]. EfficientNetV2 secara konsisten mengungguli EfficientNetV1 dan bahkan banyak model Transformer dalam hal kecepatan pelatihan dan akurasi pada berbagai benchmark [^114^]. Varian seperti **EfficientNetV2-S (21.5M parameter, 83.9% Top-1 Acc.) atau EfficientNetV2-M (54.1M parameter, 85.2% Acc.)** adalah pilihan yang sangat solid dan efisien [^80^].

*   **EVA-02:** Model ini mewakili generasi terbaru dari Vision Transformer yang dilatih dengan skala besar dan strategi pre-training yang canggih. EVA-02 dirancang untuk menjadi *foundation model* yang sangat kuat dan menunjukkan performa *transfer learning* yang luar biasa pada berbagai *downstream tasks*. Keunggulannya terletak pada **kemampuan generalisasi yang sangat tinggi** karena dilatih pada dataset yang sangat besar dengan pendekatan *masked image modeling*. Meskipun seringkali memiliki ukuran yang lebih besar, performanya yang dekat dengan *state-of-the-art* menjadikannya kandidat yang patut dipertimbangkan jika sumber daya komputasi memungkinkan.

#### 2.2.2. Kelebihan: Kapasitas Representasi yang Lebih Kaya untuk Ketiga Kelas

Keunggulan fundamental dari model-model modern ini terletak pada **kapasitas representasi yang jauh lebih kaya dan hierarkis** dibandingkan dengan InceptionV3 yang digunakan dalam paper. InceptionV3, meskipun inovatif pada masanya, memiliki mekanisme untuk menangkap fitur yang relatif lebih lokal dan tidak seefisien model-model baru dalam membangun representasi global. Sebaliknya, model seperti ConvNeXt dan Swin Transformer dirancang untuk mempelajari hierarki fitur yang mendalam: dari fitur tingkat rendah (tepi, sudut, tekstur) di lapisan awal, hingga fitur tingkat tinggi yang lebih abstrak (bentuk bagian, konteks objek) di lapisan yang lebih dalam. Kemampuan ini sangat penting untuk menangani kompleksitas kelas Electronic. Sebuah model modern dapat belajar bahwa keberadaan baris-baris paralel kecil (tombol pada keyboard), bentuk silinder (baterai), atau permukaan reflektif dengan sirkuit (mainboard) adalah fitur yang mendefinisikan kelas Electronic, terlepas dari variasi besar dalam penampilan keseluruhan objek. Selain itu, mekanisme *self-attention* dalam Transformer memungkinkan model untuk fokus pada bagian-bagian gambar yang paling relevan untuk klasifikasi. Misalnya, saat melihat gambar sampah elektronik yang berantakan, model dapat "menyoroti" area yang mengandung informasi paling penting (sebuah ponsel atau charger) dan mengabaikan latar belakang atau objek yang tidak relevan. Kemampuan untuk fokus secara selektif ini sangat berharga untuk meningkatkan akurasi, terutama pada gambar yang kompleks atau memiliki noise visual.

#### 2.2.3. Tantangan: Kebutuhan Komputasi dan Risiko Overfitting pada Data Terbatas

Meskipun menawarkan performa yang unggul, penggunaan model modern tidak tanpa tantangan. Tantangan utama adalah **kebutuhan komputasi yang signifikan**. Model seperti Swin Transformer atau EVA-02, terutama dalam varian besar (Base, Large), memerlukan GPU dengan memori video (VRAM) yang besar dan waktu pelatihan yang lebih lama. Ini bisa menjadi kendala jika tim memiliki akses terbatas ke sumber daya komputasi. Meskipun model seperti EfficientNetV2 dan ConvNeXt dirancang untuk lebih efisien, pelatihan mereka tetap membutuhkan waktu dan sumber daya yang jauh lebih banyak dibandingkan dengan melatih SVM pada fitur yang telah diekstrak. Tantangan kedua, dan yang lebih kritis, adalah **risiko overfitting**. Dataset kompetisi, dengan 26.527 gambar, tidak termasuk dalam kategori "sangat besar" dalam standar deep learning modern. Model-model state-of-the-art memiliki kapasitas (jutaan hingga miliaran parameter) yang sangat besar untuk mempelajari dataset ini secara hafalan. Tanpa strategi regularisasi yang kuat, augmentasi data yang agresif, dan teknik seperti *early stopping*, model modern ini dapat dengan mudah *overfit* pada data training, mencapai akurasi mendekati 100% pada training set, tetapi performanya akan menurun drastis pada test set. Oleh karena itu, keputusan untuk menggunakan model modern harus diiringi dengan komitmen untuk menerapkan praktik terbaik dalam regularisasi dan augmentasi data. Keberhasilan dengan model modern tidak hanya bergantung pada pemilihan arsitektur, tetapi juga pada kecakapan dalam mencegah overfitting.

### 2.3. Perbandingan Langsung: Akurasi, Generalisasi, dan Kompleksitas

Keputusan antara mereplikasi model paper dan menggunakan arsitektur modern dapat diilustrasikan dengan jelas melalui perbandingan langsung dalam beberapa dimensi kunci. Pendekatan paper (InceptionV3 + SVM) menawarkan jalan yang cepat dan efisien menuju baseline yang terhormat, tetapi dengan langit-langit performa yang terbatas dan ketidakmampuan inheren untuk menangani kompleksitas tugas multikelas secara optimal. Sebaliknya, model modern menyajikan jalan yang lebih menantang tetapi dengan potensi hasil yang jauh lebih besar.

| Aspek | Replikasi Paper (InceptionV3 + SVM) | Pendekatan Modern (e.g., ConvNeXt, EfficientNetV2) |
| :--- | :--- | :--- |
| **Potensi Akurasi** | **Terbatas (~88-92% Macro F1 perkiraan).** Didasarkan pada akurasi 96.3% untuk 2 kelas. Performa pada kelas Electronic yang kompleks kemungkinan akan menurunkan Macro F1 secara signifikan karena model tidak dioptimalkan untuk variabilitas visual yang tinggi dan ketidakseimbangan kelas. | **Sangat Tinggi (>95% Macro F1 potensial).** Dirancang untuk menangkap representasi yang kompleks dan hierarkis, yang secara teoritis lebih mampu membedakan nuansa visual antara kelas Recyclable dan Electronic. |
| **Generalisasi** | **Rendah.** Model dapat belajar fitur spesifik dan "shortcut" dari data 2 kelas, seperti warna dominan. Hal ini menyebabkan model gagal ketika dihadapkan dengan variasi baru, terutama dari kelas Electronic yang tidak terduga. | **Tinggi.** Mekanisme seperti *shifted window attention* (Swin) atau *depthwise convolution* dengan kernel besar (ConvNeXt) memungkinkan model untuk belajar fitur yang lebih invarian dan kontekstual, yang esensial untuk generalisasi yang baik pada data uji. |
| **Kompleksitas & Sumber Daya** | **Rendah.** *Feature extraction* dengan model beku sangat cepat. Pelatihan SVM adalah proses yang jauh lebih sederhana dan kurang intensif komputasi dibandingkan dengan *fine-tuning* jaringan neural secara end-to-end. | **Tinggi.** *Fine-tuning* model modern memerlukan GPU dengan VRAM yang besar, waktu pelatihan yang lebih lama, dan keahlian dalam hyperparameter tuning (learning rate, weight decay, augmentation policies). |
| **Waktu Pengembangan** | **Singkat.** Implementasi dapat diselesaikan dalam beberapa jam, yang memungkinkan validasi cepat terhadap pipeline data. | **Lebih Lama.** Memerlukan waktu untuk eksperimen dengan berbagai arsitektur, ukuran model, dan strategi training untuk menemukan kombinasi yang optimal. |
| **Inovasi** | **Minimal.** Tidak ada kontribusi baru yang signifikan. Performa akan stagnan pada level yang ditentukan oleh paper asli. | **Maksimal.** Menawarkan banyak ruang untuk inovasi melalui kombinasi arsitektur, teknik augmentasi, strategi optimasi, dan metode ensemble. |

*Tabel 1: Perbandingan Strategis antara Replikasi Paper dan Pendekatan Modeling Modern.*

Berdasarkan perbandingan ini, menjadi jelas bahwa **pendekatan modern adalah pilihan yang paling rasional untuk tujuan kompetitif**. Meskipun lebih menantang, potensi untuk mencapai skor Macro F1 yang superior jauh melebihi risiko dan biaya komputasinya. Replikasi paper sebaiknya dianggap sebagai latihan awal untuk membangun kepercayaan diri, bukan sebagai strategi final.

## 3. Strategi Transfer Learning dan Fine-tuning Optimal

Transfer learning adalah pilar utama dalam mencapai performa tinggi pada kompetisi computer vision, terutama ketika dataset pelatihan tidak sebesar ImageNet. Strategi ini melibatkan pemanfaatan pengetahuan yang telah dipelajari oleh model yang telah dilatih sebelumnya pada dataset besar (seperti ImageNet) dan mengadaptasikannya untuk tugas spesifik yang dihadapi.

### 3.1. Fine-tuning dari Pretrained ImageNet: Keharusan Mutlak

#### 3.1.1. Manfaat Bobot Awal yang Kaya Fitur

Memulai pelatihan dengan bobot yang telah dilatih sebelumnya pada ImageNet adalah praktik standar dan sangat penting. ImageNet, dengan 1,2 juta gambar dan 1.000 kelas, memberikan model dengan pemahaman visual yang sangat kaya tentang dunia. Bobot awal ini telah mempelajari hierarki fitur yang sangat umum dan berguna, mulai dari deteksi tepi dan sudut sederhana di lapisan awal, hingga pengenalan tekstur, pola, dan bagian objek yang lebih kompleks di lapisan yang lebih dalam. Dengan memulai dari titik ini, model tidak perlu belajar dari nol. Sebaliknya, ia hanya perlu **"menyesuaikan" (adapt) pengetahuannya yang sudah luas untuk fokus pada fitur-fitur yang paling relevan untuk tugas klasifikasi sampah**. Ini secara drastis mempercepat konvergensi dan, yang lebih penting, membantu model mencapai generalisasi yang jauh lebih baik daripada melatih dari awal (training from scratch), yang akan memerlukan dataset yang jauh lebih besar untuk mencapai performa serupa.

#### 3.1.2. Pemilihan Pre-training Dataset: ImageNet-1K vs. ImageNet-21K

Kebanyakan model yang tersedia secara publik dilatih pada salah satu dari dua versi ImageNet: ImageNet-1K (1.000 kelas, ~1,28 juta gambar) atau ImageNet-21K (21.843 kelas, ~14 juta gambar).

*   **ImageNet-1K Pre-trained:** Ini adalah pilihan yang paling umum dan umumnya sudah cukup kuat. Model yang dilatih pada dataset ini telah belajar representasi visual yang sangat baik untuk 1.000 kategori umum.
*   **ImageNet-21K Pre-trained:** Model yang dilatih pada dataset ini memiliki keuntungan tambahan karena telah "melihat" lebih banyak variasi visual dari jumlah kelas yang jauh lebih besar. Ini dapat menghasilkan bobot awal yang lebih robust dan mampu menangkap fitur yang lebih halus. Penelitian menunjukkan bahwa **fine-tuning dari model ImageNet-21K seringkali menghasilkan performa yang lebih tinggi**, terutama pada downstream tasks yang kompleks. Sebagai contoh, EfficientNetV2-L yang dilatih pada ImageNet-21K mencapai **87.3% akurasi pada ImageNet-1K**, melebihi model yang dilatih langsung pada ImageNet-1K (85.7%) [^79^].

Jika sumber daya memungkinkan, menggunakan model yang telah dilatih sebelumnya pada **ImageNet-21K adalah pilihan yang lebih unggul** dan dapat memberikan keuntungan kompetitif yang signifikan.

#### 3.1.3. Signifikansi Domain yang Sama (ImageNet → Waste Classification)

Salah satu alasan mengapa transfer learning sangat efektif dalam kasus ini adalah karena **kesamaan domain antara data pre-training (ImageNet) dan data target (klasifikasi sampah)**. ImageNet berisi ribuan gambar objek sehari-hari, termasuk makanan, tanaman, botol, kertas, dan perangkat elektronik. Ini berarti model yang telah dilatih sebelumnya kemungkinan besar telah melihat fitur visual yang sangat mirip dengan yang ada di dataset kompetisi. Transfer knowledge dari domain yang sangat mirip ini jauh lebih efektif daripada jika targetnya adalah domain yang sangat berbeda, seperti gambar medis atau satelit. Lapisan konvolusional awal dari model, yang menangkap fitur umum seperti bentuk dan tekstur, akan sangat relevan dan memerlukan sedikit penyesuaian. Lapisan yang lebih dalam, yang menangkap konsep yang lebih spesifik, akan menyesuaikan diri lebih banyak untuk membedakan antara tiga kelas sampah.

### 3.2. Strategi Fine-tuning Multi-Tahap (Staged Fine-tuning)

Strategi fine-tuning yang paling efektif melibatkan pendekatan bertahap daripada melatih seluruh model sekaligus. Ini membantu menjaga fitur umum yang telah dipelajari sebelumnya sambil memungkinkan model untuk beradaptasi secara bertahap.

#### 3.2.1. Tahap 1: Training Head Classifier dengan Backbone Beku

Langkah pertama adalah memanfaatkan model pre-trained sebagai *feature extractor* yang tetap. Kita akan membekukan (freeze) semua lapisan *backbone* konvolusional dan hanya melatih lapisan *head* (lapisan klasifikasi) yang baru saja ditambahkan dan diinisialisasi secara acak. Lapisan *head* ini biasanya terdiri dari satu atau lebih lapisan fully connected (FC) diikuti oleh lapisan output untuk tiga kelas.

*   **Proses:** Data gambar dilewatkan melalui backbone yang beku untuk menghasilkan vektor fitur. Vektor-vektor fitur ini kemudian digunakan untuk melatih lapisan head.
*   **Tujuan:** Tahap ini memungkinkan *head* untuk belajar memetakan fitur-fitur yang sangat informatif yang dihasilkan oleh backbone ke tiga kelas target. Karena backbone tidak diperbarui, fitur-fitur umum yang telah dipelajari tetap utuh. Ini adalah langkah yang relatif cepat dan membantu *head* mencapai performa yang cukup baik sebelum kita mulai menyesuaikan backbone.
*   **Durasi:** Biasanya dilakukan untuk beberapa epoch (misalnya, 5-10 epoch) sampai loss validasi mulai stabil.

#### 3.2.2. Tahap 2: Fine-tuning Bertahap dengan Unfreezing Lapisan

Setelah *head* dilatih, langkah selanjutnya adalah mulai menyesuaikan (*fine-tune*) bobot di dalam backbone itu sendiri. Namun, alih-alih segera melatih seluruh jaringan dengan learning rate yang sama, pendekatan bertahap yang lebih hati-hati dianjurkan.

1.  **Unfreeze Lapisan Atas:** Kita mulai dengan membuka kunci (unfreeze) beberapa lapisan terakhir (lapisan atas) dari backbone, yang bertanggung jawab untuk fitur-fitur yang paling spesifik dan tugas-dependen. Lapisan-lapisan awal tetap dibekukan. Model kemudian dilatih dengan learning rate yang sangat kecil (misalnya, 1e-5).
2.  **Unfreeze Lebih Banyak Lapisan:** Secara bertahap, kita dapat membuka kunci lebih banyak lapisan di bawahnya dan melanjutkan pelatihan dengan learning rate yang masih kecil.
*   **Logika:** Lapisan-lapisan awal menangkap fitur umum (garis, tepi) yang relevan untuk hampir semua tugas visi komputer. Lapisan-lapisan yang lebih dalam menangkap fitur yang lebih spesifik untuk dataset pre-training (ImageNet). Dengan membekukan lapisan awal, kita memastikan bahwa pengetahuan umum ini tidak terdistorsi. Hanya lapisan-lapisan yang lebih dalam yang diperbolehkan untuk beradaptasi untuk menangkap nuansa spesifik dari dataset sampah.

#### 3.2.3. Tahap 3: Fine-tuning End-to-End dengan Learning Rate Kecil

Tahap akhir adalah *fine-tuning* end-to-end, di mana seluruh model, dari lapisan pertama hingga terakhir, dibuka kuncinya dan dilatih bersama-sama. Namun, ini harus dilakukan dengan sangat hati-hati.

*   **Learning Rate yang Sangat Kecil:** Learning rate untuk seluruh model harus ditetapkan ke nilai yang sangat kecil (misalnya, 1e-6 atau 1e-7). Ini penting untuk memastikan bahwa pembaruan bobot pada lapisan awal sangat kecil dan tidak menghancurkan fitur-fitur umum yang telah dipelajari dengan baik.
*   **Discriminative Learning Rates:** Sebuah praktik yang lebih canggih adalah menggunakan *discriminative learning rates*, di mana lapisan-lapisan yang lebih dalam (yang baru saja dibuka kuncinya) diberi learning rate yang sedikit lebih tinggi, sementara lapisan-lapisan awal diberi learning rate yang jauh lebih kecil. Ini memungkinkan penyesuaian yang lebih agresif pada bagian model yang paling perlu beradaptasi, sambil menjaga stabilitas pada bagian yang sudah optimal.
*   **Tujuan:** Tahap ini memungkinkan model untuk membuat penyesuaian halus di seluruh jaringan untuk secara kolektif mengoptimalkan performa pada tugas target, yang berpotensi membuka peningkatan performa terakhir.

### 3.3. Adaptasi Model terhadap Kelas Electronic

Kelas Electronic adalah variabel yang paling tidak diketahui dan menjadi kunci untuk memenangkan kompetisi. Strategi transfer learning harus secara eksplisit dirancang untuk membantu model beradaptasi dengan kelas ini.

#### 3.3.1. Fokus Learning pada Fitur Visual yang Mendefinisikan Elektronik

Ketika melakukan fine-tuning, model perlu belajar untuk mengidentifikasi fitur-fitur visual yang menjadi *signature* dari kelas Electronic. Ini bisa berupa:
*   **Bentuk Geometris:** Persegi panjang dengan tepi tajam (ponsel, tablet), silinder (baterai), atau bentuk organik dengan kabel yang menjulur.
*   **Tekstur Material:** Permukaan plastik halus, logam berkilau, atau sirkuit yang rumit.
*   **Konteks Objek:** Keberadaan tombol, port, konektor, atau label.
Model modern dengan kapasitas representasi yang tinggi secara inheren lebih baik dalam mempelajari kombinasi fitur-fitur kompleks ini. Fine-tuning end-to-end memberikan model kebebasan untuk menyesuaikan seluruh hierarki fiturnya untuk menjadi lebih sensitif terhadap ciri-ciri yang membedakan Electronic dari Recyclable.

#### 3.3.2. Teknik untuk Mengurangi Bias terhadap Kelas Mayoritas

Dengan jumlah sampel yang jauh lebih sedikit (3.961 vs ~22.500 untuk kelas lainnya), ada risiko besar bahwa model akan menjadi "malas" dan lebih sering memprediksi kelas mayoritas untuk meminimalkan loss secara keseluruhan. Strategi fine-tuning harus dikombinasikan dengan teknik untuk mengatasi ketidakseimbangan ini:
*   **Weighted Loss Function:** Memberikan bobot yang lebih tinggi pada loss yang berasal dari kelas Electronic, sehingga memaksa model untuk lebih memperhatikan sampel-sampel dari kelas tersebut.
*   **Oversampling dan Augmentasi:** Menampilkan gambar-gambar Electronic lebih sering selama pelatihan dan menerapkan augmentasi data yang kuat untuk menciptakan variasi sintetis, secara efektif meningkatkan jumlah dan keragaman data untuk kelas tersebut.
*   **Focal Loss:** Sebagai alternatif dari cross-entropy loss, Focal Loss secara otomatis menurunkan kontribusi sampel yang mudah diklasifikasikan (kemungkinan besar dari kelas mayoritas) dan fokus pada sampel yang sulit (kemungkinan besar dari kelas minoritas).

#### 3.3.3. Penggunaan Label Smoothing untuk Meningkatkan Generalisasi

**Label Smoothing** adalah teknik regularisasi yang mengubah *hard labels* (misalnya, [1, 0, 0] untuk kelas 0) menjadi *soft labels* (misalnya, [0.9, 0.05, 0.05]). Ini mencegah model menjadi terlalu percaya diri pada prediksinya. Dalam konteks ketidakseimbangan kelas, label smoothing sangat bermanfaat. Ini mencegah model mencapai probabilitas prediksi yang sangat tinggi untuk kelas mayoritas, yang dapat menjadi tanda overfitting. Dengan "meragukan" targetnya sedikit, model didorong untuk belajar representasi yang lebih umum dan tidak terlalu fokus pada idiosinkrasi data pelatihan. Ini dapat membantu model menjadi lebih adaptif terhadap variasi dalam kelas Electronic yang langka dan lebih robust secara keseluruhan.

## 4. Penanganan Khusus Kelas Electronic

Kelas Electronic merupakan faktor penentu utama dalam kompetisi ini. Dengan jumlah sampel yang jauh lebih sedikit dan variabilitas visual yang tinggi, bagaimana tim menangani kelas ini akan secara langsung mempengaruhi skor Macro F1 akhir.

### 4.1. Permasalahan Utama: Class Imbalance

#### 4.1.1. Analisis Distribusi Data dan Dampak pada Macro F1

Distribusi data yang tidak seimbang, seperti yang terlihat pada gambar kelas (12.567 Organic, 9.999 Recyclable, dan 3.961 Electronic), menciptakan bias yang signifikan selama pelatihan. Model cenderung untuk meminimalkan loss secara keseluruhan dengan menjadi sangat pandai dalam mengenali kelas mayoritas (Organic dan Recyclable) bahkan pada pengorbanan performa pada kelas minoritas (Electronic). Dalam konteks kompetisi yang menggunakan metrik **Macro F1-Score**, dampak dari masalah ini menjadi sangat besar. Macro F1 dihitung dengan mengambil rata-rata F1-Score dari setiap kelas secara independen. Ini berarti **performa pada kelas Electronic, meskipun jumlahnya sedikit, memiliki bobot yang sama dengan performa pada kelas Organic yang jumlahnya tiga kali lipat**. Jika model gagal mengenali kelas Electronic (misalnya, F1-Score Electronic = 0.4), skor Macro F1 akan terpukul secara signifikan, bahkan jika F1-Score untuk dua kelas lainnya mendekati sempurna. Misalnya, jika F1-Score Organic = 0.98, Recyclable = 0.97, dan Electronic = 0.4, maka Macro F1 akan menjadi (0.98 + 0.97 + 0.4) / 3 = **0.783**, yang merupakan skor yang tidak kompetitif. Oleh karena itu, mengatasi ketidakseimbangan kelas bukanlah sekadar teknik untuk meningkatkan performa, melainkan **strategi inti untuk memaksimalkan metrik evaluasi**.

#### 4.1.2. Risiko Model Bias terhadap Kelas Mayoritas (Organic, Recyclable)

Jika tidak ditangani, model yang dilatih pada data tidak seimbang akan mengalami bias yang kuat. Secara intuitif, model akan "melihat" gambar Organic dan Recyclable jauh lebih sering, sehingga akan mengoptimalkan parameter internalnya untuk mengenali pola-pola yang umum pada dua kelas tersebut. Ketika dihadapkan dengan gambar Electronic, model mungkin tidak memiliki representasi yang cukup kuat untuk membuat prediksi yang akurat. Hal ini dapat menyebabkan dua jenis kesalahan utama yang merusak F1-Score Electronic: **False Negatives** (model salah mengklasifikasikan sampah elektronik sebagai Organic atau Recyclable) dan **False Positives** (model salah mengklasifikasikan sampah organik atau daur ulang sebagai Electronic). Keduanya akan menurunkan precision dan recall untuk kelas Electronic. Dalam praktiknya, model yang tidak seimbang seringkali menunjukkan tingkat kepercayaan (confidence) yang sangat rendah untuk kelas minoritas atau bahkan sama sekali tidak pernah memprediksi kelas minoritas tersebut. Analisis confusion matrix pada data validasi akan dengan jelas mengungkapkan bias ini, di mana baris dan kolom untuk kelas Electronic akan menunjukkan angka kesalahan yang tinggi. Mengatasi masalah ini memerlukan pendekatan yang bertujuan untuk memberikan perhatian lebih pada kelas Electronic selama proses pelatihan.

### 4.2. Strategi Sampling yang Efektif

Salah satu cara paling efektif untuk menangani ketidakseimbangan kelas adalah dengan mengontrol bagaimana data disajikan kepada model selama pelatihan. Alih-alih mengambil sampel secara acak dari seluruh dataset (yang akan menghasilkan batch yang didominasi oleh kelas mayoritas), kita dapat menggunakan strategi sampling yang lebih cerdas.

#### 4.2.1. Class-Balanced Sampling: Memastikan Representasi Setiap Kelas dalam Setiap Batch

Teknik ini, sering disebut sebagai **Class-Balanced Sampling** atau **Weighted Random Sampling**, bertujuan untuk memastikan bahwa setiap mini-batch yang diberikan kepada model selama pelatihan memiliki representasi yang lebih seimbang dari setiap kelas. Cara kerjanya adalah dengan memberikan probabilitas pengambilan sampel yang lebih tinggi pada kelas minoritas. Misalnya, jika kita memiliki tiga kelas dengan jumlah sampel N_organic, N_recyclable, dan N_electronic, probabilitas pengambilan sampel untuk setiap gambar dapat dihitung secara berbeda. Salah satu metode yang umum adalah memberikan bobot untuk setiap kelas yang berbanding terbalik dengan frekuensinya. Sebagai contoh, bobot untuk kelas `c` bisa dihitung sebagai `W_c = Total_Samples / (Num_Classes * N_c)`. Kemudian, setiap gambar dari kelas `c` akan memiliki probabilitas untuk dipilih yang sebanding dengan `W_c`. Implementasi ini secara efektif **"menggandakan" atau "menaikkan frekuensi" kemunculan gambar dari kelas Electronic dalam proses pelatihan**. Meskipun ini tidak menambah data baru, ini memaksa model untuk melihat dan belajar dari kelas Electronic dengan frekuensi yang sama dengan kelas lainnya, yang secara langsung mengatasi masalah bias. Di PyTorch, ini dapat diimplementasikan dengan menggunakan `WeightedRandomSampler` [^90^].

#### 4.2.2. Oversampling Kelas Electronic dengan Augmentasi Data

Oversampling adalah teknik di mana kita secara aktif menambah jumlah sampel dari kelas minoritas. Bentuk oversampling yang paling sederhana adalah dengan menggandakan gambar-gambar yang sudah ada dari kelas Electronic. Namun, teknik ini memiliki risiko overfitting karena model akan melihat gambar yang identik berulang kali. Pendekatan yang jauh lebih superior adalah **menggabungkan oversampling dengan augmentasi data**. Setiap kali gambar Electronic dipilih untuk di-oversample, alih-alih menggunakan gambar asli, kita dapat menerapkan serangkaian transformasi acak untuk menciptakan versi baru yang sedikit berbeda dari gambar tersebut. Transformasi ini dapat mencakup rotasi, flipping, perubahan kecerahan/kontras/warna, zoom, dan lainnya. Dengan melakukan ini, kita secara efektif memperluas keragaman dataset untuk kelas Electronic, membantu model untuk belajar fitur-fitur yang lebih invarian terhadap variasi tersebut dan mengurangi risiko overfitting. Strategi ini secara sintetis menyeimbangkan dataset, memberikan model lebih banyak "pengalaman" dalam mengenali berbagai bentuk dan kondisi sampah elektronik.

#### 4.2.3. Undersampling Kelas Mayoritas sebagai Alternatif

Undersampling adalah pendekatan kebalikan dari oversampling, di mana kita secara acak menghapus sampel dari kelas mayoritas untuk menyeimbangkan jumlahnya dengan kelas minoritas. Meskipun sederhana, strategi ini umumnya **tidak direkomendasikan** untuk dataset yang tidak terlalu besar seperti yang ada dalam kompetisi ini. Menghapus data dari kelas Organic dan Recyclable berarti membuang informasi berharga yang dapat membantu model belajar representasi yang lebih baik untuk kelas-kelas tersebut. Dengan ukuran dataset yang terbatas, setiap sampel data berharga, dan membuangnya dapat menyebabkan underfitting pada kelas mayoritas dan penurunan performa keseluruhan. Oleh karena itu, **oversampling (dengan augmentasi) kelas minoritas secara signifikan lebih disukai** daripada undersampling kelas mayoritas dalam skenario ini.

### 4.3. Manipulasi Fungsi Loss

Selain mengubah cara data disajikan, kita juga dapat memanipulasi fungsi loss untuk membuat model lebih "perhatian" terhadap kelas minoritas. Fungsi loss standar, seperti Cross-Entropy Loss, memberikan kontribusi yang sama pada setiap sampel. Kita dapat mengubahnya untuk memberikan penalti yang lebih besar pada kesalahan yang dibuat pada kelas Electronic.

#### 4.3.1. Weighted Cross-Entropy Loss: Memberikan Bobot Lebih pada Kelas Minoritas

Ini adalah teknik yang paling umum dan efektif. Implementasinya sangat mudah. Di PyTorch, `nn.CrossEntropyLoss` menerima argumen `weight`. Argumen ini adalah vektor satu dimensi (tensor) yang panjangnya sama dengan jumlah kelas, di mana setiap elemen berisi bobot untuk kelas yang sesuai [^90^]. Logika di balik pemilihan bobot ini adalah untuk memberikan penalti yang lebih besar pada kesalahan yang dibuat untuk kelas minoritas. Jika model salah mengklasifikasikan sampah elektronik, fungsi loss akan menghasilkan nilai yang lebih tinggi, yang pada gilirannya akan menghasilkan gradien yang lebih besar selama backpropagation, memaksa model untuk lebih banyak belajar dari kesalahan tersebut. Bobot untuk setiap kelas dapat dihitung dengan berbagai cara, misalnya **berbanding terbalik dengan frekuensi kelas**. Jika kelas Electronic memiliki jumlah sampel 3.961, kelas Organic 12.567, dan kelas Recyclable 9.999, maka bobot untuk kelas Electronic akan jauh lebih tinggi. Fungsi `compute_class_weight` dari library scikit-learn dapat digunakan untuk menghitung bobot ini secara otomatis [^95^].

#### 4.3.2. Focal Loss: Fokus pada Sampel yang Sulit Diklasifikasikan

Focal Loss adalah alternatif dari Cross-Entropy Loss yang dirancang khusus untuk menangani masalah ketidakseimbangan kelas secara ekstrem, terutama dalam deteksi objek. Intuisi di balik Focal Loss adalah untuk **secara otomatis menurunkan bobot loss untuk sampel yang mudah diklasifikasikan** (misalnya, gambar organik yang sangat jelas) dan **fokus pada sampel yang sulit** (misalnya, perangkat elektronik yang bentuknya tidak biasa). Ini dicapai dengan menambahkan faktor modulasi `(1 - p_t)^gamma` ke cross-entropy standar, di mana `p_t` adalah probabilitas prediksi model untuk kelas yang benar, dan `gamma` adalah hyperparameter fokus (biasanya diatur antara 2 dan 5) [^111^]. Ketika `p_t` mendekati 1 (model sangat percaya diri dan benar), faktor modulasi mendekati 0, sehingga loss untuk sampel tersebut menjadi sangat kecil. Sebaliknya, ketika `p_t` rendah (model tidak yakin atau salah), faktor modulasi mendekati 1, dan loss tetap tinggi. Ini secara inheren mengatasi masalah ketidakseimbangan tanpa perlu menetapkan bobot kelas secara manual, karena model akan secara dinamis belajar untuk "mengabaikan" kelas mayoritas yang mudah dan "fokus" pada kelas minoritas yang sulit [^95^].

#### 4.3.3. Kombinasi Focal Loss dengan Class Weighting untuk Penanganan Agresif

Untuk tantangan ketidakseimbangan yang signifikan seperti dalam kompetisi ini, pendekatan yang paling agresif dan potensially paling efektif adalah dengan menggabungkan Focal Loss dengan class weighting. Implementasi ini memungkinkan kita untuk tidak hanya fokus pada sampel yang sulit secara dinamis tetapi juga memberikan penalti dasar yang lebih besar pada kesalahan untuk kelas Electronic secara eksplisit. Rumus untuk Focal Loss dengan class weighting dapat ditulis sebagai: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`, di mana `alpha_t` adalah bobot untuk kelas target. Di PyTorch, ini dapat diimplementasikan dengan menggabungkan `FocalLoss` dengan `alpha` yang tidak seragam, di mana `alpha` untuk kelas Electronic diatur lebih tinggi daripada kelas lainnya [^95^]. Kombinasi ini memberikan dua mekanisme komplementer untuk menangani ketidakseimbangan: `alpha` menangani masalah frekuensi kelas secara global, sementara `(1-p_t)^gamma` menangani kesulitan per-sampel secara lokal. Pendekatan ini paling direkomendasikan untuk memastikan model memberikan perhatian maksimal pada kelas Electronic.

### 4.4. Augmentasi Data yang Ditargetkan

Augmentasi data adalah praktik standar dalam deep learning untuk meningkatkan keragaman data pelatihan dan mencegah overfitting. Untuk kompetisi ini, augmentasi data dapat dirancang secara khusus untuk membantu model dalam mengenali kelas Electronic yang kompleks.

#### 4.4.1. Pola Augmentasi untuk Meningkatkan Variabilitas Sampel Elektronik

Karena kelas Electronic memiliki variabilitas visual yang tinggi dan jumlah sampel yang terbatas, augmentasi data harus diterapkan dengan agresif pada kelas ini. Tujuannya adalah untuk mensimulasikan berbagai kondisi pengambilan gambar yang mungkin ditemui dalam aplikasi dunia nyata. Ini termasuk:
*   **Geometri:** Rotasi acak dalam rentang besar (misalnya, -45 hingga +45 derajat), *shearing*, dan zoom acak. Ini membantu model mengenali objek elektronik dari berbagai sudut pandang dan jarak.
*   **Fotometrik:** Perubahan signifikan pada kecerahan, kontras, saturasi, dan *hue*. Sampah elektronik dapat ditemukan di berbagai kondisi pencahayaan, dari terang di luar ruangan hingga redup di dalam ruangan. Model harus belajar untuk tidak bergantung pada kondisi pencahayaan spesifik.
*   **Noise:** Menambahkan noise Gaussian atau *salt-and-pepper noise* dapat membantu model menjadi lebih robust terhadap artefak kompresi gambar atau sensor noise yang umum terjadi pada gambar yang diambil dengan perangkat seluler.
Dengan menerapkan augmentasi ini secara agresif, kita secara efektif memperbesar ruang sampel untuk kelas Electronic, membantu model untuk belajar fitur-fitur yang lebih invarian dan generalisasi yang lebih baik.

#### 4.4.2. CutMix dan Mixup untuk Menciptakan Sampel Sintetis yang Kompleks

Teknik augmentasi modern seperti **CutMix** dan **Mixup** melampaui transformasi sederhana dan menciptakan sampel baru dengan cara menggabungkan dua gambar yang ada.
*   **Mixup:** Secara linier menginterpolasi dua gambar dan labelnya. Misalnya, `x_mix = lambda * x1 + (1 - lambda) * x2` dan `y_mix = lambda * y1 + (1 - lambda) * y2`. Ini mendorong model untuk berperilaku secara linier di antara kelas-kelas, yang merupakan indikasi generalisasi yang baik.
*   **CutMix:** Secara acak memotong sebuah *patch* dari satu gambar dan menempelkannya ke gambar lain. Label untuk gambar baru ini dibobotkan secara proporsional terhadap area *patch* yang dipotong.
Teknik-teknik ini sangat kuat karena mereka menciptakan sampel pelatihan yang sangat menantang dan tidak mungkin terjadi dalam dataset asli. Mereka mendorong model untuk belajar fitur yang lebih lokal dan diskriminatif, karena model tidak dapat lagi mengandalkan konteks global yang familiar. Misalnya, sebuah papan sirkuit (Electronic) yang ditempelkan pada sepotong daun (Organic) akan memaksa model untuk fokus pada tekstur dan bentuk sirkuit itu sendiri untuk membuat prediksi yang benar. Ini sangat bermanfaat untuk membantu model membedakan kelas-kelas yang mungkin memiliki latar belakang atau konteks yang serupa.

#### 4.4.3. RandAugment untuk Otomatisasi Pencarian Kebijakan Augmentasi Optimal

Salah satu tantangan dalam menggunakan augmentasi data adalah memilih parameter yang tepat untuk setiap transformasi. Pencarian manual dapat memakan waktu dan tidak optimal. **RandAugment** adalah metode yang secara signifikan menyederhanakan proses ini. Alih-alih memiliki pipeline augmentasi yang kompleks dengan banyak transformasi yang masing-masing memiliki parameter yang perlu diatur, RandAugment hanya memerlukan dua hyperparameter: `N` (jumlah transformasi yang diterapkan secara berurutan pada setiap gambar) dan `M` (magnitudo untuk semua transformasi). Ia secara acak memilih `N` transformasi dari serangkaian operasi standar (seperti terjemahan, rotasi, shearing, perubahan warna, posterizing, solarizing, dll.) dan menerapkannya dengan magnitudo `M`. Penelitian telah menunjukkan bahwa RandAugment, dengan pengaturan default yang sederhana, seringkali **dapat mencapai atau melampaui performa dari metode pencarian kebijakan augmentasi yang jauh lebih kompleks** seperti AutoAugment. Menggunakan RandAugment dapat menghemat banyak waktu pengembangan dan memastikan bahwa model dilatih dengan variasi data yang kaya dan efektif.

## 5. Strategi Training untuk Performa Maksimal

Mencapai performa maksimal dalam kompetisi deep learning memerlukan lebih dari sekadar arsitektur model yang baik. Ini memerlukan orkestrasi yang cermat dari berbagai komponen: schedule pembelajaran, augmentasi data, regularisasi, dan strategi optimasi. Setiap komponen ini berinteraksi satu sama lain, dan penyetelan kolektif mereka seringkali menjadi pembeda antara model yang baik dan model yang memenangkan kompetisi.

### 5.1. Learning Rate Schedule

Pemilihan dan penjadwalan learning rate (LR) adalah salah satu hyperparameter paling penting dalam pelatihan model deep learning. LR yang terlalu tinggi dapat menyebabkan divergensi, sementara LR yang terlalu rendah dapat membuat pelatihan sangat lambat atau terjebak di minimum lokal yang buruk. Strategi modern bertujuan untuk menemukan keseimbangan yang dinamis.

#### 5.1.1. Linear Warmup untuk Stabilisasi Awal Pelatihan

Pada awal pelatihan, terutama saat menggunakan batch size yang besar atau fine-tuning model yang telah dilatih sebelumnya, bobot model dapat mengalami fluktuasi besar. Untuk mengatasi ini, teknik **Linear Warmup** digunakan. Pada fase ini, LR dimulai dari nilai yang sangat kecil (atau nol) dan secara linear ditingkatkan hingga mencapai LR awal yang diinginkan selama beberapa epoch pertama (misalnya, 5-10 epoch). Ini memungkinkan model untuk menyesuaikan diri secara bertahap dan membuat pembaruan bobot yang lebih stabil, mencegah "kejutan" besar yang dapat merusak representasi yang telah dipelajari sebelumnya. Ini sangat penting saat fine-tuning, di mana kita ingin menjaga integritas bobot pre-trained pada tahap awal.

#### 5.1.2. Cosine Annealing untuk Konvergensi ke Minimum Lokal yang Baik

Setelah fase warmup, LR tidak dipertahankan konstan. Sebaliknya, menggunakan **Cosine Annealing LR Schedule** sangat populer. Pada strategi ini, LR diatur mengikuti fungsi kosinus yang menurun dari LR maksimum ke LR minimum (yang sangat kecil atau nol) selama sisa pelatihan. Keuntungan dari penurunan ini adalah memberikan langkah-langkah pembaruan yang lebih besar di awal fase pelatihan (untuk mengeksplorasi ruang solusi secara cepat) dan langkah-langkah yang lebih kecil dan lebih halus di akhir fase pelatihan (untuk menyempurnakan bobot dan menyelesaikan diri di minimum lokal yang dangkal). Bentuk kurva kosinus yang mulus seringkali menghasilkan konvergensi yang lebih baik dibandingkan dengan penurunan LR yang lebih tiba-tiba seperti Step Decay.

#### 5.1.3. Penggunaan Learning Rate Finder untuk Menentukan LR Optimal

Menemukan LR awal yang optimal secara manual bisa menjadi proses yang membosankan. **Learning Rate Finder** adalah teknik praktis yang sangat efektif untuk mengotomatisasi ini. Prosesnya melibatkan menjalankan model untuk satu epoch sambil secara eksponensial meningkatkan LR dari nilai yang sangat kecil ke nilai yang sangat besar, dan mencatat loss. Plotting loss terhadap LR akan menunjukkan kurva yang menurun dan kemudian naik secara tajam. **LR yang optimal biasanya berada di daerah di mana loss turun paling cepat**, yang seringkali beberapa orde lebih kecil dari LR di mana loss mulai meningkat (sebelum divergensi). Menggunakan LR Finder memberikan titik awal yang sangat baik untuk LR, menghemat waktu dan mencegah penggunaan LR yang tidak optimal yang dapat merusak pelatihan.

### 5.2. Augmentasi Data yang Direkomendasikan

Augmentasi data adalah praktik standar untuk meningkatkan ukuran dan keragaman dataset pelatihan secara artifisial, yang membantu mencegah overfitting dan meningkatkan kemampuan generalisasi model. Untuk kompetisi ini, augmentasi harus diterapkan dengan kuat, terutama pada kelas Electronic.

#### 5.2.1. Standard Augmentations: RandomResizedCrop, HorizontalFlip, ColorJitter

Seperangkat transformasi dasar yang umum digunakan dan sangat efektif meliputi:
*   **RandomResizedCrop:** Secara acak memotong dan mengubah ukuran gambar. Ini membantu model belajar untuk mengenali objek terlepas dari skala dan lokasi mereka dalam bingkai.
*   **RandomHorizontalFlip:** Secara acak membalik gambar secara horizontal. Ini adalah transformasi yang sangat umum dan efektif karena sebagian besar objek dalam dataset ini akan tetap valid setelah dibalik secara horizontal.
*   **ColorJitter:** Secara acak mengubah kecerahan, kontras, saturasi, dan rona gambar. Ini membantu model menjadi invarian terhadap perubahan pencahayaan dan kondisi kamera, yang sangat penting untuk data "in-the-wild".
Ketiga transformasi ini harus menjadi bagian standar dari pipeline augmentasi apa pun.

#### 5.2.2. Advanced Augmentations: RandAugment, AutoAugment

Untuk mencapai performa terbaik, disarankan untuk menggunakan kebijakan augmentasi yang lebih canggih. Seperti yang disebutkan sebelumnya, **RandAugment** adalah pilihan yang sangat baik karena kesederhanaan dan efektivitasnya. Alih-alih mencari kebijakan augmentasi yang kompleks secara manual atau dengan reinforcement learning (seperti AutoAugment), RandAugment secara acak memilih dan menerapkan serangkaian transformasi dasar, yang terbukti sangat kuat.

#### 5.2.3. Regularization Augmentations: CutMix, Mixup, Label Smoothing

Teknik-teknik ini tidak hanya meningkatkan keragaman data tetapi juga bertindak sebagai regularisasi yang kuat:
*   **CutMix & Mixup:** Seperti dibahas sebelumnya, teknik-teknik ini menciptakan sampel baru dengan menggabungkan gambar yang ada, yang secara efektif mencegah overfitting dan mendorong model untuk belajar representasi yang lebih robust dan lokal.
*   **Label Smoothing:** Teknik ini mengubah *hard labels* menjadi *soft labels*, yang mencegah model menjadi terlalu percaya diri pada prediksinya dan mendorong generalisasi yang lebih baik.
Menggabungkan semua tingkatan augmentasi ini—dari standar hingga regularisasi—akan menciptakan pipeline pelatihan yang sangat robust.

### 5.3. Teknik Regularisasi Lanjutan

Selain augmentasi, ada teknik regularisasi lain yang dapat diterapkan langsung pada model selama pelatihan untuk mencegah overfitting.

#### 5.3.1. Label Smoothing untuk Mengurangi Overconfidence

Seperti yang dijelaskan sebelumnya, label smoothing menggantikan label one-hot (misalnya, [0, 1, 0]) dengan versi yang "dilembutkan" (misalnya, [0.05, 0.9, 0.05]). Ini adalah bentuk regularisasi yang sangat efektif dan mudah diimplementasikan. Ini mencegah model dari memprediksi probabilitas yang terlalu tinggi untuk kelas yang benar, yang merupakan tanda umum overfitting. Hyperparameter `epsilon` (misalnya, 0.1) mengontrol seberapa jauh label dari one-hot. Ini adalah salah satu trik termudah untuk meningkatkan performa dan generalisasi.

#### 5.3.2. Stochastic Depth (DropPath) untuk Model ResNet-like

**Stochastic Depth**, juga dikenal sebagai DropPath, adalah teknik regularisasi yang dirancang khusus untuk jaringan residual (ResNet) dan turunannya (seperti ConvNeXt). Selama pelatihan, blok residual secara acak "dijatuhkan" (dilewati) dengan probabilitas `p`. Ini berarti gradien aliran melalui lebih sedikit blok residual pada setiap iterasi, yang secara efektif melatih ensemble dari jaringan yang lebih dangkal secara implisit. Ini telah terbukti sangat efektif untuk melatih jaringan yang sangat dalam dan mencegah overfitting, terutama pada dataset yang lebih kecil.

#### 5.3.3. Dropout pada Lapisan Fully Connected

**Dropout** adalah teknik regularisasi klasik di mana neuron secara acak "dinonaktifkan" (outputnya diatur ke nol) selama pelatihan dengan probabilitas tertentu. Ini paling umum diterapkan pada lapisan fully connected (FC) di bagian akhir model (lapisan *head*). Dropout mencegah kodependensi yang terlalu besar antar neuron, yang memaksa model untuk belajar representasi yang lebih redundan dan generalisasi yang lebih baik. Pengaturan dropout rate (misalnya, 0.2 - 0.5) adalah hyperparameter yang penting untuk disetel.

### 5.4. Optimizer dan Hyperparameter Lainnya

Pemilihan optimizer dan hyperparameter terkaitnya memainkan peran penting dalam kecepatan konvergensi dan kualitas akhir model.

#### 5.4.1. AdamW sebagai Optimizer Pilihan Utama

**AdamW** adalah varian dari optimizer Adam yang telah menjadi standar de facto untuk melatih model vision modern. AdamW memisahkan efek regularisasi weight decay dari komputasi gradien adaptif, yang memungkinkan penyetelan weight decay yang lebih efektif. Dibandingkan dengan SGD, AdamW seringkali konvergen lebih cepat dan lebih tidak sensitif terhadap pemilihan learning rate awal, menjadikannya pilihan yang sangat baik untuk eksperimen cepat.

#### 5.4.2. Weight Decay sebagai Regularisasi Tambahan

**Weight decay** (atau L2 regularisasi) adalah teknik untuk mencegah bobot model menjadi terlalu besar. Ini dilakukan dengan menambahkan penalti pada fungsi loss yang sebanding dengan kuadrat besar bobot. Ini mendorong model untuk memiliki bobot yang lebih kecil dan tersebar, yang secara inheren membantu mencegah overfitting. AdamW memungkinkan penyetelan weight decay yang lebih efektif, dan nilai seperti `1e-2` atau `5e-2` seringkali merupakan titik awal yang baik.

#### 5.4.3. Gradient Clipping untuk Stabilitas Training

Pada beberapa kasus, terutama saat menggunakan model recurrent atau transformer, gradien dapat menjadi sangat besar dan menyebabkan instabilitas pelatihan (dikenal sebagai "exploding gradients"). **Gradient Clipping** adalah teknik untuk mencegah ini dengan menetapkan ambang batas pada norma gradien. Jika norma gradien melebihi ambang batas ini, gradien akan diskalakan sehingga normanya sama dengan ambang batas. Meskipun tidak selalu diperlukan untuk model vision, ini adalah praktik yang baik untuk stabilitas, terutama pada fase awal pelatihan atau saat bereksperimen dengan learning rate yang tinggi.

### 5.5. Early Stopping dan Model Checkpointing

Dengan jumlah submission yang sangat terbatas, sangat penting untuk mengidentifikasi model terbaik selama pelatihan tanpa harus menunggu semua epoch selesai.

#### 5.5.1. Monitoring Validation Macro F1 Score

Metrik yang harus dipantau untuk early stopping adalah **Macro F1-Score pada set validasi**. Pelatihan model harus dihentikan secara otomatis jika skor validasi ini tidak meningkat setelah sejumlah epoch tertentu (misalnya, 10 epoch). Ini mencegah overfitting pada data training.

#### 5.5.2. Penyimpanan Model Terbaik Berdasarkan Performa Validasi

Selama pelatihan, **hanya model dengan skor validasi Macro F1 tertinggi yang harus disimpan**. Praktik terbaik adalah menyimpan *checkpoint* model setiap kali skor validasi mencapai nilai tertinggi baru. Model final yang akan digunakan untuk prediksi adalah model terbaik yang telah disimpan ini, bukan model pada epoch terakhir.

#### 5.5.3. Penggunaan Exponential Moving Average (EMA) untuk Stabilisasi Bobot Model

**Exponential Moving Average (EMA)** adalah teknik di mana kita memelihara versi "halus" dari bobot model. Pada setiap langkah pelatihan, bobot EMA diperbarui sebagai rata-rata bergerak eksponensial dari bobot pelatihan saat ini: `ema_weight = decay * ema_weight + (1 - decay) * current_weight`. Bobot EMA ini cenderung lebih stabil dan seringkali menghasilkan performa generalisasi yang lebih baik daripada bobot model yang sebenarnya. Menggunakan bobot EMA untuk inferensi (prediksi) pada test set dapat memberikan peningkatan performa yang signifikan dan gratis.

## 6. Model Selection: Perbandingan Komprehensif

### 6.1. Kandidat Model Unggulan

Memilih model yang tepat adalah keputusan strategis yang menyeimbangkan kapasitas, efisiensi, dan kompleksitas. Berdasarkan analisis mendalam, berikut adalah perbandingan komprehensif antara empat kandidat unggulan.

| Model (Varian) | ImageNet-1K Top-1 Acc. | Parameters | FLOPs | Kelebihan Utama | Keterbatasan | Rekomendasi Penggunaan |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ConvNeXt V2-T** | 82.94% | 28.6M | 4.47B | Paling efisien, akurasi tinggi, desain modern [^105^] | Kapasitas mungkin terbatas untuk Electronic yang sangat kompleks | **Baseline modern, sumber daya terbatas** |
| **EfficientNetV2-S** | 83.9% | 21.5M | 8.4B | Kecepatan training & inference terbaik [^80^] | Trade-off akurasi vs kecepatan | **Eksperimen cepat, iterasi banyak** |
| **Swin V2-T** | ~81.8% | 28M | 4.5B | Mampu menangkap konteks global & jarak jauh | Biaya memori lebih tinggi | **Jika konteks spasial kompleks** |
| **ConvNeXt V2-B** | 84.87% | 88.7M | 15.4B | Kapasitas representasi tertinggi di kelasnya [^105^] | Lebih lambat, membutuhkan VRAM lebih besar | **Submission final, performa maksimal** |

*Tabel 2: Perbandingan Komprehensif Kandidat Model Unggulan.*

#### 6.1.1. ConvNeXt V2-Tiny: Efisiensi dan Performa Tinggi

ConvNeXt V2-Tiny muncul sebagai salah satu pilihan paling menarik. Dengan 28.6 juta parameter dan 4.47B FLOPs, ia menawarkan **efisiensi yang luar biasa** sambil mencapai akurasi ImageNet 82.94% [^105^]. Desainnya yang modern, yang menggabungkan kekuatan ConvNet dengan prinsip Transformer, memberinya kapasitas representasi yang kuat. Ini adalah model yang ideal untuk **membangun baseline modern yang kuat** atau jika tim memiliki keterbatasan sumber daya komputasi.

#### 6.1.2. EfficientNetV2-Small: Kecepatan Training dan Inference Terbaik

EfficientNetV2-S adalah pilihan utama jika **kecepatan adalah prioritas**. Dirancang dengan fokus pada efisiensi training, ia dapat berkonvergen jauh lebih cepat dari model lain [^75^]. Ini memungkinkan tim untuk melakukan **lebih banyak iterasi eksperimen** dalam waktu yang sama. Dengan 21.5M parameter dan akurasi ImageNet 83.9% [^80^], ia memberikan kombinasi kecepatan dan performa yang sangat kompetitif.

#### 6.1.3. Swin Transformer V2-Tiny: Kapasitas Representasi Global yang Kuat

Swin V2-Tiny unggul dalam menangkap **dependensi jangka panjang** dalam gambar. Jika tim percaya bahwa konteks global (misalnya, hubungan antara objek dalam gambar) sangat penting untuk membedakan kelas, terutama Electronic, maka Swin Transformer adalah pilihan yang kuat. Kelemahannya adalah biaya memori yang sedikit lebih tinggi, yang mungkin membatasi ukuran batch.

#### 6.1.4. ConvNeXt V2-Base: Kapasitas Lebih Besar untuk Potensi Akurasi Lebih Tinggi

Jika sumber daya komputasi tidak terlalu menjadi masalah, **ConvNeXt V2-Base (88.7M parameter, 84.87% akurasi)** [^105^] adalah pilihan yang jelas untuk **submission final**. Kapasitas representasi yang lebih besar memberinya potensi untuk mengekstrak fitur yang lebih halus dan kompleks, yang dapat menjadi kunci untuk membedakan kelas Electronic yang paling sulit.

### 6.2. Model yang Kurang Direkomendasikan

Model-model berikut, meskipun populer, mungkin kurang optimal untuk kompetisi ini karena alasan yang akan dijelaskan.

#### 6.2.1. MobileViT v2: Optimasi Mobile yang Tidak Diperlukan

MobileViT dirancang khusus untuk perangkat seluler dengan sumber daya terbatas. Karena kompetisi ini tidak memberlakukan batasan pada ukuran model atau kecepatan inference, **mengorbankan kapasitas model untuk efisiensi mobile tidak memberikan keuntungan apa pun** dan kemungkinan besar akan menghasilkan performa yang lebih rendah.

#### 6.2.2. EfficientFormer V2: Performa yang Mungkin Kurang Kompetitif

Meskipun EfficientFormer adalah arsitektur yang menarik, dalam banyak benchmark, ia masih tertinggal dari EfficientNetV2 dan ConvNeXt dalam hal trade-off akurasi dan efisiensi. Menggunakannya akan menjadi pilihan yang kurang optimal kecuali ada alasan spesifik yang kuat.

#### 6.2.3. EdgeNeXt: Fokus pada Edge Computing, Bukan Akurasi Maksimal

Sama seperti MobileViT, EdgeNeXt dioptimalkan untuk perangkat edge. Tujuannya adalah efisiensi ekstrem, bukan akurasi maksimal. Untuk kompetisi di mana tujuan utamanya adalah skor tertinggi, ini bukan pilihan yang tepat.

### 6.3. Trade-off Akurasi, Generalisasi, Kompleksitas, dan Waktu

Keputusan akhir harus didasarkan pada keseimbangan yang cermat dari trade-off berikut:

#### 6.3.1. Analisis Trade-off untuk Setiap Kandidat Model

Pertanyaan kuncinya adalah: "Berapa banyak peningkatan akurasi yang kita dapatkan untuk setiap peningkatan kompleksitas?"
*   **ConvNeXt V2-T vs. V2-B:** Perbedaan akurasi pada ImageNet adalah sekitar 2% (82.94% vs 84.87%). Apakah peningkatan 2% ini signifikan pada dataset kompetisi? Dalam kompetisi yang ketat, **setiap pecahan persen bisa menjadi pembeda**. Jika waktu dan sumber daya memungkinkan, upaya untuk menggunakan varian Base layak untuk dipertimbangkan untuk submission akhir.
*   **ConvNeXt V2-T vs. EfficientNetV2-S:** EfficientNetV2-S sedikit lebih akurat dan jauh lebih cepat, tetapi ConvNeXt V2-T lebih efisien dalam FLOPs. Pilihan di antara keduanya mungkin bergantung pada spesifikasi hardware (apakah bottleneck-nya di waktu atau di VRAM).

#### 6.3.2. Rekomendasi Berdasarkan Keterbatasan Waktu dan Sumber Daya

Dengan hanya **tiga submission** dan waktu yang terbatas, strategi yang paling pragmatis adalah:
1.  **Fase Eksperimen:** Gunakan **EfficientNetV2-S** karena kecepatan pelatihannya. Ini memungkinkan untuk iterasi cepat pada augmentasi, loss function, dan hyperparameter lainnya.
2.  **Fase Finalisasi:** Setelah pipeline dan hyperparameter terbaik ditemukan, latih model **ConvNeXt V2-Base** dengan konfigurasi terbaik tersebut. Gunakan model ini untuk **submission pertama atau kedua**.
3.  **Fase Ensemble:** Jika masih ada sisa submission, gunakan untuk submission hasil **ensemble** dari beberapa model terbaik (misalnya, ensemble dari ConvNeXt V2-B dan Swin V2-T).

## 7. Competition Strategy: Roadmap Efisien untuk 3 Submission

### 7.1. Submission 1: Baseline yang Kuat dan Dapat Diandalkan

Tujuan dari submission pertama adalah untuk membangun fondasi yang kuat dan mendapatkan umpan balik awal dari leaderboard.

#### 7.1.1. Tujuan: Menetapkan Fondasi dan Memvalidasi Strategi

Submission ini bertujuan untuk menvalidasi seluruh pipeline, dari preprocessing hingga inference, dan mendapatkan skor baseline yang kompetitif. Ini adalah *proof of concept* bahwa strategi modern bekerja pada dataset ini.

#### 7.1.2. Implementasi: Single Best Model (ConvNeXt V2 atau EfficientNetV2)

Gunakan model terbaik yang telah dilatih selama fase eksperimen, kemungkinan besar varian **ConvNeXt V2-Base** atau **EfficientNetV2-S** yang telah difine-tune dengan cermat. Pastikan untuk menggunakan semua teknik terbaik: augmentasi kuat, weighted loss, dan cosine annealing LR.

#### 7.1.3. Teknik: Fine-tuning Penuh dengan Augmentasi Standar

Pastikan model ini adalah hasil dari fine-tuning penuh dari model pre-trained ImageNet-21K dengan pipeline augmentasi yang komprehensif (RandAugment, CutMix, Mixup). Simpan checkpoint model terbaik berdasarkan validasi Macro F1.

### 7.2. Submission 2: Peningkatan Melalui Ensemble

Submission kedua harus bertujuan untuk meningkatkan skor dari submission pertama dengan menggunakan kekuatan kolektif dari beberapa model.

#### 7.2.1. Tujuan: Meningkatkan Skor dengan Ensemble Beberapa Model

Ensemble adalah salah satu teknik paling ampuh untuk meningkatkan performa di kompetisi machine learning. Dengan menggabungkan prediksi dari beberapa model yang berbeda, kita dapat mengurangi varians dan bias, yang seringkali menghasilkan skor yang lebih tinggi dan lebih stabil.

#### 7.2.2. Implementasi: Snapshot Ensemble atau SWA (Stochastic Weight Averaging)

Dua teknik ensemble yang sangat efisien adalah:
*   **Snapshot Ensemble:** Melatih satu model dengan LR schedule yang siklikal (naik turun secara berkala). Setiap kali LR mencapai minimum, model cenderung berada di minimum lokal yang berbeda. Kita dapat menyimpan "snapshot" model pada setiap minimum lokal ini dan menggabungkan prediksinya.
*   **Stochastic Weight Averaging (SWA):** Sebuah teknik yang lebih halus di mana kita menghitung rata-rata bergerak dari bobot model selama fase akhir pelatihan. Model yang dihasilkan dari rata-rata bobot ini seringkali memiliki generalisasi yang jauh lebih baik daripada model pada epoch terakhir.

#### 7.2.3. Teknik: Menggabungkan Prediksi dari 3-5 Model Terbaik

Jika sumber daya memungkinkan, teknik ensemble yang paling umum adalah melatih beberapa model secara independen (menggunakan arsitektur berbeda atau inisialisasi acak yang berbeda) dan menggabungkan prediksi mereka. Penggabungan dapat dilakukan dengan:
*   **Voting:** Mengambil kelas yang paling sering diprediksi (hard voting) atau merata-ratakan probabilitas prediksi (soft voting).
*   **Weighted Averaging:** Memberikan bobot yang lebih tinggi pada model yang memiliki performa validasi yang lebih baik.

### 7.3. Submission 3: Final Tweak atau Submission Eksperimental

Submission terakhir adalah kesempatan terakhir untuk mendapatkan poin ekstra. Ini harus digunakan dengan hati-hati.

#### 7.3.1. Tujuan: Maksimalisasi Skor dengan Sisa Anggaran Submission

Jika skor dari dua submission pertama sangat dekat, submission ketiga bisa digunakan untuk menguji satu variabel yang berbeda. Jika submission kedua (ensemble) jauh lebih unggul, submission ketiga bisa menjadi ensemble dengan konfigurasi yang sedikit berbeda.

#### 7.3.2. Implementasi: Submission dengan Test Time Augmentation (TTA)

**Test Time Augmentation (TTA)** adalah teknik di mana, selama fase inferensi, setiap gambar uji diaugmentasi beberapa kali (misalnya, di-flip secara horizontal, di-rotasi sedikit). Model kemudian membuat prediksi pada setiap versi gambar yang diaugmentasi, dan prediksi akhir adalah rata-rata dari semua prediksi ini. TTA adalah cara yang hampir "gratis" untuk meningkatkan performa dan dapat memberikan peningkatan beberapa poin persen.

#### 7.3.3. Teknik: Pseudo-Labeling (Jika Diizinkan oleh Aturan Kompetisi)

**Pseudo-labeling** adalah teknik semi-supervised learning di mana model terbaik digunakan untuk memprediksi label pada data uji yang tidak berlabel. Sampel-sampel dengan prediksi yang paling percaya diri kemudian ditambahkan ke set pelatihan (dengan label pseudo mereka), dan model dilatih ulang pada data yang diperluas ini. **Teknik ini sangat kuat tetapi harus digunakan dengan sangat hati-hati dan hanya jika diizinkan secara eksplisit oleh aturan kompetisi**. Penggunaan pseudo-labeling yang tidak benar dapat dengan cepat merusak model jika pseudo-labels berisi terlalu banyak noise.

## 8. Final Verdict dan Rekomendasi Akhir

### 8.1. Model Paling Direkomendasikan

Berdasarkan analisis komprehensif, **ConvNeXt V2-Base** adalah model paling direkomendasikan untuk submission akhir. Ia menawarkan kapasitas representasi tertinggi di antara kandidat yang efisien, yang sangat penting untuk menangani kompleksitas kelas Electronic. Arsitekturnya yang modern memberikan keseimbangan sempurna antara performa dan efisiensi. Sebagai alternatif yang sangat kuat, terutama untuk iterasi cepat, adalah **EfficientNetV2-S**, yang unggul dalam kecepatan training.

### 8.2. Strategi Training Paling Masuk Akal

Strategi training harus menjadi kombinasi dari praktik terbaik modern:
1.  **Pre-trained Weights:** Gunakan model pre-trained pada **ImageNet-21K**.
2.  **Fine-tuning:** Terapkan **staged fine-tuning** dengan discriminative learning rates.
3.  **Augmentasi:** Gunakan pipeline augmentasi kuat yang mencakup **RandAugment, CutMix, dan Mixup**.
4.  **Loss Function:** Gunakan **Weighted Focal Loss** untuk menangani ketidakseimbangan kelas secara agresif.
5.  **Regularisasi:** Terapkan **Label Smoothing dan Dropout**.
6.  **Optimasi:** Gunakan **AdamW** dengan **Cosine Annealing LR schedule** dan **Linear Warmup**.
7.  **Stabilisasi:** Gunakan **Exponential Moving Average (EMA)** pada bobot model.

### 8.3. Prioritas Eksperimen dari Tahap Awal hingga Final

| Fase | Prioritas | Durasi (Perkiraan) |
| :--- | :--- | :--- |
| **1. Eksplorasi & Baseline** | Replikasi paper untuk validasi pipeline. Implementasi EfficientNetV2-S dengan augmentasi standar. | 1-2 hari |
| **2. Optimasi & Iterasi** | Eksperimen dengan augmentasi canggih (CutMix, RandAugment). Implementasi Weighted Focal Loss. Tuning LR dan weight decay. | 3-5 hari |
| **3. Finalisasi Model** | Latih ConvNeXt V2-Base dengan konfigurasi terbaik. Implementasi EMA dan SWA. | 1-2 hari |
| **4. Ensemble & Submission** | Buat ensemble dari model-model terbaik. Terapkan TTA. Lakukan 3 submission sesuai strategi. | 1 hari |

*Tabel 3: Roadmap Eksperimen Prioritas.*

### 8.4. Risiko Terbesar yang Mungkin Muncul

1.  **Overfitting pada Kelas Electronic:** Meskipun menggunakan teknik ketidakseimbangan, model mungkin masih gagal menggeneralisasi pada variasi Electronic yang tidak terduga. **Mitigasi:** Augmentasi yang sangat agresif dan diversifikasi arsitektur dalam ensemble.
2.  **Suboptimal Hyperparameter Tuning:** Keterbatasan waktu mungkin mencegah penyetelan hyperparameter yang sempurna. **Mitigasi:** Gunakan LR Finder dan library untuk hyperparameter optimization (seperti Optuna) untuk mengotomatisasi pencarian.
3.  **Kegagalan Ensemble:** Ensemble tidak selalu berhasil, terutama jika model dasarnya terlalu serupa. **Mitigasi:** Pastikan untuk menggabungkan model dengan arsitektur yang beragam (misalnya, ConvNeXt + Swin Transformer).
4.  **Kesalahan Pipeline Data:** Kesalahan kecil dalam preprocessing atau data loading dapat merusak seluruh eksperimen. **Mitigasi:** Validasi pipeline secara menyeluruh pada tahap awal dengan baseline yang sederhana.

### 8.5. Tingkat Keyakinan dan Alasannya

**Tingkat Keyakinan: TINGGI**

Alasan keyakinan ini tinggi didasarkan pada beberapa pilar:
1.  **Bukti Empiris yang Kuat:** Model modern seperti ConvNeXt dan EfficientNetV2 telah terbukti secara konsisten mengungguli arsitektur lama seperti InceptionV3 di hampir semua benchmark.
2.  **Kesesuaian dengan Tantangan:** Masalah utama dalam kompetisi ini (ketidakseimbangan kelas, kompleksitas kelas baru) adalah masalah yang terdokumentasi dengan baik dalam literatur deep learning, dan solusi yang efektif (Focal Loss, CutMix, model modern) telah tersedia.
3.  **Strategi yang Berbasis Data:** Seluruh strategi dirancang secara spesifik untuk menangani karakteristik dataset yang diketahui, bukan strategi generik.
4.  **Pendekatan Bertahap:** Roadmap yang diusulkan memungkinkan validasi dan iterasi, yang meminimalkan risiko kegagalan besar.

Dengan mengikuti rekomendasi strategis dalam dokumen ini, tim memiliki peluang yang sangat kuat untuk mencapai peringkat teratas di kompetisi Big Data Challenge Satria Data 2026.