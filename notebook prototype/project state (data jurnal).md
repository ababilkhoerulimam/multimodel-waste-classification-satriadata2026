# PROJECT STATE — PAPER DATASET (WASTE CLASSIFICATION)
File ini khusus mendokumentasikan eksplorasi dataset dari paper publik yang diduga menjadi sumber data kompetisi.
Dibuat: 2 Juli 2026

## META
- Nama Dataset         : Waste Classification Dataset (paper)
- Path                 : c:\Users\HP\Documents\BDC 2026\satria-data-bdc-2026-1\Waste Classification Dataset\waste_dataset
- Struktur             : 1 folder pool (tidak ada split train/test)
- Kelas                : organic, recyclable (2 kelas)
- Total file gambar    : 24.705 (organic: 13.880, recyclable: 10.825)
- File non-gambar      : 4 notebook .ipynb + 1 .txt (dideteksi sebagai "corrupt" oleh proses loading, abaikan)
- Status Eksplorasi    : Selesai (statistik, duplikasi, perbandingan dengan kompetisi)

## RINGKASAN TEMUAN UTAMA
1. Paper dataset memiliki 2 kelas: organic (13.880) dan recyclable (10.825) — TIDAK ada kelas Electronic.
2. Statistik warna, brightness, dimensi, background, dan foreground ratio SANGAT MIRIP dengan kelas Organic & Recyclable di dataset kompetisi. Selisih sangat kecil, mengonfirmasi bahwa kompetisi menggunakan subset dari dataset ini untuk kedua kelas tersebut.
3. Cross-class duplicate di paper (organic_012716 vs recyclable_008879) adalah file yang identik dengan cross-class duplicate di kompetisi (Flag 7: O_8873.jpg vs R_799.jpg). Ini membuktikan noise labeling berasal dari sumber asli, bukan kesalahan panitia.
4. Electronic di kompetisi tidak ada di paper; kelas ini adalah tambahan murni dari sumber lain.

## STATISTIK LENGKAP PAPER DATASET
*Statistik dihitung dari sample 500 gambar per kelas (kecuali jumlah total file dan duplikasi yang dihitung dari seluruh dataset).*

### 1. Integritas File
- File corrupt riil: 0
- 5 file tidak bisa dibuka karena bukan gambar (notebook .ipynb, .txt) — bukan bagian dataset gambar.

### 2. Dimensi & Resolusi
| Kelas      | Width Mean | Height Mean | Common Size   | % Common |
|------------|------------|-------------|---------------|----------|
| organic    | 262.23     | 196.48      | 275x183       | 14.0%    |
| recyclable | 241.85     | 211.69      | 225x225       | 29.6%    |

- Tidak ada gambar berukuran 150x150 (tidak seperti Electronic di kompetisi).
- Organic lebih bervariasi dimensinya, Recyclable banyak yang persegi.

### 3. Aspect Ratio & Orientasi
| Kelas      | AR Mean | % Landscape | % Portrait | % Square |
|------------|---------|-------------|------------|----------|
| organic    | 1.412   | 73.4%       | 12.2%      | 14.4%    |
| recyclable | 1.207   | 46.6%       | 21.4%      | 32.0%    |

- Organic didominasi landscape (objek memanjang seperti buah/sayur).
- Recyclable lebih seimbang, dengan square paling banyak (produk simetris).

### 4. Warna & Brightness
| Kelas      | Brightness Mean | R Mean | G Mean | B Mean | Pola Warna               |
|------------|-----------------|--------|--------|--------|--------------------------|
| organic    | 140.58          | 163.78 | 145.79 | 112.17 | R dominan, B rendah (warna alami) |
| recyclable | 180.52          | 186.47 | 180.45 | 174.63 | R≈G≈B, terang (studio)   |

- Brightness organic lebih rendah, recyclable lebih tinggi (konsisten dengan background putih).
- Standar deviasi brightness ~45-48, menunjukkan variasi dalam kelas.

### 5. Background Variance
| Kelas      | Variance Mean | % Plain Background (variance < threshold) |
|------------|---------------|-------------------------------------------|
| organic    | 1872.39       | 15.4%                                     |
| recyclable | 1284.24       | 35.2%                                     |

- Recyclable lebih sering memiliki background polos (studio), organic lebih bervariasi (campuran studio dan natural).

### 6. Foreground Ratio (Ukuran Objek Relatif)
| Kelas      | Mean   | Median | 25%    | 75%    |
|------------|--------|--------|--------|--------|
| organic    | 0.786  | 0.881  | 0.676  | 0.958  |
| recyclable | 0.600  | 0.612  | 0.370  | 0.862  |

- Organic cenderung close-up (objek besar memenuhi frame).
- Recyclable lebih bervariasi, banyak objek kecil di tengah ruang kosong.

## DUPLIKASI INTERNAL PAPER DATASET
- Total grup duplikat (exact hash): 326
- Within-class: 325 grup (organic: 630 file, recyclable: 20 file)
- Cross-class: 1 grup (kritis, lihat di bawah)

### Cross-Class Duplicate
Hash: 95bd2693fd68b87d40601c3002ebdf21
organic_012716_photo.jpg (label: organic)
recyclable_008879_photo.jpg (label: recyclable)

- **Ini adalah file yang sama dengan cross-class duplicate di kompetisi** (Flag 7: `O_8873.jpg` dan `R_799.jpg`), hanya berbeda nama.
- Gambar: tas kain "Say No To Plastic" — seharusnya Recyclable, bukan Organic.
- Noise labeling ini sudah ada di sumber asli; bukan kesalahan panitia kompetisi.

## PERBANDINGAN DENGAN DATASET KOMPETISI
| Parameter              | Paper (sample 500) | Kompetisi (Jeremy) | Kecocokan |
|------------------------|-------------------|-------------------|-----------|
| Brightness Organic     | 140.58            | 145.0             | Sangat dekat |
| Brightness Recyclable  | 180.52            | 181.8             | Sangat dekat |
| Pola Warna             | R≈G≈B terang / R dominan | Sama        | Identik secara pola |
| Dimensi (umum)         | Tidak ada 150x150 | Ada 150x150 hanya di Electronic | Paper tidak ada Electronic |
| Background Variance    | 15.4% / 35.2%     | Visual: konsisten | Cocok |
| Foreground Ratio       | 0.786 / 0.600     | 0.761 / 0.589     | Hampir sama |
| Cross-Class Duplicate  | Ada (1 grup)      | Ada (Flag 7)      | Sama persis (hash identik) |

## IMPLIKASI & STATUS
- **Hipotesis terbukti kuat**: Kompetisi menggunakan subset dari paper dataset untuk kelas Organic dan Recyclable, lalu menambahkan Electronic dari sumber lain.
- **Konfirmasi absolut** menunggu MD5 hash matching antara seluruh file paper dengan file train/test kompetisi.
- **Risiko**: Jika test set kompetisi juga mengandung file dari paper, label ground truth-nya bisa diketahui. Perlu fatwa dari Vierico tentang kepatuhan aturan sebelum digunakan, karena masuk wilayah abu-abu (data eksternal).
- **Sumber noise**: Cross-class mislabel berasal dari paper, artinya ada kemungkinan mislabel lain yang tidak terdeteksi. Rekomendasi audit manual oleh Jeremy tetap berlaku.

## NEXT STEPS
- [ ] **MD5 matching** seluruh 24.705 file paper vs train kompetisi dan test kompetisi. 
- [ ] **Vierico** berikan fatwa mengenai boleh/tidaknya menggunakan informasi paper (label) untuk keperluan validasi atau inisialisasi model.
- [ ] Update status dari HYPOTHESIS menjadi CONFIRMED setelah MD5 match.