Ini list lengkap **semua** perubahan data sepanjang proyek — exclude, relabel, dan yang belum dieksekusi — digabung jadi satu, urut kronologis.

## 1. EXCLUDE (Batch 0 — sebelum training pertama)

**97 file** → `exclude_from_cv=True` (exact-duplicate MD5 antara train & test, dikeluarkan dari CV supaya tidak leakage; detail lengkap 97 nama file ada di `train_test_overlap.csv`, tidak saya list manual di sini karena banyak dan sudah tersimpan sebagai file terpisah)

**1 file** → `exclude_from_training=True`:
```
O_8873.jpg   # Duplicate byte-identik dengan R_799.jpg (tas kain "SAY NO TO PLASTIC")
             # True class = Recyclable, tapi dilabel Organic. Dikeluarkan total, BUKAN direlabel.
```

## 2. RELABEL — Batch 1 (Cell 34, investigasi Electronic, SUDAH DIEKSEKUSI)

```python
RELABEL_MAP = {
    "R_3825.jpg":     1,  # was Recyclable(0) -> Electronic (confirmed: laptop)
    "R_3733.jpg":     1,  # was Recyclable(0) -> Electronic (confirmed: laptop)
    "O_7776.jpg":     1,  # was Organic(2)    -> Electronic (confirmed: control panel)
    "battery_61.jpg": 0,  # was Electronic(1) -> Recyclable (confirmed: bottles)
}
```
**4 file**, sudah live di `train_master_with_folds.csv` sejak 4 Juli — ini yang dipakai di exp002 dst.

## 3. RELABEL & DROP — Batch 2 (Gate 2, investigasi Recyclable↔Organic, **BELUM DIEKSEKUSI**)

**Relabel → Recyclable (0)**, 51 file:
```
R_386, R_8257, R_2067, R_6246, R_9577, R_531, R_600, R_5929,
O_1892, O_6271, O_1598, O_1876, O_5084, O_5263, O_262, O_1666, O_6852, O_1783,
O_6812, O_9175, O_5295, O_1909, O_8864, O_6629, O_454, O_4910, O_1816, O_6757,
O_1494, O_1556, O_1807, O_10473, O_5225, O_1949, O_712, O_10014, O_6841, O_1888,
O_9183, O_5257, O_7969, O_1635, O_6640, O_1625, O_517, O_7248,
O_1538, O_1800, O_345, O_5259
```

**Relabel → Organic (2)**, baru 3 dari 8 yang tercatat lengkap namanya:
```
R_566, R_9775, R_2451
```
⚠️ **5 nama file lainnya belum tercatat lengkap di chat kita** — perlu diambil ulang dari `Miss_Label_Report_1.xlsx` sebelum eksekusi.

**DROP (exclude_from_training)**, 56 file:
```
R_9957, R_4583, R_850, R_4638, R_8327, R_2909, R_6142, R_4602, R_5044, R_6228,
R_759, R_258, R_6054, R_8315, R_6087, R_4841, R_6935, R_4090, R_891, R_2835,
R_6092, R_4413, R_2786, R_4198, R_4626, R_810, R_509, R_548, R_4046,
O_8835, O_1345, O_221, O_572, O_6751, O_10358, O_11223, O_1263, O_9569, O_1627,
O_7628, O_10298, O_4286, O_8924, O_6391, O_9746, O_5376, O_6755, O_9919, O_3948,
O_5017, O_1935, O_3621, O_1864, O_162, O_10400, O_11423
```

**Tidak diubah**, 11 file — nama belum tercatat lengkap di chat (ada di `combined_verification.csv` kategori "Label Sudah Benar").

---

## Ringkasan Total

| Kategori | Jumlah | Status |
|---|---|---|
| Exclude (train-test dup) | 97 | ✅ Live |
| Exclude (mislabel awal) | 1 | ✅ Live |
| Relabel Batch 1 (Electronic) | 4 | ✅ Live |
| Relabel Batch 2 → Recyclable | 51 | ⏳ Pending Gate 3 |
| Relabel Batch 2 → Organic | 8 (3 nama lengkap) | ⏳ Pending Gate 3 |
| Drop Batch 2 (noise) | 56 | ⏳ Pending Gate 3 |
| Tidak diubah | 11 | — |

**Sebelum kamu eksekusi Gate 3**, saya sarankan buka ulang `Miss_Label_Report_1.xlsx` untuk ambil 5 nama file "Organic" yang hilang dan 11 nama "Label Sudah Benar" — supaya kodenya bisa pakai list 100% lengkap, bukan yang saya rekonstruksi dari chat. Mau saya siapkan kode yang **langsung baca dari file Excel** (bukan hardcode list manual) supaya tidak ada risiko typo/kurang lengkap sama sekali?