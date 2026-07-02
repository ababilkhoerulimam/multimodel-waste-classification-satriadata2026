# PROJECT STATE
# Update ini setiap kali ada handoff atau ganti akun Claude.
# Saat ganti akun: paste AGENTS.md dulu, lalu paste file ini, lalu ketik "lanjutkan dari [stage]"

---

## META

```
Competition / Project : [nama kompetisi atau project]
Kaggle URL            : [link]
Deadline              : [tanggal dan jam WIB]
Metric                : [AUC / RMSE / LogLoss / dll]
Submission budget     : [sisa slot] / 28
Submissions used today: [angka] / 2
Last updated          : [tanggal dan jam]
Updated by            : [Ababil / Jeremy / Vierico]
```

---

## CURRENT STATUS

```
Active phase          : [FASE 1 EDA / FASE 2 FE / FASE 3 MODELING / FASE 4 ENSEMBLE]
Last completed stage  : [contoh: Jeremy E6, atau Ababil Stage 8]
Next action           : [siapa yang harus jalan berikutnya dan apa yang harus dilakukan]
Blocker (if any)      : [ada veto Vierico? delegasi Jeremy yang belum selesai? dll]
```

---

## DATASET

```
Train file    : [nama file] — [jumlah baris] rows, [jumlah kolom] cols
Test file     : [nama file] — [jumlah baris] rows
Target column : [nama kolom]
Task type     : [Binary classification / Multiclass / Regression]
Time-based    : [YES / NO — jika YES, kolom waktu: nama kolom]
```

---

## EXPLORATION REPORT STATUS (Jeremy)

```
Jeremy stage saat ini : [E1 / E2 / E3 / E4 / E5 / E6 / E7 / E8 / E9 / DONE]
Exploration Report    : [BELUM / DRAFT / SENT TO ABABIL]
Post-FE Report (E9)   : [BELUM / PENDING DELEGASI / SENT TO ABABIL]
```

**Key findings dari Jeremy (isi setelah E8 selesai):**
```
- [Finding 1]
- [Finding 2]
- [Finding 3]
```

**Hipotesis yang sudah divalidasi:**
```
H1: [hipotesis] — Status: [CONFIRMED / REJECTED / PENDING]
H2: [hipotesis] — Status: [CONFIRMED / REJECTED / PENDING]
H3: [hipotesis] — Status: [CONFIRMED / REJECTED / PENDING]
```

**Risk flags dari Jeremy:**
```
- [Flag 1: deskripsi singkat]
- [Flag 2: deskripsi singkat]
```

---

## BUSINESS BRIEF STATUS (Vierico)

```
Checkpoint B1 (Problem Brief)  : [BELUM / DONE]
Checkpoint B2 (EDA Commentary) : [BELUM / DONE]
Checkpoint B3 (Strategy Review): [BELUM / DONE]
Checkpoint B4 (Error Cost)     : [BELUM / DONE]
Checkpoint B5 (Explainability) : [BELUM / DONE]
Checkpoint B6 (Exec Summary)   : [BELUM / DONE]
```

**Active veto dari Vierico:**
```
- [NONE / deskripsi veto yang belum resolved]
```

**Business constraints yang sudah dikonfirmasi:**
```
- [Constraint 1: misal "fitur income_estimated tidak boleh dipakai — regulasi"]
- [Constraint 2]
```

---

## FEATURE SET (Ababil)

```
Ababil stage saat ini : [Stage 3 / Stage 5 / Stage 8 / Stage 9 / dll]
FE status             : [BELUM / IN PROGRESS / DONE — menunggu Jeremy E9]
Leakage check         : [BELUM / PASSED / FLAG AKTIF]
Vierico FE review     : [BELUM / APPROVED / VETO AKTIF]
```

**Features yang sudah di-approve (isi setelah Stage 8 selesai):**

| Feature | Type | Source | Leakage Check | Vierico | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [nama] | [num/cat/datetime] | [raw/engineered] | [PASS/FLAG] | [OK/VETO] | [IN MODEL/DROPPED] |
| | | | | | |

**Features yang di-drop:**
```
- [nama]: alasan drop
```

---

## EXPERIMENT LOG SUMMARY

```
Anchor model (Slot 1) : [model type] — CV: [score] — LB: [score] — Delta: [delta]
Anchor model (Slot 2) : [model type] — CV: [score] — LB: [score] — Delta: [delta]
Best CV so far        : [score] — exp_id: [id]
Best LB so far        : [score] — exp_id: [id]
```

**Recent experiments (last 5):**

| exp_id | Stage | Model | CV | LB | Delta | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [exp001] | [9] | [LGB] | [0.xxx] | [0.xxx] | [+/-0.xxx] | [catatan singkat] |
| | | | | | | |

---

## VALIDATION STRATEGY (locked after Stage 7)

```
CV method     : [KFold / StratifiedKFold / GroupKFold / TimeSeriesSplit]
n_folds       : [angka]
Seed          : [angka]
Group column  : [nama kolom / N/A]
Locked        : [YES / NO]
```

---

## PENDING ACTIONS

Isi ini setiap sesi sebelum tutup chat.
```
[ ] [siapa] perlu [apa] — prioritas [HIGH/MED/LOW]
[ ] [siapa] perlu [apa] — prioritas [HIGH/MED/LOW]
[ ] [siapa] perlu [apa] — prioritas [HIGH/MED/LOW]
```

---

## CONTEXT RESET PROTOCOL

Saat ganti akun Claude (limit habis), lakukan urutan ini:

**Untuk Ababil:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_ababil.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari [stage terakhir Ababil]. Semua context ada di atas."

**Untuk Jeremy:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_jeremy.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari [stage terakhir Jeremy]. Semua context ada di atas."

**Untuk Vierico:**
1. Buka akun baru
2. Pesan 1: paste isi `AGENTS_vierico.md`
3. Pesan 2: paste isi file ini (`project_state.md`)
4. Pesan 3: "Lanjutkan dari checkpoint [B-sekian]. Semua context ada di atas."

**Jangan skip langkah ini.** Tanpa project_state, Claude mulai dari nol.

---

## CATATAN BEBAS

Gunakan bagian ini untuk hal-hal yang tidak masuk kategori di atas:
```
[tanggal] [siapa]: [catatan]
[tanggal] [siapa]: [catatan]
```
