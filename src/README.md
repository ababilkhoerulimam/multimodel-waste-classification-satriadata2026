```python
# Cell 0
import os
import pandas as pd
from pathlib import Path

current_dir = Path.cwd()
data_dir = None

check_dir = current_dir
while check_dir != check_dir.parent:
    if "satria-data-bdc" in check_dir.name.lower():
        data_dir = check_dir
        break
    check_dir = check_dir.parent

if data_dir is None:
    data_dir = current_dir

# Derive subdirs
train_dir = data_dir / "train"
test_dir  = data_dir / "test"

submission_path = data_dir / "submission.csv"

# Sanity checks
assert train_dir.exists(), f"train_dir not found: {train_dir}"
assert test_dir.exists(),  f"test_dir not found:  {test_dir}"
assert submission_path.exists(), f"submission.csv not found: {submission_path}"

print(f"data_dir  : {data_dir}")
print(f"train_dir : {train_dir}  ({len(list(train_dir.glob('*/*')))} files)")
print(f"test_dir  : {test_dir}  ({len(list(test_dir.glob('*')))} files)")
print(f"submission: {submission_path}  ({submission_path.stat().st_size:,} bytes)")
```


```python
# Cell 1 (Directory structure utilities and execution)

def print_tree(root_dir, max_files_per_folder=3, prefix=""):
    root_dir = Path(root_dir)
    entries = sorted(root_dir.iterdir(), key=lambda x: (x.is_file(), x.name))
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        if entry.is_dir():
            n_files = len(list(entry.glob("*")))
            print(f"{prefix}{connector}{entry.name}/  ({n_files} items)")
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, max_files_per_folder, prefix + extension)
        else:
            print(f"{prefix}{connector}{entry.name}")

def summarize_structure(data_dir):
    data_dir = Path(data_dir)
    print(f"Folder structure: {data_dir}\n")
    print(data_dir.name + "/")
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        print(f"├── {split_dir.name}/")
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]

        # Show per-class file counts if class subfolders exist, otherwise list flat files
        if subdirs:
            for cls_dir in sorted(subdirs):
                n_files = len(list(cls_dir.glob("*.*")))
                print(f"│   ├── {cls_dir.name}/  -> {n_files} files")
        else:
            files = list(split_dir.glob("*.*"))
            print(f"│   -> {len(files)} files (no subfolders/labels)")
            if files:
                print(f"│   -> example filenames: {[f.name for f in files[:5]]}")

summarize_structure(data_dir)
print_tree(data_dir)
```


```python
# Cell 2 (Corrupt image detection for train and test sets)

from PIL import Image

# Scan all images recursively and return corrupt files with their error messages
def find_corrupt_images(folder_path):
    corrupt_files = {}
    all_files = list(Path(folder_path).rglob("*.*"))

    for file_path in all_files:
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            corrupt_files[str(file_path)] = str(e)

    return corrupt_files, len(all_files)

# Check train split for corrupt images
train_corrupt, train_total = find_corrupt_images(data_dir / "train")
print(f"TRAIN total files scanned: {train_total}")
print(f"TRAIN corrupt files found: {len(train_corrupt)}")
if train_corrupt:
    print("TRAIN corrupt file list:")
    for path, err in train_corrupt.items():
        print(f"  - {path} -> {err}")

# Check test split for corrupt images
test_corrupt, test_total = find_corrupt_images(data_dir / "test")
print(f"\nTEST total files scanned: {test_total}")
print(f"TEST corrupt files found: {len(test_corrupt)}")
if test_corrupt:
    print("TEST corrupt file list:")
    for path, err in test_corrupt.items():
        print(f"  - {path} -> {err}")
```


```python
# Cell 3 (Detect duplicate files between train and test splits using MD5 hashing)

import hashlib

# Compute MD5 hash in chunks to handle large files efficiently
def compute_file_hash(file_path, chunk_size=8192):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

# Build hash lookup table for all test files
test_hashes = {}  # {hash: file_path}
test_files = list((data_dir / "test").rglob("*.*"))

for file_path in test_files:
    file_hash = compute_file_hash(file_path)
    test_hashes[file_hash] = str(file_path)

print(f"Total unique hashes in test: {len(test_hashes)} (from {len(test_files)} files)")

# Find train files whose hash matches any test file
train_test_duplicates = {}  # {train_file_path: matching_test_file_path}
train_files = list((data_dir / "train").rglob("*.*"))

for file_path in train_files:
    file_hash = compute_file_hash(file_path)
    if file_hash in test_hashes:
        train_test_duplicates[str(file_path)] = test_hashes[file_hash]

print(f"\nTrain files identical (exact match) to test files: {len(train_test_duplicates)}")
if train_test_duplicates:
    print("Duplicate pairs (train -> test):")
    for train_path, test_path in train_test_duplicates.items():
        print(f"  - {train_path}  <->  {test_path}")
```


```python
# Cell 4 (Define class folders and sample files per class)

import random
from PIL import Image

random.seed(42)

# Sample up to n_samples files per class from the train directory
def sample_files_per_class(data_dir, class_folders, n_samples=500):
    samples = {}
    for cls in class_folders:
        files = list((data_dir / "train" / cls).rglob("*.*"))
        samples[cls] = random.sample(files, min(n_samples, len(files)))
    return samples

class_folders = ["0_Recyclable", "1_Electronic", "2_Organic"]
sampled_files = sample_files_per_class(data_dir, class_folders, n_samples=500)
```


```python
# Cell 5 (Visual sample grid per class for quick image inspection)

import matplotlib.pyplot as plt

# Display a grid of random sample images to visually inspect class content and backgrounds
def show_sample_grid(sampled_files, class_folders, n_per_class=8):
    fig, axes = plt.subplots(len(class_folders), n_per_class, figsize=(n_per_class * 2, len(class_folders) * 2.2))

    for row, cls in enumerate(class_folders):
        files_to_show = random.sample(sampled_files[cls], n_per_class)
        for col, f in enumerate(files_to_show):
            ax = axes[row, col]
            ax.imshow(Image.open(f).convert("RGB"))
            ax.axis("off")
        axes[row, 0].set_title(cls, loc="left", fontsize=11, fontweight="bold", x=-0.1, y=1.05)

    plt.tight_layout()
    plt.show()

# Render sample grid using files sampled in Cell 4
show_sample_grid(sampled_files, class_folders, n_per_class=8)
```


```python
# Cell 5 (Visual sample grid per class for quick image inspection)

import matplotlib.pyplot as plt

# Display a grid of random sample images to visually inspect class content and backgrounds
def show_sample_grid(sampled_files, class_folders, n_per_class=8):
    fig, axes = plt.subplots(len(class_folders), n_per_class, figsize=(n_per_class * 2, len(class_folders) * 2.2))

    for row, cls in enumerate(class_folders):
        files_to_show = random.sample(sampled_files[cls], n_per_class)
        for col, f in enumerate(files_to_show):
            ax = axes[row, col]
            ax.imshow(Image.open(f).convert("RGB"))
            ax.axis("off")
        axes[row, 0].set_title(cls, loc="left", fontsize=11, fontweight="bold", x=-0.1, y=1.05)

    plt.tight_layout()
    plt.show()

# Render sample grid using files sampled in Cell 4
show_sample_grid(sampled_files, class_folders, n_per_class=8)
```


```python
# Cell 7 (Image dimension, aspect ratio, and orientation analysis per class)

# Collect width, height, aspect ratio, and orientation for all sampled images
dim_results = []

for cls, files in sampled_files.items():
    for f in files:
        try:
            with Image.open(f) as img:
                w, h = img.size
                dim_results.append({
                    "class": cls,
                    "file": str(f),
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 3),
                    "orientation": "square" if abs(w - h) <= 5 else ("landscape" if w > h else "portrait")
                })
        except Exception:
            pass

df_dim = pd.DataFrame(dim_results)

print("Width & Height Statistics per Class")
print(df_dim.groupby("class")[["width", "height"]].describe())

print("\nAspect Ratio Statistics per Class")
print(df_dim.groupby("class")["aspect_ratio"].describe())

print("\nOrientation Distribution per Class (proportion)")
print(pd.crosstab(df_dim["class"], df_dim["orientation"], normalize="index").round(3))
```


```python
# Cell 8 (Electronic class dimension subgroup analysis: 150x150 vs larger images)

# Isolate Electronic class and flag images with uniform 150x150 dimensions
electronic_dims = df_dim[df_dim["class"] == "1_Electronic"].copy()
electronic_dims["is_small_150"] = (electronic_dims["width"] == 150) & (electronic_dims["height"] == 150)

n_small = electronic_dims["is_small_150"].sum()
n_total = len(electronic_dims)
print(f"Electronic images with exact 150x150 size: {n_small} / {n_total} ({n_small/n_total:.1%})")

# Inspect width distribution of non-150x150 images to assess size variation
print("\nWidth statistics for Electronic images that are NOT 150x150:")
print(electronic_dims[~electronic_dims["is_small_150"]]["width"].describe())

# Sample representative files from each subgroup for manual visual inspection
small_samples = electronic_dims[electronic_dims["is_small_150"]]["file"].head(5).tolist()
large_samples = (electronic_dims[~electronic_dims["is_small_150"]]
                 .sort_values("width", ascending=False)["file"]
                 .head(5).tolist())

print("\nExample files: uniform small (150x150):")
for f in small_samples:
    print(f)

print("\nExample files: large (not 150x150):")
for f in large_samples:
    print(f)
```


```python
# Cell 9 (Visual comparison of 150x150 vs large Electronic images)

import matplotlib.pyplot as plt

# Plot sampled images from each subgroup side by side for visual inspection
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, f in enumerate(small_samples):
    axes[0, i].imshow(Image.open(f))
    axes[0, i].set_title(f"150x150\n{Path(f).stem[:15]}", fontsize=8)
    axes[0, i].axis("off")

for i, f in enumerate(large_samples):
    axes[1, i].imshow(Image.open(f))
    axes[1, i].set_title(f"Large\n{Path(f).stem[:15]}", fontsize=8)
    axes[1, i].axis("off")

plt.suptitle("Electronic: 150x150 (top) vs Large (bottom)")
plt.tight_layout()
plt.show()
```


```python
# Cell 10 (Visual sample comparison of Recyclable vs Organic classes)

random.seed(42)
recyclable_samples = random.sample(sampled_files["0_Recyclable"], 5)
organic_samples = random.sample(sampled_files["2_Organic"], 5)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, f in enumerate(recyclable_samples):
    axes[0, i].imshow(Image.open(f))
    axes[0, i].set_title(f"Recyclable\n{Path(f).stem[:15]}", fontsize=8)
    axes[0, i].axis("off")

for i, f in enumerate(organic_samples):
    axes[1, i].imshow(Image.open(f))
    axes[1, i].set_title(f"Organic\n{Path(f).stem[:15]}", fontsize=8)
    axes[1, i].axis("off")

plt.suptitle("Sample Grid: Recyclable (top) vs Organic (bottom)")
plt.tight_layout()
plt.show()
```


```python
# Cell 11 (Brightness and per-channel color mean analysis per class)

# Compute brightness and RGB channel means for all sampled images
brightness_results = []

for cls, files in sampled_files.items():
    for f in files:
        try:
            with Image.open(f) as img:
                arr = np.array(img.convert("RGB"))
                brightness_results.append({
                    "class": cls,
                    "file": str(f),
                    "brightness": arr.mean(),
                    "r_mean": arr[:,:,0].mean(),
                    "g_mean": arr[:,:,1].mean(),
                    "b_mean": arr[:,:,2].mean()
                })
        except Exception:
            pass

df_brightness = pd.DataFrame(brightness_results)

print("Brightness statistics per class:")
print(df_brightness.groupby("class")["brightness"].describe())

print("\nMean color channel (R/G/B) per class:")
print(df_brightness.groupby("class")[["r_mean", "g_mean", "b_mean"]].mean())

# Plot brightness distribution as histogram and boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cls in class_folders:
    subset = df_brightness[df_brightness["class"] == cls]["brightness"]
    axes[0].hist(subset, bins=40, alpha=0.5, label=cls)
axes[0].set_xlabel("Brightness (mean pixel value)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Brightness Distribution per Class")
axes[0].legend()

df_brightness.boxplot(column="brightness", by="class", ax=axes[1])
axes[1].set_title("Brightness per Class (Boxplot)")
axes[1].set_xlabel("")
plt.suptitle("")
plt.tight_layout()
plt.show()
```


```python
# Cell 12 (Exact duplicate detection within train set: cross-class and within-class)

from collections import defaultdict

# Collect all train files across classes and compute their hashes
hash_to_files = defaultdict(list)
all_train_files = []

for cls_dir in sorted(train_dir.iterdir()):
    if not cls_dir.is_dir():
        continue
    for f in cls_dir.glob("*.*"):
        all_train_files.append((cls_dir.name, f))

print(f"Total train files to hash: {len(all_train_files)}")

for cls_name, f in all_train_files:
    try:
        h = compute_file_hash(f)  # defined in Cell 3
        hash_to_files[h].append((cls_name, f))
    except Exception as e:
        print(f"Hash failed: {f} -> {e}")

# Identify hashes shared by more than one file
duplicate_groups = {h: files for h, files in hash_to_files.items() if len(files) > 1}
print(f"\nExact-hash duplicate groups in train: {len(duplicate_groups)}")

# Separate into cross-class and within-class duplicates
cross_class_dupes = {}
within_class_dupes = {}

for h, files in duplicate_groups.items():
    classes_involved = set(cls for cls, f in files)
    if len(classes_involved) > 1:
        cross_class_dupes[h] = files
    else:
        within_class_dupes[h] = files

print(f"\nCross-class duplicates (same file, different label): {len(cross_class_dupes)} groups")
print(f"Within-class duplicates (same file, same label): {len(within_class_dupes)} groups")

# Show cross-class duplicate details — indicate labeling errors
if cross_class_dupes:
    print("\nCross-class duplicate details:")
    for h, files in list(cross_class_dupes.items())[:10]:
        print(f"\nHash: {h}")
        for cls, f in files:
            print(f"  [{cls}] {f.name}")

# Count files involved in within-class duplicates per class
within_class_count = defaultdict(int)
for h, files in within_class_dupes.items():
    cls = files[0][0]
    within_class_count[cls] += len(files)

print("\nWithin-class duplicate file count per class:")
for cls, count in within_class_count.items():
    print(f"  {cls}: {count} files")
```


```python
# Cell 13 (Visual inspection of cross-class duplicates and largest within-class duplicate groups)

# Display the first cross-class duplicate pair to verify labeling conflict visually
cross_pair = list(cross_class_dupes.values())[0]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i, (cls, f) in enumerate(cross_pair):
    axes[i].imshow(Image.open(f))
    axes[i].set_title(f"[{cls}] {f.name}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# Identify within-class duplicate groups with the most copies
group_sizes = sorted([(len(files), h) for h, files in within_class_dupes.items()], reverse=True)
print("5 largest within-class duplicate groups:")
for size, h in group_sizes[:5]:
    files = within_class_dupes[h]
    print(f"\nCopy count: {size}")
    for cls, f in files:
        print(f"  [{cls}] {f.name}")
```


```python
# Cell 14 (Regenerate exact-MD5 duplicate groups and investigate near-duplicates via filename pattern)
# Part A: regenerate exact-MD5 groups (validated baseline: 57 groups, 97 overlap, 1 cross-class)
# Part B: investigate gap between 57 vs 62 — find near-duplicate candidates via "xxx(1).ext" filename pattern

import re

def hash_all(root_dirs):
    hash_map = defaultdict(list)
    exts = {".jpg", ".jpeg", ".png"}
    all_files = []
    for label, folder in root_dirs.items():
        folder = Path(folder)
        for fp in folder.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                h = compute_file_hash(fp)  # defined in Cell 3
                hash_map[h].append((label, str(fp)))
                all_files.append((label, fp))
    return hash_map, all_files

train_dirs = {
    "0_Recyclable": data_dir / "train" / "0_Recyclable",
    "1_Electronic": data_dir / "train" / "1_Electronic",
    "2_Organic": data_dir / "train" / "2_Organic",
}
test_dir_map = {"test": data_dir / "test"}

print("Hashing train...")
train_hashes, train_files = hash_all(train_dirs)
print("Hashing test...")
test_hashes, _ = hash_all(test_dir_map)

# Part A: exact-MD5 train-test overlap
overlap_rows = []
for h, train_entries in train_hashes.items():
    if h in test_hashes:
        for (train_label, train_path) in train_entries:
            for (_, test_path) in test_hashes[h]:
                overlap_rows.append({
                    "md5": h, "train_label": train_label, "train_path": train_path,
                    "train_filename": Path(train_path).name, "test_path": test_path,
                    "test_id": int(Path(test_path).stem),
                })
overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(data_dir / "train_test_overlap.csv", index=False)
print(f"\n[A] Train-test overlap: {len(overlap_df)} rows")

group_rows = []
group_id = 0
for h, entries in train_hashes.items():
    if len(entries) > 1:
        group_id += 1
        labels_in_group = set(label for label, _ in entries)
        is_cross_class = len(labels_in_group) > 1
        for (label, path) in entries:
            group_rows.append({
                "md5": h, "duplicate_group_id": group_id, "label": label,
                "filepath": path, "filename": Path(path).name,
                "is_cross_class_group": is_cross_class, "match_type": "exact_md5",
            })
dup_df = pd.DataFrame(group_rows)
print(f"[A] Exact-MD5 duplicate groups: {dup_df['duplicate_group_id'].nunique()} groups")

# Part B: near-duplicate candidates via "xxx(1).ext" filename pattern
copy_pattern = re.compile(r"^(.*)\((\d+)\)(\.\w+)$")
candidates = []
name_to_path = {fp.name: (label, fp) for label, fp in train_files}

for label, fp in train_files:
    m = copy_pattern.match(fp.name)
    if m:
        base_name = m.group(1) + m.group(3)  # e.g. "630.jpeg"
        if base_name in name_to_path:
            orig_label, orig_fp = name_to_path[base_name]
            h_copy = compute_file_hash(fp)
            h_orig = compute_file_hash(orig_fp)
            candidates.append({
                "file_a": fp.name, "file_b": base_name,
                "label_a": label, "label_b": orig_label,
                "md5_match": h_copy == h_orig,
                "path_a": str(fp), "path_b": str(orig_fp),
            })

cand_df = pd.DataFrame(candidates)
print(f"\n[B] Copy-pattern candidate pairs found: {len(cand_df)}")
if len(cand_df) > 0:
    print(f"[B] Already matched MD5 (already in 57 groups): {cand_df['md5_match'].sum()}")
    near_dup = cand_df[~cand_df["md5_match"]]
    print(f"[B] No MD5 match (near-duplicate candidates, likely missing 5 groups):")
    print(near_dup[["file_a", "file_b", "label_a", "label_b"]])
    near_dup.to_csv(data_dir / "near_duplicate_candidates.csv", index=False)

dup_df.to_csv(data_dir / "train_duplicate_groups.csv", index=False)
```


```python
# Cell 15 (Load saved duplicate/overlap CSVs into dataframes)

overlap_df = pd.read_csv(data_dir / "train_test_overlap.csv")
groups_df = pd.read_csv(data_dir / "train_duplicate_groups.csv")
near_dup_df = pd.read_csv(data_dir / "near_duplicate_candidates.csv")

print(overlap_df.shape, groups_df.shape, near_dup_df.shape)
```


```python
# Cell 16 (Build train_master: enumerate all train files with label and filepath)

label_map = {"0_Recyclable": 0, "1_Electronic": 1, "2_Organic": 2}

records = []
for folder in train_dir.iterdir():
    if not folder.is_dir():
        continue
    if folder.name not in label_map:
        print(f"WARNING: unexpected folder '{folder.name}' — skipped, not in label_map")
        continue
    label_id = label_map[folder.name]
    for fp in folder.glob("*"):
        if fp.is_file():
            records.append({
                "filename": fp.name,
                "filepath": str(fp),
                "label": label_id,
                "label_folder": folder.name
            })

train_master = pd.DataFrame(records)
print(train_master.shape)
train_master.head()
```


```python
# Cell 17 (Merge exact-duplicate group IDs into train_master with collision safety check)

train_master["duplicate_group_id"] = range(len(train_master))

filename_to_group = dict(zip(groups_df["filename"], groups_df["duplicate_group_id"]))
offset = train_master["duplicate_group_id"].max() + 1
filename_to_group = {k: v + offset for k, v in filename_to_group.items()}

train_master["duplicate_group_id"] = train_master.apply(
    lambda row: filename_to_group.get(row["filename"], row["duplicate_group_id"]),
    axis=1
)

# Safety check: merged groups must collapse the unique count below per-row baseline,
# and above raw group count (i.e. merge actually happened, no silent collision)
n_files = len(train_master)
n_exact_dup_files = groups_df["filename"].nunique()
expected_unique = n_files - n_exact_dup_files + groups_df["duplicate_group_id"].nunique()

assert train_master["duplicate_group_id"].nunique() == expected_unique, (
    f"Group collapse mismatch: got {train_master['duplicate_group_id'].nunique()}, "
    f"expected {expected_unique}"
)
print("Unique groups after exact-dup merge:", train_master["duplicate_group_id"].nunique())
```


```python
# Cell 18 (Merge near-duplicate group IDs into train_master using high-offset ID range)

# Pinned constants — safe to re-run this cell independently
NEAR_DUP_GROUP_START = 900000  # far above any realistic file-count-based ID, avoids collision

near_dup_pairs = list(near_dup_df[["file_a", "file_b"]].itertuples(index=False, name=None))

for i, (fa, fb) in enumerate(near_dup_pairs):
    group_id = NEAR_DUP_GROUP_START + i
    mask = train_master["filename"].isin([fa, fb])
    matched = train_master.loc[mask, "filename"].tolist()
    if len(matched) != 2:
        print(f"WARNING: expected 2 matches for pair ({fa}, {fb}), found {matched}")
        continue
    train_master.loc[mask, "duplicate_group_id"] = group_id

print("Unique groups after near-dup merge:", train_master["duplicate_group_id"].nunique())
```


```python
# Cell 19 (Flag train-test overlap and confirmed mislabel for exclusion)

overlap_filenames = set(overlap_df["train_filename"])

# O_8873.jpg: confirmed mislabel — duplicate of R_799.jpg (cloth bag),
# true class = Recyclable, mislabeled as Organic in source data. Excluded from training entirely.
MISLABEL_EXCLUDE = {"O_8873.jpg"}

train_master["exclude_from_cv"] = train_master["filename"].isin(overlap_filenames)
train_master["exclude_from_training"] = train_master["filename"].isin(MISLABEL_EXCLUDE)

print("exclude_from_cv count:", train_master["exclude_from_cv"].sum(), "(expected 97)")
print("exclude_from_training count:", train_master["exclude_from_training"].sum(), "(expected 1)")
```


```python
# Cell 20 (Assert train_master integrity: exclusion counts, no duplicate filenames, no unexpected cross-class groups)

assert train_master["exclude_from_cv"].sum() == 97, "Overlap count mismatch!"
assert train_master["exclude_from_training"].sum() == 1, "Mislabel exclude mismatch!"
assert train_master["filename"].duplicated().sum() == 0, "Duplicate filenames in master list!"

# No group should span multiple classes unless explicitly flagged cross-class in source data
cross_class_check = train_master.groupby("duplicate_group_id")["label"].nunique()
unexpected_cross = set(cross_class_check[cross_class_check > 1].index)

flagged_groups = set(
    groups_df.loc[groups_df["is_cross_class_group"] == True, "duplicate_group_id"] + offset
)

assert unexpected_cross.issubset(flagged_groups), (
    f"Unexpected cross-class groups found: {unexpected_cross - flagged_groups}"
)

print("Total train files:", len(train_master))
print("Files available for CV (exclude_from_cv=False, exclude_from_training=False):",
      len(train_master[(~train_master["exclude_from_cv"]) & (~train_master["exclude_from_training"])]))
print("Cross-class groups detected:", len(unexpected_cross | flagged_groups), "-> all expected:", unexpected_cross.issubset(flagged_groups))

train_master.to_csv(data_dir / "train_master_with_groups.csv", index=False)
train_master.head(10)
```


```python
# Cell 21 (StratifiedGroupKFold: assign fold indices to CV pool)

from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
SEED = 42

cv_pool = train_master[
    (~train_master["exclude_from_cv"]) & (~train_master["exclude_from_training"])
].reset_index(drop=True)

print("CV pool size:", len(cv_pool))
print("Unique groups in CV pool:", cv_pool["duplicate_group_id"].nunique())

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

cv_pool["fold"] = -1

X = cv_pool["filename"]
y = cv_pool["label"]
groups = cv_pool["duplicate_group_id"]

for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    cv_pool.loc[val_idx, "fold"] = fold_idx

assert (cv_pool["fold"] == -1).sum() == 0, "Some rows were not assigned a fold!"
print(cv_pool["fold"].value_counts().sort_index())
```


```python
# Cell 22 (Validate fold assignments: group integrity, class balance, fold size)

# 1) No group should be split across multiple folds
group_fold_check = cv_pool.groupby("duplicate_group_id")["fold"].nunique()
leaky_groups = group_fold_check[group_fold_check > 1]
assert len(leaky_groups) == 0, f"GROUP LEAKAGE: {len(leaky_groups)} groups span multiple folds!\n{leaky_groups}"
print("Group integrity check: PASSED (no group spans multiple folds)")

# 2) Class balance per fold (Macro F1 sensitive to Electronic minority)
balance = cv_pool.groupby("fold")["label"].value_counts(normalize=True).unstack().round(4)
balance.columns = ["Recyclable", "Electronic", "Organic"]
print(balance)

# 2b) Deviation from global class ratio — flags if greedy stratification skewed any fold
global_ratio = cv_pool["label"].value_counts(normalize=True).sort_index()
global_ratio.index = ["Recyclable", "Electronic", "Organic"]
deviation = (balance - global_ratio).abs().round(4)
print("\nAbsolute deviation from global class ratio (global = {}):".format(global_ratio.round(4).to_dict()))
print(deviation)

max_dev = deviation.values.max()
print(f"\nMax deviation across all folds/classes: {max_dev:.4f}")
if max_dev > 0.02:
    print("WARNING: deviation > 2pp — flag before Phase 1 baseline training.")
else:
    print("OK: deviation within acceptable range (<2pp).")

# 3) Fold size check
print("\nFold sizes:")
print(cv_pool["fold"].value_counts().sort_index())
```


```python
# Cell 23 (Merge fold assignments back into train_master and save final CSV)

# Rows not in cv_pool (excluded files) get fold = -1
fold_map = dict(zip(cv_pool["filename"], cv_pool["fold"]))
train_master["fold"] = train_master["filename"].map(fold_map).fillna(-1).astype(int)

train_master.to_csv(data_dir / "train_master_with_folds.csv", index=False)
print("Saved train_master_with_folds.csv")
train_master["fold"].value_counts().sort_index()
```


```python
# Cell 24 (Load submission.csv, build test filepaths, verify all test files exist on disk)

submission_df = pd.read_csv(submission_path)
print(submission_df.shape)
print(submission_df.head())
print(submission_df.dtypes)

# Build filename column explicitly from id (test files are named "{id}.jpg")
submission_df["filename"] = submission_df["id"].astype(str) + ".jpg"
submission_df["filepath"] = submission_df["filename"].apply(lambda f: str(test_dir / f))

# Sanity: every expected test file must actually exist on disk
missing = submission_df[~submission_df["filepath"].apply(lambda p: Path(p).exists())]
assert len(missing) == 0, f"MISSING TEST FILES: {missing['filename'].tolist()}"

print(f"\nAll {len(submission_df)} test files found on disk, matching submission.csv order.")
submission_df.head()
```


```python
# Cell 25 (Audit image mode distribution across train and test sets)

def get_image_mode(filepath):
    try:
        with Image.open(filepath) as img:
            return img.mode
    except Exception as e:
        return f"ERROR: {e}"

# Audit all train and test files for non-RGB modes
train_master["img_mode"] = train_master["filepath"].apply(get_image_mode)
submission_df["img_mode"] = submission_df["filepath"].apply(get_image_mode)

print("TRAIN mode distribution:")
print(train_master["img_mode"].value_counts())
print("\nTEST mode distribution:")
print(submission_df["img_mode"].value_counts())

non_rgb_train = train_master[train_master["img_mode"] != "RGB"]
non_rgb_test = submission_df[submission_df["img_mode"] != "RGB"]
print(f"\nNon-RGB train files: {len(non_rgb_train)} (expected 312: 293 Palette + 17 RGBA + 2 Grayscale)")
print(f"Non-RGB test files: {len(non_rgb_test)}")
```


```python
# Cell 26 (Define robust RGB loader and verify on a known RGBA sample)

def load_image_as_rgb(filepath):
    """
    Robust image loader — always returns a PIL Image in RGB mode.
    Handles RGBA (drops alpha, composites on white bg to avoid black artifacts),
    Palette (P mode, common in GIF/PNG), Grayscale (L), and CMYK edge cases.
    """
    img = Image.open(filepath)

    if img.mode == "RGBA":
        # Composite onto white background instead of naive convert (avoids black
        # halos where alpha=0, since transparent pixels default to black otherwise)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # use alpha channel as mask
        img = background
    elif img.mode != "RGB":
        # Covers P (palette), L (grayscale), CMYK, etc.
        img = img.convert("RGB")

    return img

# Quick test on a known non-RGB file (adjust filename to one from Cell 25 output)
sample_non_rgb = train_master[train_master["img_mode"] == "RGBA"]
if len(sample_non_rgb) > 0:
    test_fp = sample_non_rgb.iloc[0]["filepath"]
    converted = load_image_as_rgb(test_fp)
    print(f"Tested on: {test_fp}")
    print(f"Converted mode: {converted.mode}, size: {converted.size}")
else:
    print("No RGBA sample found in current train_master — check img_mode column.")
```


```python
# Cell 27 (Verify load_image_as_rgb conversion for all non-RGB modes in train and test)

modes_to_test = ["P", "RGBA", "L"]

for mode in modes_to_test:
    sample = train_master[train_master["img_mode"] == mode]
    if len(sample) == 0:
        print(f"No {mode} sample found, skipping")
        continue
    fp = sample.iloc[0]["filepath"]
    try:
        converted = load_image_as_rgb(fp)
        assert converted.mode == "RGB", f"Conversion failed for {mode}: got {converted.mode}"
        print(f"OK: {mode} -> RGB | sample: {Path(fp).name} | size: {converted.size}")
    except Exception as e:
        print(f"FAILED: {mode} conversion error: {e}")

# Verify all test Palette files convert cleanly
test_palette_samples = submission_df[submission_df["img_mode"] == "P"]
print(f"\nVerifying all {len(test_palette_samples)} test Palette files convert cleanly:")
for _, row in test_palette_samples.iterrows():
    converted = load_image_as_rgb(row["filepath"])
    assert converted.mode == "RGB"
print("OK: all test Palette files convert successfully.")
```


```python
# Cell 28 (WasteDataset: PyTorch Dataset wrapping train/val/test dataframes)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class WasteDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        """
        df: DataFrame with at least ['filepath'] column, and ['label'] if not is_test
        transform: torchvision transform pipeline (applied after PIL RGB conversion)
        is_test: if True, returns (image, filename) instead of (image, label) — no label available
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image_as_rgb(row["filepath"])  # defined in Cell 26, handles P/RGBA/L safely

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, row["filename"]
        else:
            return img, int(row["label"])
```


```python
# Cell 29 (Define train and eval transform pipelines)

IMG_SIZE = 224  # baseline resolution for ConvNeXt V2 pretrained models (224x224)

# Train: RandAugment (light-medium) + HFlip + limited ColorJitter (brightness/contrast/saturation) + normalization
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
    T.RandomHorizontalFlip(p=0.5),
    T.RandAugment(num_ops=2, magnitude=7),  # light-medium per lock; tune if baseline unstable
    T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),  # limited — brightness/color is discriminative
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats (ConvNeXt V2 pretrained)
])

# Val/Test: deterministic, no augmentation
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"Transforms ready. Baseline resolution: {IMG_SIZE}x{IMG_SIZE}")
```


```python
# Cell 30 (Build train, val, and test DataLoaders for fold 0 baseline)

BATCH_SIZE = 64  # ConvNeXt V2-Tiny @ 224px, 16GB VRAM baseline (adjust up if want to use 5090 32GB)
NUM_WORKERS = 4  # adjust based on CPU cores

fold_df = pd.read_csv(data_dir / "train_master_with_folds.csv")

FOLD_TO_VALIDATE = 0  # start with fold 0 for baseline sanity check

train_df = fold_df[(fold_df["fold"] != FOLD_TO_VALIDATE) & (fold_df["fold"] != -1)].reset_index(drop=True)
val_df = fold_df[fold_df["fold"] == FOLD_TO_VALIDATE].reset_index(drop=True)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
print(f"Train label distribution:\n{train_df['label'].value_counts(normalize=True).sort_index()}")
print(f"Val label distribution:\n{val_df['label'].value_counts(normalize=True).sort_index()}")

train_dataset = WasteDataset(train_df, transform=train_transform, is_test=False)
val_dataset = WasteDataset(val_df, transform=eval_transform, is_test=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")
```


```python
# Cell 31 (Sanity check: verify train and test batch shapes, dtypes, and pixel value range)

images, labels = next(iter(train_loader))
print("Batch images shape:", images.shape)  # expect [BATCH_SIZE, 3, 224, 224]
print("Batch images dtype:", images.dtype)
print("Batch labels shape:", labels.shape)
print("Batch labels unique values:", labels.unique())
print("Pixel value range (post-normalize):", images.min().item(), "to", images.max().item())

# Sanity check the test loader (no labels)
test_dataset = WasteDataset(submission_df, transform=eval_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

test_images, test_filenames = next(iter(test_loader))
print("\nTest batch images shape:", test_images.shape)
print("Test batch filenames (first 5):", test_filenames[:5])
```
