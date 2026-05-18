# NPZ Identity Management Toolkit

A suite of three Python scripts for managing face identity datasets stored as NPZ files. The toolkit covers filtering bad images, splitting mixed-identity files, and detecting duplicate identities across two dataset folders — all without requiring raw images or ONNX models at runtime.

---

## Table of Contents

- [Overview](#overview)
- [Scripts](#scripts)
  - [filter_bad_images.py](#1-filter_bad_imagespy)
  - [organize_identities_npz.py](#2-organize_identities_npzpy)
  - [check_npz_overlap_faiss.py](#3-check_npz_overlap_faisspy)
- [NPZ File Structure](#npz-file-structure)
- [Installation](#installation)
- [Recommended Workflow](#recommended-workflow)
- [Output Files Reference](#output-files-reference)
- [Tolerance Guide](#tolerance-guide)

---

## Overview

| Script | Purpose |
|---|---|
| `filter_bad_images.py` | Remove blacklisted image entries from NPZ files using a CSV |
| `organize_identities_npz.py` | Split mixed-identity NPZ files into per-person clusters |
| `check_npz_overlap_faiss.py` | Detect duplicate identities across two NPZ dataset folders |

---

## Scripts

### 1. `filter_bad_images.py`

Removes bad image entries from NPZ files based on a CSV blacklist. Operates in-place or writes cleaned files to a separate output directory.

**Usage**

```bash
python filter_bad_images.py \
    --bad_csv    bad.csv \
    --npz_dir    /path/to/npz_folder \
    --output_dir /path/to/cleaned_npz_folder
```

**Arguments**

| Argument | Required | Description |
|---|---|---|
| `--bad_csv` | Yes | Path to CSV file containing bad image paths. Must have an `image` column. |
| `--npz_dir` | Yes | Folder containing the source NPZ files. |
| `--output_dir` | No | Destination folder for cleaned NPZ files. Omitting this overwrites files in-place. |

**CSV Format (`bad.csv`)**

```
image
/data/identities/person_01/img_003.jpg
/data/identities/person_07/img_011.jpg
```

**Console Output**

```
─────────────────────────────────────────────────
  NPZ files changed : 1,204
  Entries removed   : 8,731
  Entries kept      : 412,005
─────────────────────────────────────────────────
Done ✓
```

---

### 2. `organize_identities_npz.py`

Splits mixed-identity NPZ files into per-person NPZ files and CSV reports using precomputed embeddings. Only processes folders listed in an `iffy.csv` — no raw images or ONNX model needed.

**Usage**

```bash
python organize_identities_npz.py \
    --npz-dir    /path/to/cleaned_npz \
    --iffy-csv   iffy.csv \
    --output-dir /path/to/output \
    --tolerance  0.6
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--npz-dir` | Yes | — | Folder containing cleaned `.npz` files. |
| `--iffy-csv` | Yes | — | CSV with `folder` and `total_similarity` columns identifying suspect identities. |
| `--output-dir` | Yes | — | Folder where split NPZ files and CSV reports are saved. |
| `--tolerance` | No | `0.6` | Cosine-distance threshold for clustering. See [Tolerance Guide](#tolerance-guide). |
| `--dry-run` | No | `False` | Preview what would be written without creating any files. |
| `--log-file` | No | `None` | Path to write the dry-run report. Only active with `--dry-run`. |

**CSV Format (`iffy.csv`)**

```
folder,total_similarity
/data/identities/Batch-9-m.01mnfws_zipped,1.87
/data/identities/Batch-9-m.02xyzabc_zipped,2.14
```

**Output Structure**

```
output_dir/
├── Batch-9-m.01mnfws_zipped_1.npz       ← Person A
├── Batch-9-m.01mnfws_zipped_2.npz       ← Person B
├── Batch-9-m.01mnfws_zipped_clusters.csv
├── Batch-9-m.02xyzabc_zipped_1.npz
└── Batch-9-m.02xyzabc_zipped_clusters.csv
```

**Dry-run example**

```bash
python organize_identities_npz.py \
    --npz-dir   /path/to/cleaned_npz \
    --iffy-csv  iffy.csv \
    --output-dir /path/to/output \
    --dry-run \
    --log-file  dry_run_report.txt
```

No files are written; a full plan is printed to console and saved to `dry_run_report.txt`.

---

### 3. `check_npz_overlap_faiss.py`

Detects overlapping identities between two NPZ dataset folders by comparing per-identity representative embeddings using FAISS (cosine similarity via inner product on L2-normalised vectors).

**Strategy:** For each NPZ, one representative embedding is computed (the L2-normalised mean of all embeddings). A FAISS `IndexFlatIP` index is built over folder2 representatives and queried with folder1 representatives. Pairs whose cosine distance is below `--tolerance` are flagged as the same person.

**Usage**

```bash
python check_npz_overlap_faiss.py \
    --folder1    /path/to/cleaned_npz \
    --folder2    /path/to/g360k_npz_folder \
    --tolerance  0.6 \
    --output-dir /path/to/overlap_results
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--folder1` | Yes | — | First NPZ folder. |
| `--folder2` | Yes | — | Second NPZ folder. |
| `--tolerance` | No | `0.6` | Cosine distance threshold. Lower = stricter matching. |
| `--output-dir` | No | `overlap_results/` | Folder to save CSV reports. |

**Console Output**

```
════════════════════════════════════════════════════════════
  Folder 1 identities           :    460,000
  Folder 2 identities           :    360,000
  Overlapping (same person)     :     12,500
  Only in Folder 1              :    447,500
  Only in Folder 2              :    347,500
════════════════════════════════════════════════════════════
```

**Output Files**

| File | Description |
|---|---|
| `overlap_report.csv` | Matched identity pairs with columns `npz_folder1`, `npz_folder2`, `cosine_distance` |
| `only_in_folder1.csv` | Identities in folder1 with no match in folder2 |
| `only_in_folder2.csv` | Identities in folder2 with no match in folder1 |

---

## NPZ File Structure

All three scripts expect NPZ files with the following keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `paths` | `str array` | `(N,)` | File paths of images for this identity |
| `embeddings` | `float32` | `(N, D)` | Face embedding vectors (L2-normalised) |
| `similarity_matrix` | `float32` | `(N, N)` | Pairwise cosine similarity matrix |

---

## Installation

**Python:** 3.8+

```bash
pip install numpy faiss-cpu tqdm
```

> For GPU acceleration, replace `faiss-cpu` with `faiss-gpu`.

---

## Recommended Workflow

Run the scripts in this order for a clean dataset pipeline:

```
1. filter_bad_images.py
        ↓  Remove known-bad images from all NPZ files
        ↓
2. organize_identities_npz.py
        ↓  Split mixed-identity NPZ files into per-person files
        ↓
3. check_npz_overlap_faiss.py
           Detect duplicate identities across two dataset folders
```

---

## Output Files Reference

| Script | Output File | Description |
|---|---|---|
| `filter_bad_images.py` | `<name>.npz` (cleaned) | NPZ with blacklisted entries removed |
| `organize_identities_npz.py` | `<identity>_N.npz` | Per-cluster NPZ for each detected person |
| `organize_identities_npz.py` | `<identity>_clusters.csv` | Image-to-cluster assignment report |
| `organize_identities_npz.py` | `dry_run_report.txt` | Dry-run plan (with `--dry-run --log-file`) |
| `check_npz_overlap_faiss.py` | `overlap_report.csv` | Matched identity pairs across folders |
| `check_npz_overlap_faiss.py` | `only_in_folder1.csv` | Identities exclusive to folder1 |
| `check_npz_overlap_faiss.py` | `only_in_folder2.csv` | Identities exclusive to folder2 |

---

## Tolerance Guide

All three scripts accept a `--tolerance` parameter representing the **cosine distance** threshold:

| Value | Strictness | Use Case |
|---|---|---|
| `0.3` | Strict | High-precision deduplication; few false positives |
| `0.6` | Balanced *(default)* | Recommended for most datasets |
| `0.7` | Lenient | Catches more duplicates; higher false-positive rate |

> **Cosine distance** = `1 − cosine_similarity`. A value of `0` means identical vectors; `2` means opposite vectors.
