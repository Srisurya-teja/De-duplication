"""
filter_bad_images.py
--------------------
Removes bad image entries from NPZ files based on a CSV blacklist.

Expected NPZ structure: ['paths', 'embeddings', 'similarity_matrix']
Expected CSV: one column containing file paths of bad images
              (with or without a header row)

Usage:
    python filter_bad_images.py \
        --bad_csv    bad.csv \
        --npz_dir    /path/to/npz_folder \
        --output_dir /path/to/cleaned_npz_folder   # optional; default = overwrites in-place
"""

import argparse
import csv
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm   # pip install tqdm  (optional but nice for ~1 lakh files)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load bad paths into a set (fast O(1) lookup)
# ─────────────────────────────────────────────────────────────────────────────

def load_bad_paths(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV file does not exist: '{csv_path}'")
    
    bad = set()
    
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if "image" column exists
            if "image" not in reader.fieldnames:
                raise KeyError(f"[ERROR] CSV file is missing 'image' column. Found columns: {reader.fieldnames}")
            
            for i, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    path = row["image"].strip()
                except KeyError:
                    raise KeyError(f"[ERROR] Missing 'image' key in row {i}: {row}")
                
                if not path:
                    raise ValueError(f"[ERROR] Empty 'image' path found in row {i}")
                
                bad.add(path)
                
    except Exception as e:
        # Catch any exception and re-raise it with a helpful message
        raise RuntimeError(f"[ERROR] Failed to read CSV '{csv_path}': {e}") from e
    
    print(f"[INFO] Loaded {len(bad):,} bad paths from '{csv_path}'")
    return bad

# ─────────────────────────────────────────────────────────────────────────────
# 2. Filter a single NPZ file
# ─────────────────────────────────────────────────────────────────────────────

def filter_npz(npz_path: str, bad_paths: set, output_path: str) -> dict:
    """
    Returns a small stats dict: {total, kept, removed}
    Returns None if the file was skipped (already clean / unexpected structure).
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] Could not load {npz_path}: {e}")
        return None

    # Validate expected keys
    required = {"paths", "embeddings", "similarity_matrix"}
    if not required.issubset(set(data.files)):
        print(f"[WARN] Unexpected keys in {npz_path}: {data.files} — skipping")
        return None

    paths      = data["paths"]           # 1-D array of path strings
    embeddings = data["embeddings"]      # shape (N, D)
    sim_matrix = data["similarity_matrix"]  # shape (N, N)

    total = len(paths)

    # Build boolean keep-mask
    keep_mask = np.array([p not in bad_paths for p in paths])
    n_removed = int((~keep_mask).sum())

    if n_removed == 0:
        # Nothing to do — just copy if output differs from input
        if output_path != npz_path:
            import shutil
            shutil.copy2(npz_path, output_path)
        return {"total": total, "kept": total, "removed": 0}

    # Apply mask
    clean_paths      = paths[keep_mask]
    clean_embeddings = embeddings[keep_mask]          # rows to keep
    clean_sim        = sim_matrix[np.ix_(keep_mask, keep_mask)]  # sub-matrix

    np.savez_compressed(
        output_path,
        paths=clean_paths,
        embeddings=clean_embeddings,
        similarity_matrix=clean_sim,
    )
    return {"total": total, "kept": int(keep_mask.sum()), "removed": n_removed}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter bad images from NPZ files.")
    parser.add_argument("--bad_csv",    required=True,  help="Path to bad.csv")
    parser.add_argument("--npz_dir",    required=True,  help="Folder containing NPZ files")
    parser.add_argument("--output_dir", default=None,
                        help="Output folder (default: overwrite in-place)")
    args = parser.parse_args()

    bad_paths = load_bad_paths(args.bad_csv)

    npz_dir    = Path(args.npz_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving cleaned NPZ files to: {output_dir}")
    else:
        print("[INFO] Overwriting NPZ files in-place (no --output_dir given)")

    npz_files = sorted(npz_dir.glob("*.npz"))
    print(f"[INFO] Found {len(npz_files):,} NPZ files to process\n")

    total_removed = 0
    total_kept    = 0
    files_changed = 0

    for npz_path in tqdm(npz_files, unit="file"):
        out_path = (output_dir / npz_path.name) if output_dir else str(npz_path)
        stats = filter_npz(str(npz_path), bad_paths, str(out_path))

        if stats:
            total_removed += stats["removed"]
            total_kept    += stats["kept"]
            if stats["removed"] > 0:
                files_changed += 1

    print(f"\n{'─'*50}")
    print(f"  NPZ files changed : {files_changed:,}")
    print(f"  Entries removed   : {total_removed:,}")
    print(f"  Entries kept      : {total_kept:,}")
    print(f"{'─'*50}")
    print("Done ✓")


if __name__ == "__main__":
    main()