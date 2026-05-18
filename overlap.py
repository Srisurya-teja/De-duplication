"""
check_npz_overlap_faiss.py
--------------------------
Finds overlapping identities between two NPZ folders by comparing
per-identity representative embeddings using FAISS.

Strategy
--------
1. For each NPZ in both folders, compute one representative embedding
   (L2-normalised mean of all embeddings in that NPZ).
2. Build a FAISS IndexFlatIP (inner-product = cosine similarity for
   L2-normalised vectors) from folder2 representatives.
3. Query with folder1 representatives.
4. Flag pairs whose cosine distance < --tolerance as the same person.

This reduces millions of individual embeddings to ~4.6 lakh representative
vectors, making the search fast and memory-efficient.

Outputs
-------
- Console summary
- overlap_report.csv  : pairs that overlap  (npz1, npz2, cosine_distance)
- no_overlap_f1.csv   : NPZ files in folder1 with no match in folder2
- no_overlap_f2.csv   : NPZ files in folder2 with no match in folder1

Usage
-----
    python check_npz_overlap_faiss.py \
        --folder1    /path/to/cleaned_npz \
        --folder2    /path/to/g360k_npz_folder \
        --tolerance  0.6 \
        --output-dir /path/to/overlap_results
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import faiss

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def l2_normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D array (row-wise for 2-D)."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def compute_representative(npz_path: Path):
    """
    Load one NPZ and return its representative embedding:
    L2-normalised mean of all embeddings inside.

    Returns (representative: np.ndarray float32, n_images: int)
    or (None, 0) on failure.
    """
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        log.warning("Cannot load %s: %s", npz_path.name, e)
        return None, 0

    if "embeddings" not in data.files:
        log.warning("No 'embeddings' key in %s — skipping.", npz_path.name)
        return None, 0

    embs = data["embeddings"].astype(np.float32)   # (N, D)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return None, 0

    mean_emb = embs.mean(axis=0)                   # (D,)
    return l2_normalise(mean_emb), embs.shape[0]


def load_representatives(folder: Path, label: str):
    """
    Load representative embeddings for all NPZ files in a folder.

    Returns
    -------
    names  : list of str       NPZ stem names
    matrix : np.ndarray        shape (M, D), float32, L2-normalised
    counts : list of int       number of images per NPZ
    """
    npz_files = sorted(folder.glob("*.npz"))
    log.info("[%s]  Found %d NPZ files.", label, len(npz_files))

    names  = []
    vecs   = []
    counts = []

    for npz_path in tqdm(npz_files, desc=f"Loading {label}", unit="npz"):
        rep, n = compute_representative(npz_path)
        if rep is not None:
            names.append(npz_path.stem)
            vecs.append(rep)
            counts.append(n)

    matrix = np.stack(vecs).astype(np.float32)   # (M, D)
    log.info("[%s]  Loaded %d valid representatives (D=%d).", label, len(names), matrix.shape[1])
    return names, matrix, counts


# ── Core overlap detection ────────────────────────────────────────────────────

def find_overlaps(
    names1: list, matrix1: np.ndarray,
    names2: list, matrix2: np.ndarray,
    tolerance: float,
):
    """
    For each representative in folder1, find the nearest neighbour in folder2.
    Flag as overlap if cosine distance ≤ tolerance.

    Cosine distance = 1 - inner_product  (for L2-normalised vectors).

    Returns
    -------
    overlaps    : list of (name1, name2, cosine_dist)  — matched pairs
    no_match_1  : list of name1 that had no match in folder2
    no_match_2  : set of name2 that were never matched by folder1
    """
    dim = matrix2.shape[1]

    # FAISS inner-product index (equivalent to cosine similarity for L2-normalised vectors)
    log.info("Building FAISS index over %d folder2 representatives...", len(names2))
    index = faiss.IndexFlatIP(dim)
    index.add(matrix2)
    log.info("FAISS index built.")

    # Query: for each folder1 representative, find top-1 match in folder2
    log.info("Querying FAISS index with %d folder1 representatives...", len(names1))
    similarities, indices = index.search(matrix1, k=1)   # (M1, 1) each
    log.info("Search complete.")

    overlaps   = []
    no_match_1 = []
    matched_2  = set()

    for i, (sim, idx) in enumerate(zip(similarities[:, 0], indices[:, 0])):
        cosine_dist = float(1.0 - sim)
        if cosine_dist <= tolerance:
            overlaps.append((names1[i], names2[idx], round(cosine_dist, 6)))
            matched_2.add(idx)
        else:
            no_match_1.append(names1[i])

    no_match_2 = [names2[j] for j in range(len(names2)) if j not in matched_2]

    return overlaps, no_match_1, no_match_2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find overlapping identities between two NPZ folders "
            "by comparing per-identity representative embeddings using FAISS."
        )
    )
    parser.add_argument("--folder1",    required=True, type=Path,
                        help="First NPZ folder (e.g. cleaned_npz)")
    parser.add_argument("--folder2",    required=True, type=Path,
                        help="Second NPZ folder (e.g. g360k_npz_folder)")
    parser.add_argument("--tolerance",  type=float, default=0.6,
                        help="Cosine distance threshold to consider same person "
                             "(default: 0.6). Lower = stricter.")
    parser.add_argument("--output-dir", type=Path, default=Path("overlap_results"),
                        help="Folder to save CSV reports (default: overlap_results/)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load representatives ──────────────────────────────────────────────────
    names1, matrix1, counts1 = load_representatives(args.folder1, "folder1")
    names2, matrix2, counts2 = load_representatives(args.folder2, "folder2")

    # ── Find overlaps ─────────────────────────────────────────────────────────
    overlaps, no_match_1, no_match_2 = find_overlaps(
        names1, matrix1, names2, matrix2, args.tolerance
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Folder 1 identities           : {len(names1):>10,}")
    print(f"  Folder 2 identities           : {len(names2):>10,}")
    print(f"  Overlapping (same person)     : {len(overlaps):>10,}")
    print(f"  Only in Folder 1              : {len(no_match_1):>10,}")
    print(f"  Only in Folder 2              : {len(no_match_2):>10,}")
    print(f"{'═' * 60}\n")

    if overlaps:
        log.info("Sample overlapping pairs (first 10):")
        for n1, n2, dist in overlaps[:10]:
            print(f"    {n1}  ↔  {n2}  (cosine dist: {dist:.4f})")
        if len(overlaps) > 10:
            print(f"    ... and {len(overlaps) - 10} more (see overlap_report.csv)")
    else:
        log.info("No overlapping identities found at tolerance=%.2f.", args.tolerance)

    # ── Save CSV reports ──────────────────────────────────────────────────────

    # 1. Overlap pairs
    overlap_csv = args.output_dir / "overlap_report.csv"
    with open(overlap_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["npz_folder1", "npz_folder2", "cosine_distance"])
        writer.writerows(overlaps)
    log.info("Overlap pairs saved → %s  (%d rows)", overlap_csv, len(overlaps))

    # 2. Folder1-only identities
    f1_only_csv = args.output_dir / "only_in_folder1.csv"
    with open(f1_only_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["npz_folder1"])
        writer.writerows([[n] for n in no_match_1])
    log.info("Folder1-only saved     → %s  (%d rows)", f1_only_csv, len(no_match_1))

    # 3. Folder2-only identities
    f2_only_csv = args.output_dir / "only_in_folder2.csv"
    with open(f2_only_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["npz_folder2"])
        writer.writerows([[n] for n in no_match_2])
    log.info("Folder2-only saved     → %s  (%d rows)", f2_only_csv, len(no_match_2))

    log.info("All done. Results saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()