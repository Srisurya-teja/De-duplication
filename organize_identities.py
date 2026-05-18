"""
organize_identities_npz.py
--------------------------
Splits mixed identity NPZ files into per-person NPZ files and CSV reports
using PRECOMPUTED embeddings. No images or ONNX model needed.

Inputs
------
--npz-dir    : folder containing cleaned .npz files (one per identity folder)
--iffy-csv   : CSV with columns: folder, total_similarity
               Only folders listed here are processed.
               The 'folder' column must contain the full path to the identity folder.
--output-dir : folder where split NPZ files and CSV reports are saved

Outputs (per iffy folder, written to --output-dir)
-------
Per cluster NPZ  : <identity>_1.npz, <identity>_2.npz, ...
                   Each contains: paths, embeddings, similarity_matrix (sub-matrix)
CSV report       : <identity>_clusters.csv
                   Columns: image_path, cluster

Example
-------
Input NPZ  : cleaned_npz/Batch-9-m.01mnfws_zipped.npz
             paths      → [img1.jpg, img2.jpg, img3.jpg, img4.jpg]
             embeddings → (4, 512)

Output:
    output_dir/
        Batch-9-m.01mnfws_zipped_1.npz   ← person A embeddings + paths
        Batch-9-m.01mnfws_zipped_2.npz   ← person B embeddings + paths
        Batch-9-m.01mnfws_zipped_clusters.csv

Usage
-----
    python organize_identities_npz.py \
        --npz-dir    /path/to/cleaned_npz \
        --iffy-csv   iffy.csv \
        --output-dir /path/to/output \
        --tolerance  0.6 \
        [--dry-run] [--log-file dry_run.txt]
"""

import sys
import csv
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Dry-run file logger ───────────────────────────────────────────────────────

class DryRunLogger:
    """
    Records every planned action and a final summary table into a
    plain-text report file when --dry-run + --log-file are both active.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lines   = []
        self._summary = []
        self._write_header()

    def _write_header(self):
        from datetime import datetime
        header = [
            "=" * 72,
            "  DRY-RUN REPORT — organize_identities_npz.py",
            f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            "=" * 72,
            "",
        ]
        self.log_path.write_text("\n".join(header), encoding="utf-8")

    def record_folder_start(self, identity_name: str, total: int):
        self._lines.append(
            f"\n{'─' * 60}\n"
            f"Identity: {identity_name}  ({total} embedding(s))\n"
            f"{'─' * 60}"
        )

    def record_cluster(self, identity_name: str, cluster_name: str, count: int):
        self._lines.append(
            f"  NPZ  →  {cluster_name}.npz  ({count} image(s))"
        )

    def record_csv(self, csv_name: str):
        self._lines.append(f"  CSV  →  {csv_name}")

    def record_folder_summary(self, identity_name: str, n_clusters: int, n_total: int):
        self._summary.append({
            "identity":  identity_name,
            "total":     n_total,
            "clusters":  n_clusters,
        })

    def flush(self, args_str: str):
        sections = []
        sections.append("PARAMETERS\n" + "-" * 40)
        sections.append(args_str)
        sections.append("")
        sections.append("PLANNED OUTPUTS\n" + "-" * 40)
        sections.extend(self._lines)
        sections.append("")
        sections.append("\nSUMMARY\n" + "-" * 40)

        col_w   = [45, 10, 10]
        headers = ["Identity Folder", "Embeddings", "Clusters"]
        sep     = "  ".join("-" * w for w in col_w)

        def fmt_row(vals):
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

        sections.append(fmt_row(headers))
        sections.append(sep)

        total_embeddings = total_clusters = 0
        for row in self._summary:
            sections.append(fmt_row([row["identity"], row["total"], row["clusters"]]))
            total_embeddings += row["total"]
            total_clusters   += row["clusters"]

        sections.append(sep)
        sections.append(fmt_row(["TOTAL", total_embeddings, total_clusters]))
        sections.append("")
        sections.append("No files were written — this was a dry run.")
        sections.append("=" * 72)

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(sections))
        log.info("Dry-run report written → %s", self.log_path)


# Module-level dry logger
_dry_logger: DryRunLogger = None


# ── Iffy CSV loader ───────────────────────────────────────────────────────────

def load_iffy_folders(csv_path: Path) -> list:
    """
    Read iffy.csv and return a list of (identity_name, full_folder_path) tuples.

    CSV format (with header):
        folder,total_similarity
        /data/.../Batch-9-m.01mnfws_zipped,1.87
        ...

    identity_name = Path(folder).name  →  "Batch-9-m.01mnfws_zipped"
    """
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["folder"].strip()
            if not raw:
                continue
            folder_path   = Path(raw)
            identity_name = folder_path.name
            entries.append((identity_name, folder_path))

    log.info("Loaded %d iffy folder(s) from '%s'", len(entries), csv_path)
    return entries


# ── Clustering ────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical vectors."""
    return float(1.0 - np.dot(a, b))


def cluster_embeddings(paths: list, embeddings: np.ndarray, tolerance: float) -> dict:
    """
    Greedy nearest-neighbour clustering by cosine distance.

    Parameters
    ----------
    paths      : list of strings — image paths (from NPZ)
    embeddings : np.ndarray shape (N, D), L2-normalised
    tolerance  : cosine-distance threshold (same person if dist ≤ tolerance)

    Returns
    -------
    {path_str: int}  cluster id (0-based) for each image path
    """
    cluster_ids             = [-1] * len(paths)
    next_cluster            = 0
    cluster_representatives = []

    for i, emb in enumerate(embeddings):
        if not cluster_representatives:
            cluster_ids[i] = next_cluster
            cluster_representatives.append(emb.copy())
            next_cluster += 1
            continue

        distances = np.array([cosine_distance(emb, r) for r in cluster_representatives])
        best_idx  = int(np.argmin(distances))

        if distances[best_idx] <= tolerance:
            cluster_ids[i] = best_idx
            n   = sum(1 for c in cluster_ids[:i+1] if c == best_idx)
            rep = (cluster_representatives[best_idx] * (n - 1) + emb) / n
            nrm = np.linalg.norm(rep)
            cluster_representatives[best_idx] = rep / nrm if nrm > 0 else rep
        else:
            cluster_ids[i] = next_cluster
            cluster_representatives.append(emb.copy())
            next_cluster += 1

    return {paths[i]: cluster_ids[i] for i in range(len(paths))}


# ── Per-folder logic ──────────────────────────────────────────────────────────

def process_folder(
    identity_name: str,
    npz_path: Path,
    output_dir: Path,
    tolerance: float,
    dry_run: bool,
):
    """
    Load the NPZ, cluster embeddings, then write:
      - one split NPZ per cluster  → output_dir/<identity>_1.npz, _2.npz ...
      - one CSV report             → output_dir/<identity>_clusters.csv
    """

    # ── Load NPZ ──────────────────────────────────────────────────────────────
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        log.warning("[%s]  Could not load NPZ: %s — skipping.", identity_name, e)
        return

    if not {"paths", "embeddings", "similarity_matrix"}.issubset(set(data.files)):
        log.warning("[%s]  NPZ missing required keys (got: %s) — skipping.",
                    identity_name, data.files)
        return

    stored_paths = list(data["paths"])        # list of strings
    embeddings   = data["embeddings"]         # (N, D)
    sim_matrix   = data["similarity_matrix"]  # (N, N)

    n_total = len(stored_paths)
    if n_total == 0:
        log.info("[%s]  Empty NPZ — skipping.", identity_name)
        return

    log.info("[%s]  %d embedding(s) loaded.", identity_name, n_total)

    if _dry_logger:
        _dry_logger.record_folder_start(identity_name, n_total)

    # ── Cluster ───────────────────────────────────────────────────────────────
    assignment = cluster_embeddings(stored_paths, embeddings, tolerance)
    # assignment: {path_str: cluster_id}

    clusters: dict = defaultdict(list)   # {cluster_id: [indices]}
    for idx, (path_str, cid) in enumerate(assignment.items()):
        clusters[cid].append(idx)

    n_clusters = len(clusters)
    log.info("[%s]  Distinct cluster(s) found: %d", identity_name, n_clusters)

    if dry_run:
        for cid in sorted(clusters.keys()):
            cluster_name = f"{identity_name}_{cid + 1}"
            count        = len(clusters[cid])
            log.info("[DRY-RUN]  Would write %s.npz  (%d image(s))", cluster_name, count)
            if _dry_logger:
                _dry_logger.record_cluster(identity_name, cluster_name, count)

        csv_name = f"{identity_name}_clusters.csv"
        log.info("[DRY-RUN]  Would write %s", csv_name)
        if _dry_logger:
            _dry_logger.record_csv(csv_name)
            _dry_logger.record_folder_summary(identity_name, n_clusters, n_total)
        return

    # ── Write split NPZ files ─────────────────────────────────────────────────
    stored_paths_arr = np.array(stored_paths)

    for cid in sorted(clusters.keys()):
        cluster_name = f"{identity_name}_{cid + 1}"
        indices      = clusters[cid]
        mask         = np.array(indices)

        cluster_paths = stored_paths_arr[mask]
        cluster_embs  = embeddings[mask]
        cluster_sim   = sim_matrix[np.ix_(mask, mask)]   # square sub-matrix

        out_npz = output_dir / f"{cluster_name}.npz"
        np.savez_compressed(
            str(out_npz),
            paths            = cluster_paths,
            embeddings       = cluster_embs,
            similarity_matrix = cluster_sim,
        )
        log.info("[%s]  Written %s  (%d image(s))", identity_name, out_npz.name, len(indices))

    # ── Write CSV report ──────────────────────────────────────────────────────
    csv_path = output_dir / f"{identity_name}_clusters.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "cluster"])
        for path_str, cid in assignment.items():
            cluster_name = f"{identity_name}_{cid + 1}"
            writer.writerow([path_str, cluster_name])

    log.info("[%s]  CSV report written → %s", identity_name, csv_path.name)

    if _dry_logger:
        _dry_logger.record_folder_summary(identity_name, n_clusters, n_total)


# ── Orchestration ─────────────────────────────────────────────────────────────

def organise(
    npz_dir: Path,
    iffy_csv: Path,
    output_dir: Path,
    tolerance: float,
    dry_run: bool,
    log_file: Path = None,
):
    global _dry_logger
    if dry_run and log_file:
        _dry_logger = DryRunLogger(log_file)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load iffy folders ─────────────────────────────────────────────────────
    entries = load_iffy_folders(iffy_csv)
    if not entries:
        log.warning("No folders found in '%s'.", iffy_csv)
        return

    # ── Match each folder name to its NPZ ─────────────────────────────────────
    matched = []
    for identity_name, folder_path in entries:
        npz_path = npz_dir / f"{identity_name}.npz"
        if npz_path.exists():
            matched.append((identity_name, npz_path))
        else:
            log.warning(
                "No NPZ found for '%s' (looked for: %s) — skipping.",
                identity_name, npz_path,
            )

    if not matched:
        log.warning("No matching NPZ files found in '%s'.", npz_dir)
        return

    log.info(
        "Iffy folders: %d  |  Matched NPZs: %d  |  Output: %s  |  Tolerance: %.2f",
        len(entries), len(matched), output_dir, tolerance,
    )
    if dry_run:
        log.info("*** DRY-RUN — no files will be written ***")

    for identity_name, npz_path in tqdm(
        matched, desc="Processing iffy folders", unit="folder"
    ):
        process_folder(identity_name, npz_path, output_dir, tolerance, dry_run)

    if _dry_logger:
        args_str = (
            f"  --npz-dir     {npz_dir}\n"
            f"  --iffy-csv    {iffy_csv}\n"
            f"  --output-dir  {output_dir}\n"
            f"  --tolerance   {tolerance}\n"
            f"  --dry-run     {dry_run}\n"
            f"  --log-file    {log_file}"
        )
        _dry_logger.flush(args_str)

    log.info("All done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split mixed identity NPZ files into per-person NPZ files and CSV reports "
            "using precomputed embeddings. No images or ONNX model needed."
        )
    )
    parser.add_argument(
        "--npz-dir",
        required=True,
        type=Path,
        help="Folder containing cleaned .npz files (one per identity folder).",
    )
    parser.add_argument(
        "--iffy-csv",
        required=True,
        type=Path,
        help=(
            "CSV with columns: folder, total_similarity. "
            "The 'folder' column must contain the full path to each identity folder. "
            "Only these folders will be processed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where split NPZ files and CSV reports will be saved.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help=(
            "Cosine-distance threshold for clustering. "
            "0.3 = strict · 0.6 = balanced (default) · 0.7 = lenient"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without actually writing any files.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to write the dry-run report. Only used with --dry-run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.log_file and not args.dry_run:
        log.warning("--log-file is ignored without --dry-run.")
    if not args.npz_dir.is_dir():
        sys.exit(f"ERROR: '{args.npz_dir}' is not a valid directory.")
    if not args.iffy_csv.is_file():
        sys.exit(f"ERROR: '{args.iffy_csv}' is not a valid file.")

    organise(
        args.npz_dir,
        args.iffy_csv,
        args.output_dir,
        args.tolerance,
        args.dry_run,
        args.log_file,
    )