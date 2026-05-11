# Face Duplicate Detection Pipeline

A GPU-accelerated Python pipeline that detects and clusters duplicate identities across a large facial image dataset. It uses a pretrained ONNX face recognition model to extract 512-dimensional embeddings, then runs approximate nearest-neighbor search via FAISS to find and group matching identities.

## Requirements

- Windows (paths and virtual environment are Windows-specific)
- Python 3.10
- NVIDIA GPU with CUDA (required for Stage 1)
- ONNX face recognition model at `models`
- Dataset at `data\` (one subfolder per identity, containing `.jpg`/`.png`/`.jpeg` images)


## Running the Pipeline

Run each script in order. Each stage reads the outputs of the previous one.

```powershell
# Stage 1 — Extract per-image embeddings (GPU, takes hours on large datasets)
python distributed_embedding_extraction.py

# Stage 2 — Compute per-identity mean embeddings
python build_identity_mean_embeddings.py

# Stage 3 — Build FAISS IVF-PQ index
python Faiss.py

# Stage 4 — Search for duplicate identity pairs
python duplicate_search.py

# Stage 5 — Cluster duplicates via Union-Find
python graph_clustering_duplicates.py
```

## Pipeline Stages

### Stage 1 — Distributed Embedding Extraction

Reads up to 25 images per identity folder, preprocesses them (resize to 112×112, normalize to `[-1, 1]`), and runs batched ONNX inference across 3 GPU workers fed by 24 CPU loader processes. A dedicated writer process flushes results to a memory-mapped array.

**Output:** `./embedding_store/embeddings.memmap`, `./embedding_store/image_paths.npy`

### Stage 2 — Identity Mean Embeddings

Groups embeddings by identity folder name. Filters out noisy images (those with average pairwise cosine similarity < 0.3 within the identity), then computes a normalized mean embedding per identity. Skips identities with fewer than 2 images.

**Output:** `./identity_store/identity_embeddings.npy`, `identity_names.npy`, `identity_counts.npy`

### Stage 3 — FAISS Index

Builds an IVF-PQ index (4096 Voronoi cells, M=64 sub-quantizers, 8-bit codes) trained on up to 500k random embeddings, then adds all identity embeddings. Sets `nprobe=32` for search.

**Output:** `./faiss_store/identity_ivfpq.index`

### Stage 4 — Duplicate Search

Queries each identity against its top-50 FAISS neighbors. Retains pairs with cosine similarity ≥ 0.72, excluding self-matches. Outputs sorted by descending similarity.

**Output:** `./faiss_store/duplicate_pairs.npy` — array of `(identity_a, identity_b, similarity)` triples

### Stage 5 — Graph Clustering

Runs Union-Find on all duplicate pairs. Before merging two clusters, verifies that their centroids have cosine similarity ≥ 0.78 — this prevents over-merging. Saves each cluster as a `members.txt` file listing identity names.

**Output:** `./clusters/cluster_<id>/members.txt`

## Configuration

All parameters are hardcoded at the top of each script. To change a value, open the relevant script and edit the constant at the top of the file.

### Stage 1 — `distributed_embedding_extraction.py`

| Parameter | Value | What it does |
|---|---|---|
| `MAX_IMAGES_PER_ID` | 25 | Maximum number of images sampled from each identity folder. Caps memory and compute per identity; identities with more images are silently truncated. |
| `EMBEDDING_DIM` | 512 | Dimensionality of the embedding vectors produced by the ONNX model. Must match the model's output shape. |
| `CPU_IMAGE_WORKERS` | 24 | Number of parallel CPU processes that read and preprocess images. Scale with available CPU cores. |
| `NUM_GPU_WORKERS` | 3 | Number of parallel ONNX inference processes, one per logical GPU. Each holds its own CUDA session. Set to 1 if you have a single GPU. |
| `BATCH_SIZE` | 256 | Number of images fed to the ONNX model in a single forward pass. Larger batches improve GPU utilisation but increase VRAM usage. |
| `QUEUE_SIZE` | 2048 | Maximum number of preprocessed images buffered in the inter-process queue between CPU loaders and GPU workers. Prevents CPU workers from running too far ahead. |

### Stage 2 — `build_identity_mean_embeddings.py`

| Parameter | Value | What it does |
|---|---|---|
| `MIN_IMAGES_PER_ID` | 2 | Identities with fewer than this many images are skipped entirely. A single image is not enough to compute a reliable mean embedding. |
| `NOISE_FILTER_THRESHOLD` | 0.30 | An image is considered noisy and dropped if its average cosine similarity to all other images of the same identity falls below this value. Removes mislabeled or heavily occluded faces that would corrupt the mean. Lowering this keeps more images; raising it enforces stricter consistency. |

### Stage 3 — `Faiss.py`

| Parameter | Value | What it does |
|---|---|---|
| `N_LIST` | 4096 | Number of Voronoi cells (clusters) in the IVF index. More cells mean faster search but require more training data and memory. Recommended to be `sqrt(N)` to `4*sqrt(N)` for the dataset size. |
| `M` | 64 | Number of sub-quantizers used by Product Quantization. Higher M preserves more distance precision at the cost of more memory per vector. |
| `BITS` | 8 | Number of bits per sub-quantizer code. 8 bits = 256 centroids per sub-vector; standard choice for most use cases. |
| `nprobe` | 32 | Number of Voronoi cells visited during a search query. Higher values improve recall (fewer missed duplicates) but increase search time. Set at index build time and carried into Stage 4. |

### Stage 4 — `duplicate_search.py`

| Parameter | Value | What it does |
|---|---|---|
| `TOP_K` | 50 | Number of nearest neighbors retrieved from FAISS per identity. Must be large enough that all true duplicates fall within the top K. |
| `SIM_THRESHOLD` | 0.72 | Minimum cosine similarity for two identities to be considered duplicates. Pairs below this score are discarded. **Lowering this produces more (but noisier) duplicate pairs; raising it produces fewer (but more confident) pairs.** This is the most impactful tuning knob in the pipeline. |

### Stage 5 — `graph_clustering_duplicates.py`

| Parameter | Value | What it does |
|---|---|---|
| `CENTROID_THRESHOLD` | 0.78 | Before two clusters are merged by Union-Find, their mean embeddings (centroids) must have cosine similarity ≥ this value. Acts as a guard against transitive over-merging — where A≈B and B≈C does not necessarily mean A≈C. Should be set higher than `SIM_THRESHOLD` to enforce tighter cluster coherence. |
