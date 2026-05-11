# build_faiss_ivfpq.py
import os
import time
import faiss
import numpy as np

_start = time.time()


def _fmt(secs):
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

# CONFIG

IDENTITY_EMBEDDINGS_PATH = (
    "./identity_store/identity_embeddings.npy"
)

FAISS_DIR = "./faiss_store"

os.makedirs(FAISS_DIR, exist_ok=True)

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "identity_ivfpq.index"
)


# SETTINGS

N_LIST = 1024
M = 128
BITS = 8


# LOAD

embeddings = np.load(
    IDENTITY_EMBEDDINGS_PATH
).astype("float32")

faiss.normalize_L2(embeddings)
num_embeddings, dim = embeddings.shape
print("Embeddings:", embeddings.shape)

# BUILD INDEX


quantizer = faiss.IndexFlatIP(dim)

index = faiss.IndexIVFPQ(
    quantizer,
    dim,
    N_LIST,
    M,
    BITS
)


# TRAIN


print("Training index...")
_t0 = time.time()

random_subset_size = min(
    500000,
    num_embeddings
)

random_indices = np.random.choice(
    num_embeddings,
    random_subset_size,
    replace=False
)

train_data = embeddings[random_indices]

index.train(train_data)
print(f"Training time: {_fmt(time.time() - _t0)}")

print("Adding vectors...")
_t1 = time.time()

index.add(embeddings)
print(f"Add time: {_fmt(time.time() - _t1)}")

# SEARCH SETTINGS

index.nprobe = 64


# SAVE

faiss.write_index(
    index,
    INDEX_PATH
)

elapsed = time.time() - _start
print("===================================")
print("FAISS IVF-PQ index saved")
print(f"Total time: {_fmt(elapsed)}")