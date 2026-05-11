# ann_duplicate_search.py

import os
import time
import faiss
import numpy as np
from tqdm import tqdm

_start = time.time()


def _fmt(secs):
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# CONFIG


INDEX_PATH = (
    "./faiss_store/identity_ivfpq.index"
)

IDENTITY_EMBEDDINGS_PATH = (
    "./identity_store/identity_embeddings.npy"
)

IDENTITY_NAMES_PATH = (
    "./identity_store/identity_names.npy"
)

OUTPUT_PATH = "./faiss_store/duplicate_pairs.npy"


# SETTINGS


TOP_K = 100
SIM_THRESHOLD = 0.4


# LOAD


index = faiss.read_index(INDEX_PATH)

embeddings = np.load(
    IDENTITY_EMBEDDINGS_PATH
).astype("float32")

identity_names = np.load(
    IDENTITY_NAMES_PATH,
    allow_pickle=True
)

faiss.normalize_L2(embeddings)


# SEARCH

print("Searching...")
_t0 = time.time()

D, I = index.search(
    embeddings,
    TOP_K
)
print(f"FAISS search time: {_fmt(time.time() - _t0)}")

duplicate_pairs = set()

for i in tqdm(range(len(embeddings))):

    source_identity = identity_names[i]

    for rank in range(1, TOP_K):

        target_idx = I[i][rank]

        if target_idx == -1:
            continue

        sim = D[i][rank]

        if sim < SIM_THRESHOLD:
            continue

        target_identity = identity_names[target_idx]

        if source_identity == target_identity:
            continue

        pair = tuple(
            sorted(
                [
                    source_identity,
                    target_identity
                ]
            )
        )

        duplicate_pairs.add(
            (
                pair[0],
                pair[1],
                float(sim)
            )
        )

duplicate_pairs = list(duplicate_pairs)

duplicate_pairs.sort(
    key=lambda x: x[2],
    reverse=True
)

np.save(
    OUTPUT_PATH,
    np.array(duplicate_pairs, dtype=object)
)

elapsed = time.time() - _start
print("===================================")
print("Duplicate pairs:", len(duplicate_pairs))
print(f"Total time: {_fmt(elapsed)}")