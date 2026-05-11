
import os
import time
import numpy as np
from tqdm import tqdm

_start = time.time()


def _fmt(secs):
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# CONFIG


EMBEDDING_STORE = "./embedding_store/embeddings.memmap"
IMAGE_PATHS = "./embedding_store/image_paths.npy"
OUTPUT_DIR = "./identity_store"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IDENTITY_EMBEDDINGS_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_embeddings.npy"
)

IDENTITY_NAMES_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_names.npy"
)

IDENTITY_COUNTS_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_counts.npy"
)


# SETTINGS


EMBEDDING_DIM = 512
NOISE_FILTER_THRESHOLD = 0.3
MIN_IMAGES_PER_ID = 2

# LOAD


image_paths = np.load(
    IMAGE_PATHS,
    allow_pickle=True
)

num_images = len(image_paths)

embeddings = np.memmap(
    EMBEDDING_STORE,
    dtype="float32",
    mode="r",
    shape=(num_images, EMBEDDING_DIM)
)


# GROUP BY IDENTITY


identity_to_indices = {}

for idx, img_path in enumerate(image_paths):

    identity_name = os.path.basename(
        os.path.dirname(img_path)
    )

    if identity_name not in identity_to_indices:
        identity_to_indices[identity_name] = []

    identity_to_indices[identity_name].append(idx)

print(f"Total identities: {len(identity_to_indices)}")


# HELPERS


def normalize(x):

    norm = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True
    )
    return x / (norm + 1e-8)


# BUILD MEAN EMBEDDINGS


identity_embeddings = []
identity_names = []
identity_counts = []

for identity_name, indices in tqdm(
    identity_to_indices.items()
):

    if len(indices) < MIN_IMAGES_PER_ID:
        continue
    embs = embeddings[indices]

    embs = normalize(embs)


    # REMOVE NOISY IMAGES


    if len(embs) > 2:

        sims = np.dot(embs, embs.T)

        avg_sims = (
            np.sum(sims, axis=1) - 1
        ) / (len(embs) - 1)

        keep_mask = avg_sims > NOISE_FILTER_THRESHOLD

        filtered_embs = embs[keep_mask]

        if len(filtered_embs) >= 2:
            embs = filtered_embs


    # MEAN EMBEDDING


    mean_emb = np.mean(
        embs,
        axis=0
    )

    mean_emb = mean_emb / (
    np.linalg.norm(mean_emb) + 1e-8
    )

    identity_embeddings.append(
        mean_emb.astype("float32")
    )

    identity_names.append(identity_name)

    identity_counts.append(len(embs))


# SAVE


identity_embeddings = np.stack(identity_embeddings)

np.save(
    IDENTITY_EMBEDDINGS_PATH,
    identity_embeddings
)

np.save(
    IDENTITY_NAMES_PATH,
    np.array(identity_names)
)

np.save(
    IDENTITY_COUNTS_PATH,
    np.array(identity_counts)
)

elapsed = time.time() - _start
print("===================================")
print("Identity embeddings:", identity_embeddings.shape)
print("Saved successfully")
print(f"Total time: {_fmt(elapsed)}")