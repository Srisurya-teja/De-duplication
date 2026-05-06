import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from insightface.model_zoo import get_model
import shutil
import multiprocessing
import os

# Detect total CPU cores
total_cores = multiprocessing.cpu_count()
print(f"Total CPU cores available: {total_cores}")

# CONFIG

DATASET_PATH = "/data/datasets-san-backup/buffalo-crop-dedupe"
SAMPLE_SIZE = 100
MAX_IMAGES_PER_ID = 7

MODEL_PATH = "/home/sima/srisurya/models/model.onnx"
THRESHOLD = 0.35

# LOAD MODEL

model = get_model(MODEL_PATH)
model.prepare(ctx_id=0)  # use -1 for CPU


# HELPERS

def get_images(identity_path):
    return [
        os.path.join(identity_path, f)
        for f in os.listdir(identity_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ][:MAX_IMAGES_PER_ID]


def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))

    emb = model.get_feat(img)
    emb = emb.flatten()
    emb = emb / np.linalg.norm(emb)

    return emb


def filter_embeddings(embs, threshold=0.3):
    """Remove noisy images inside a folder"""
    embs = np.stack(embs)

    sims = np.dot(embs, embs.T)

    keep = []
    for i in range(len(embs)):
        avg_sim = (np.sum(sims[i]) - 1) / (len(embs) - 1)
        if avg_sim > threshold:
            keep.append(embs[i])

    if len(keep) == 0:
        return embs

    return np.stack(keep)


def get_identity_embedding(embs):
    embs = filter_embeddings(embs)
    mean_emb = np.mean(embs, axis=0)
    return mean_emb / np.linalg.norm(mean_emb)



# SAMPLE IDENTITIES

all_ids = [
    os.path.join(DATASET_PATH, d)
    for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]

sample_ids = random.sample(all_ids, min(SAMPLE_SIZE, len(all_ids)))
print(f"Sampled {len(sample_ids)} identities")


# COMPUTE IDENTITY EMBEDDINGS

identity_embeddings = {}
valid_ids = []

for identity in tqdm(sample_ids):
    imgs = get_images(identity)

    embs = []
    for img in imgs:
        e = get_embedding(img)
        if e is not None:
            embs.append(e)

    if len(embs) >= 2:
        identity_embeddings[identity] = get_identity_embedding(embs)
        valid_ids.append(identity)

print(f"Valid identities: {len(valid_ids)}")


# DUPLICATE DETECTION

duplicates = []

for i in range(len(valid_ids)):
    for j in range(i + 1, len(valid_ids)):
        emb1 = identity_embeddings[valid_ids[i]]
        emb2 = identity_embeddings[valid_ids[j]]

        sim = np.dot(emb1, emb2)

        if sim > THRESHOLD:
            duplicates.append((valid_ids[i], valid_ids[j], float(sim)))

# sort by similarity
duplicates.sort(key=lambda x: x[2], reverse=True)
print(len(duplicates))

print("\n=== TOP DUPLICATES ===")
for d in duplicates[:20]:
    print(f"{d[0]}  <-->  {d[1]}   sim={d[2]:.3f}")


# CLUSTERING (UNION-FIND)

parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[py] = px

# initialize
for i in valid_ids:
    parent[i] = i

# merge duplicates
for a, b, _ in duplicates:
    union(a, b)

# build clusters
clusters = {}
for i in valid_ids:
    root = find(i)
    clusters.setdefault(root, []).append(i)


# PRINT CLUSTERS

print("\n=== FINAL CLUSTERS ===")

for cluster_id, members in clusters.items():
    if len(members) > 1:
        print(f"\nCluster ({len(members)} identities):")
        for m in members:
            print("  ", m)

OUTPUT_DIR = "clustered_identities"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# COPY IMAGES INTO CLUSTER FOLDERS

for cluster_idx, (cluster_id, members) in enumerate(clusters.items(), 1):
    if len(members) < 2:
        # skip clusters with only 1 identity
        continue

    cluster_folder = os.path.join(OUTPUT_DIR, f"cluster_{cluster_idx}")
    os.makedirs(cluster_folder, exist_ok=True)

    for identity_path in members:
        identity_name = os.path.basename(identity_path)
        dest_identity_folder = os.path.join(cluster_folder, identity_name)
        os.makedirs(dest_identity_folder, exist_ok=True)

        # copy images of this identity
        for img_file in os.listdir(identity_path):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(identity_path, img_file)
                dst = os.path.join(dest_identity_folder, img_file)
                shutil.copy2(src, dst)  # preserves metadata

print(f"All cluster images copied to '{OUTPUT_DIR}'")