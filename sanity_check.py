import os
import torch
import random
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from insightface.model_zoo import get_model
import matplotlib.pyplot as plt


# CONFIG

DATASET_PATH = "/data/datasets-san-backup/buffalo-crop-dedupe"
SAMPLE_SIZE = 100
MAX_IMAGES_PER_ID = 7

MODEL_PATH = "/home/sima/srisurya/models/model.onnx"  # ONNX model path


# LOAD MODEL

model = get_model(MODEL_PATH)
model.prepare(ctx_id=0)  # use -1 for CPU if needed


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

    # Convert BGR → RGB (important)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, (112, 112))

    emb = model.get_feat(img)

    # Flatten (IMPORTANT FIX)
    emb = emb.flatten()

    # Normalize
    emb = emb / np.linalg.norm(emb)

    return emb


def get_identity_embedding(embs):
    # Stack embeddings on GPU
    embs = torch.tensor(np.stack(embs), dtype=torch.float32, device='cuda')
    mean_emb = torch.mean(embs, dim=0)
    mean_emb = mean_emb / torch.norm(mean_emb)
    return mean_emb.cpu().numpy()  # m


# SAMPLE IDENTITIES

all_ids = [
    os.path.join(DATASET_PATH, d)
    for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]

sample_ids = random.sample(all_ids, min(SAMPLE_SIZE, len(all_ids)))

print(f"Sampled {len(sample_ids)} identities")


# COMPUTE EMBEDDINGS

intra_sims = []
inter_sims = []

embeddings_per_id = {}


for identity in tqdm(sample_ids):
    imgs = get_images(identity)

    embs = []
    for img in imgs:
        e = get_embedding(img)
        if e is not None:
            embs.append(e)

    if len(embs) >= 2:
        embeddings_per_id[identity] = embs

print(f"Valid identities after filtering: {len(embeddings_per_id)}")


# INTRA SIMILARITY (within identity)

for identity, embs in embeddings_per_id.items():
    embs = np.stack(embs)  # IMPORTANT
    sims = cosine_similarity(embs)

    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            intra_sims.append(sims[i, j])


# IDENTITY EMBEDDINGS

identity_embeddings = {
    k: get_identity_embedding(v)
    for k, v in embeddings_per_id.items()
}

# Debug check
sample_emb = list(identity_embeddings.values())[0]
print("Embedding dimension:", sample_emb.shape)


# INTER SIMILARITY (between identities)

ids = list(identity_embeddings.keys())

for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        emb1 = identity_embeddings[ids[i]]
        emb2 = identity_embeddings[ids[j]]

        sim = np.dot(emb1, emb2)
        inter_sims.append(sim)


# RESULTS

print("\n=== SANITY CHECK RESULTS ===")

print("\nIntra (same person):")
print(f"  mean = {np.mean(intra_sims):.3f}")
print(f"  min  = {np.min(intra_sims):.3f}")
print(f"  max  = {np.max(intra_sims):.3f}")

print("\nInter (different people):")
print(f"  mean = {np.mean(inter_sims):.3f}")
print(f"  min  = {np.min(inter_sims):.3f}")
print(f"  max  = {np.max(inter_sims):.3f}")


# PERCENTILES (important for threshold tuning)

print("\n=== DISTRIBUTION ===")

print("\nIntra percentiles:")
print(f"  25% = {np.percentile(intra_sims, 25):.3f}")
print(f"  50% = {np.percentile(intra_sims, 50):.3f}")
print(f"  75% = {np.percentile(intra_sims, 75):.3f}")

print("\nInter percentiles:")
print(f"  25% = {np.percentile(inter_sims, 25):.3f}")
print(f"  50% = {np.percentile(inter_sims, 50):.3f}")
print(f"  75% = {np.percentile(inter_sims, 75):.3f}")




# PLOT HISTOGRAMS

plt.figure(figsize=(10, 6))

# Plot intra-person similarities
plt.hist(intra_sims, bins=50, alpha=0.6, color='blue', label='Intra-person')

# Plot inter-person similarities
plt.hist(inter_sims, bins=50, alpha=0.6, color='red', label='Inter-person')

plt.title('Similarity Distributions')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig("similarity_histogram.png")
print("Plot saved as similarity_histogram.png")