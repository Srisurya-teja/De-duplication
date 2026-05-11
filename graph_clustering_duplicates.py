import os
import time
import numpy as np
from collections import defaultdict

_start = time.time()


def _fmt(secs):
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# CONFIG

DUPLICATES_PATH = (
    "./faiss_store/duplicate_pairs.npy"
)
IDENTITY_EMBEDDINGS_PATH = (
    "./identity_store/identity_embeddings.npy"
)
IDENTITY_NAMES_PATH = (
    "./identity_store/identity_names.npy"
)

OUTPUT_DIR = "./clusters"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# SETTINGS


CENTROID_THRESHOLD = 0.46


# LOAD


duplicates = np.load(
    DUPLICATES_PATH,
    allow_pickle=True
)

identity_embeddings = np.load(
    IDENTITY_EMBEDDINGS_PATH
).astype("float32")

identity_names = np.load(
    IDENTITY_NAMES_PATH,
    allow_pickle=True
)

name_to_embedding = {
    name: emb
    for name, emb in zip(
        identity_names,
        identity_embeddings
    )
}


# UNION FIND


class UnionFind:

    def __init__(self):

        self.parent = {}
        self.members = {}

    def add(self, x):

        if x not in self.parent:

            self.parent[x] = x
            self.members[x] = [x]

    def find(self, x):

        if self.parent[x] != x:

            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def get_centroid(self, root):

        embs = [
            name_to_embedding[m]
            for m in self.members[root]
        ]

        centroid = np.mean(
            embs,
            axis=0
        )

        centroid /= (
            np.linalg.norm(centroid) + 1e-8
        )

        return centroid

    def union(self, x, y):

        px = self.find(x)
        py = self.find(y)

        if px == py:
            return

   
        # CENTROID VERIFICATION
   

        centroid_x = self.get_centroid(px)
        centroid_y = self.get_centroid(py)

        sim = np.dot(
            centroid_x,
            centroid_y
        )

        if sim < CENTROID_THRESHOLD:
            return

        # merge smaller into larger

        if len(self.members[px]) < len(self.members[py]):
            px, py = py, px

        self.parent[py] = px

        self.members[px].extend(
            self.members[py]
        )

        del self.members[py]


# BUILD GRAPH


uf = UnionFind()

for a, b, sim in duplicates:

    uf.add(a)
    uf.add(b)

    uf.union(a, b)


# BUILD FINAL CLUSTERS


clusters = defaultdict(list)

for node in uf.parent.keys():

    root = uf.find(node)

    clusters[root].append(node)


# SAVE


cluster_sizes = []

for idx, members in enumerate(clusters.values()):

    cluster_dir = os.path.join(
        OUTPUT_DIR,
        f"cluster_{idx}"
    )

    os.makedirs(cluster_dir, exist_ok=True)

    with open(
        os.path.join(cluster_dir, "members.txt"),
        "w"
    ) as f:

        for m in members:
            f.write(m + "\n")

    cluster_sizes.append(len(members))


cluster_sizes = np.array(cluster_sizes)

elapsed = time.time() - _start
print("===================================")
print("Clusters:", len(cluster_sizes))
print("Largest cluster:", cluster_sizes.max())
print("Average cluster:", cluster_sizes.mean())
print(f"Total time: {_fmt(elapsed)}")