import os
import cv2
import math
import queue
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
from tqdm import tqdm
import onnxruntime as ort

_start = time.time()


def _fmt(secs):
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# CONFIG


DATASET_PATH = r"C:\Users\smaruboina\Downloads\data"
MODEL_PATH = r"C:\Users\smaruboina\Downloads\model.onnx"
OUTPUT_DIR = "embedding_store"
BATCH_SIZE = 512
NUM_GPU_WORKERS = 1
CPU_IMAGE_WORKERS = 20
MAX_IMAGES_PER_ID = 25
EMBEDDING_DIM = 512
QUEUE_SIZE = 4096


# OUTPUTS


os.makedirs(OUTPUT_DIR, exist_ok=True)

MEMMAP_PATH = os.path.join(OUTPUT_DIR, "embeddings.memmap")
PATHS_PATH = os.path.join(OUTPUT_DIR, "image_paths.npy")


# IMAGE COLLECTION
# Runs in every spawned worker process to populate the all_images global
# needed by image_loader_worker. No file writes here — safe to re-run.


all_images = []
all_ids = [
    os.path.join(DATASET_PATH, d)
    for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]
for identity_path in all_ids:
    images = [
        os.path.join(identity_path, f)
        for f in os.listdir(identity_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ][:MAX_IMAGES_PER_ID]
    all_images.extend(images)

NUM_IMAGES = len(all_images)


# IMAGE PREPROCESSING


def preprocess_image(img_path):
    try:

        img = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
    except Exception:
        return None
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img


# CPU IMAGE LOADER


def image_loader_worker(image_indices, output_queue):
    for idx in image_indices:
        img_path = all_images[idx]
        img = preprocess_image(img_path)
        if img is None:
            continue
        output_queue.put((idx, img))


# GPU EMBEDDING WORKER


def gpu_worker(gpu_id, input_queue, output_queue):
    providers = [("CUDAExecutionProvider", {"device_id": gpu_id})]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    batch_imgs = []
    batch_indices = []

    while True:
        try:
            item = input_queue.get(timeout=5)
        except queue.Empty:
            item = None

        if item is None:
            break

        idx, img = item
        batch_imgs.append(img)
        batch_indices.append(idx)

        if len(batch_imgs) >= BATCH_SIZE:
            batch = np.stack(batch_imgs).astype(np.float32)
            embeddings = session.run(None, {input_name: batch})[0]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            output_queue.put((batch_indices, embeddings.astype("float32")))
            batch_imgs = []
            batch_indices = []

    # flush remainder
    if len(batch_imgs) > 0:
        batch = np.stack(batch_imgs).astype(np.float32)
        embeddings = session.run(None, {input_name: batch})[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        output_queue.put((batch_indices, embeddings.astype("float32")))

    output_queue.put(None)


# WRITER PROCESS
# Opens its own memmap handle (mode="r+") since the main process
# already created the file with mode="w+" before spawning workers.


def writer_worker(output_queue, memmap_path, num_images, emb_dim):
    store = np.memmap(
        memmap_path,
        dtype="float32",
        mode="r+",
        shape=(num_images, emb_dim)
    )
    finished_gpus = 0
    pbar = tqdm(total=num_images)
    while True:
        item = output_queue.get()
        if item is None:
            finished_gpus += 1
            if finished_gpus == NUM_GPU_WORKERS:
                break
            continue
        indices, embeddings = item
        store[indices] = embeddings
        pbar.update(len(indices))
    store.flush()
    pbar.close()


# MAIN


if __name__ == "__main__":

    print(f"Total images: {NUM_IMAGES}")

    np.save(PATHS_PATH, np.array(all_images))

    # Create the memmap file in the main process, then release the handle.
    # writer_worker opens its own r+ handle after the file exists.
    embedding_store = np.memmap(
        MEMMAP_PATH,
        dtype="float32",
        mode="w+",
        shape=(NUM_IMAGES, EMBEDDING_DIM)
    )
    del embedding_store

    mp.set_start_method("spawn", force=True)

    image_queue = Queue(maxsize=QUEUE_SIZE)
    embedding_queue = Queue(maxsize=QUEUE_SIZE)


    # CPU IMAGE WORKERS


    image_chunks = np.array_split(np.arange(NUM_IMAGES), CPU_IMAGE_WORKERS)
    cpu_processes = []
    for chunk in image_chunks:
        p = Process(target=image_loader_worker, args=(chunk, image_queue))
        p.start()
        cpu_processes.append(p)


    # GPU WORKERS


    gpu_processes = []
    for gpu_id in range(NUM_GPU_WORKERS):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, image_queue, embedding_queue)
        )
        p.start()
        gpu_processes.append(p)


    # WRITER


    writer = Process(
        target=writer_worker,
        args=(embedding_queue, MEMMAP_PATH, NUM_IMAGES, EMBEDDING_DIM)
    )
    writer.start()


    # JOIN


    for p in cpu_processes:
        p.join()

    for _ in range(NUM_GPU_WORKERS):
        image_queue.put(None)

    for p in gpu_processes:
        p.join()

    writer.join()

    elapsed = time.time() - _start
    print("\nDONE")
    print("===================================")
    print(f"Total time: {_fmt(elapsed)}")