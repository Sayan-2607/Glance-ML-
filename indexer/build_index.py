"""
indexer/build_index.py
------------------------
PART A: THE INDEXER

Usage:
    python -m indexer.build_index --images data/images --out outputs/index --backend offline
    python -m indexer.build_index --images data/images --out outputs/index --backend clip

For every image in `--images`:
  1. Compute a global embedding (embedding_backend.encode_image) -> stored
     in the vector index for fast approximate nearest-neighbour recall.
  2. Compute a structured attribute record (attribute_extractor.extract_attributes)
     -> stored in the metadata store for compositional re-ranking.

This keeps "feature extraction" (ML logic, swappable backend) completely
separate from "vector storage" (infra, swappable store) and from the
retrieval logic (Part B) -- each can be changed/tested independently.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from indexer.embedding_backend import get_backend  # noqa: E402
from indexer.attribute_extractor import extract_attributes  # noqa: E402
from vectorstore.store import FlatIndex, MetadataStore  # noqa: E402


def build_index(images_dir: str, out_prefix: str, backend_name: str, limit: int | None = None):
    backend = get_backend(backend_name)
    filenames = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if limit:
        filenames = filenames[:limit]

    print(f"[indexer] backend={backend_name}  images={len(filenames)}")

    sample_vec = backend.encode_image(Image.open(os.path.join(images_dir, filenames[0])).convert("RGB"))
    index = FlatIndex(dim=sample_vec.shape[0])
    meta = MetadataStore()

    ids, vectors = [], []
    t0 = time.time()
    for i, fname in enumerate(filenames):
        path = os.path.join(images_dir, fname)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
            continue

        vec = backend.encode_image(image)
        attrs = extract_attributes(backend, image)

        image_id = os.path.splitext(fname)[0]
        ids.append(image_id)
        vectors.append(vec)
        meta.add(image_id, path, attrs)

        if (i + 1) % 100 == 0 or (i + 1) == len(filenames):
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(filenames)}] elapsed={elapsed:.1f}s")

    import numpy as np
    index.add(ids, np.stack(vectors))
    index.save(out_prefix)
    meta.save(f"{out_prefix}.meta.json")
    print(f"[indexer] saved index -> {out_prefix}.vectors.npy / .ids.json")
    print(f"[indexer] saved metadata -> {out_prefix}.meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="directory of images to index")
    parser.add_argument("--out", required=True, help="output path prefix, e.g. outputs/index")
    parser.add_argument("--backend", default=config.DEFAULT_BACKEND, choices=["clip", "offline"])
    parser.add_argument("--limit", type=int, default=None, help="only index the first N images (debug)")
    args = parser.parse_args()
    build_index(args.images, args.out, args.backend, args.limit)
