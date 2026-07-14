"""
vectorstore/store.py
---------------------
Deliberately the simplest possible "vector DB": a flat matrix of
L2-normalized embeddings + a parallel JSON metadata file (image path +
structured attributes from attribute_extractor.py). Cosine similarity on
normalized vectors == dot product, so a brute-force matmul against the
whole matrix is used for search.

Why not a "real" vector database? The assignment explicitly asks to
prioritize ML logic over infra, and to pick "the easiest and most
convenient" option. For a few thousand to a few hundred thousand images, a
flat in-memory matrix + matmul is faster to build, easier to debug, and has
zero moving parts (no server, no client library version issues) compared
to standing up Milvus/Pinecone/Weaviate/Chroma for this exercise.

FAISS integration point: if `faiss` is installed, `FlatIndex.search()`
transparently uses `faiss.IndexFlatIP` instead of the numpy matmul -- same
results, faster at scale (near-drop-in). This is exactly the interface
`build_index.py` / `search.py` use, so scaling this project up later is a
one-line change, not a rewrite (see "Scalability" in the write-up for how
this would evolve towards IndexIVFPQ / HNSW at the 1M-image mark).
"""
from __future__ import annotations
import json
import os
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss_index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None

    def add(self, ids: List[str], vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        self._vectors = np.vstack([self._vectors, vectors]) if self.ids else vectors
        self.ids.extend(ids)
        if _HAS_FAISS:
            self._faiss_index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        if _HAS_FAISS:
            scores, idxs = self._faiss_index.search(query_vec, min(k, len(self.ids)))
            return [(self.ids[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
        # numpy fallback: cosine similarity via dot product (vectors are unit-normalized upstream)
        sims = self._vectors @ query_vec[0]
        top_idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top_idx]

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        np.save(f"{path_prefix}.vectors.npy", self._vectors)
        with open(f"{path_prefix}.ids.json", "w") as f:
            json.dump(self.ids, f)

    @classmethod
    def load(cls, path_prefix: str) -> "FlatIndex":
        vectors = np.load(f"{path_prefix}.vectors.npy")
        with open(f"{path_prefix}.ids.json") as f:
            ids = json.load(f)
        idx = cls(dim=vectors.shape[1])
        idx.add(ids, vectors)
        return idx


class MetadataStore:
    """Plain JSON-lines store: {id -> {path, attributes}}. Swap for SQLite
    or a parquet file without touching any other module if this needs to
    scale past what comfortably fits in memory as JSON."""

    def __init__(self):
        self.records = {}

    def add(self, image_id: str, path: str, attributes: dict):
        self.records[image_id] = {"path": path, "attributes": attributes}

    def get(self, image_id: str) -> dict:
        return self.records[image_id]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetadataStore":
        store = cls()
        with open(path) as f:
            store.records = json.load(f)
        return store
