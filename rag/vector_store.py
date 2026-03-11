"""
vector_store.py — FAISS-backed vector store for RC-TGAD (Person 2)

Stores embeddings z with their ground-truth labels.
Used by H_RAG to retrieve k nearest neighbors and compute label entropy.
"""

import faiss
import numpy as np
from typing import List, Dict


class VectorStore:
    """
    FAISS L2 index that stores embeddings + binary labels (0=normal, 1=anomaly).

    Usage:
        store = VectorStore(dim=64)
        store.add(z_numpy, label=0)
        neighbors = store.query(z_numpy, k=10)
        # neighbors: [{'label': int, 'dist': float}, ...]
    """

    def __init__(self, dim: int = 64):
        """
        Args:
            dim: Embedding dimensionality. Must match backbone d_z (agreed as 64 on Day 1).
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)   # exact L2 search — fine at this scale
        self.labels: List[int] = []            # parallel list to FAISS internal storage

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, z: np.ndarray, label: int) -> None:
        """
        Add one embedding to the store.

        Args:
            z:     1-D numpy array of shape (dim,) OR torch.Tensor.
                   Will be cast to float32 internally.
            label: Ground-truth label — 0 (normal) or 1 (anomaly).
        """
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        if z_np.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {z_np.shape[1]}"
            )
        self.index.add(z_np)
        self.labels.append(int(label))

    def add_batch(self, zs: np.ndarray, labels: List[int]) -> None:
        """
        Bulk add — slightly faster than calling add() in a loop.

        Args:
            zs:     2-D array shape (N, dim).
            labels: List of N integer labels.
        """
        zs_np = _to_numpy(zs).astype("float32")
        assert zs_np.shape[0] == len(labels), "zs and labels must have same length"
        self.index.add(zs_np)
        self.labels.extend([int(l) for l in labels])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, z: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Retrieve k nearest neighbors.

        Args:
            z: Query embedding — shape (dim,) or (1, dim).
            k: Number of neighbors to retrieve.

        Returns:
            List of dicts [{'label': int, 'dist': float}, ...], ordered
            nearest-first.  Returns [] if store is empty.
        """
        n_stored = self.index.ntotal
        if n_stored == 0:
            return []

        k_actual = min(k, n_stored)
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(z_np, k_actual)

        results = []
        for j, idx in enumerate(indices[0]):
            if idx == -1:          # FAISS returns -1 when fewer than k exist
                continue
            results.append({
                "label": self.labels[idx],
                "dist":  float(distances[0][j]),
            })
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Clear the store (useful between datasets / ablation runs)."""
        self.index.reset()
        self.labels.clear()

    def save(self, path: str) -> None:
        """Persist index to disk (labels saved as .npy alongside)."""
        faiss.write_index(self.index, path)
        np.save(path + ".labels.npy", np.array(self.labels, dtype=np.int32))

    def load(self, path: str) -> None:
        """Restore a previously saved index."""
        self.index = faiss.read_index(path)
        self.labels = np.load(path + ".labels.npy").tolist()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Accept torch.Tensor or np.ndarray, always return np.ndarray."""
    if hasattr(x, "detach"):          # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)