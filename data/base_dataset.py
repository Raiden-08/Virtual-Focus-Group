"""
base_dataset.py — Sliding window generator with graph construction.
Owned by Person 1. Used by all teammates.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BaseTimeSeriesDataset(Dataset):
    """
    Generates (node_id, t, window, graph, label) tuples from
    a multivariate time-series array.

    Args:
        signals   : np.ndarray  shape [T, N]  — T timesteps, N nodes
        labels    : np.ndarray  shape [T, N]  — 0/1 anomaly labels
        window    : int         sliding window length W (default 30)
        stride    : int         step between windows (default 1)
        graph     : torch_geometric.data.Data | None
                    If None, graph is built from Pearson correlation.
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 window: int = 30, stride: int = 1,
                 graph: Data | None = None):
        super().__init__()
        assert signals.shape == labels.shape, (
            f"signals {signals.shape} and labels {labels.shape} must match")

        T, N = signals.shape
        self.window = window
        self.stride = stride
        self.N = N

        # --- normalise per node to zero mean / unit variance ---
        self.mean = signals.mean(axis=0, keepdims=True)        # [1, N]
        self.std  = signals.std(axis=0, keepdims=True) + 1e-8  # [1, N]
        self.signals = (signals - self.mean) / self.std        # [T, N]
        self.labels  = labels                                  # [T, N]

        # --- build or store graph ---
        self.graph = graph if graph is not None else \
            build_graph_from_correlation(self.signals, threshold=0.5)

        # --- pre-compute index list: (node_id, t) ---
        self.index = []
        for t in range(window, T, stride):          # t = end of window
            for v in range(N):
                self.index.append((v, t))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        node_id, t = self.index[idx]

        # window: [W, d] where d=1 per-node signal (can be extended)
        x_window = torch.tensor(
            self.signals[t - self.window: t, node_id],
            dtype=torch.float32
        ).unsqueeze(-1)                             # [W, 1]

        # label for this (node, t): anomaly if ANY step in window is 1
        label = int(self.labels[t - self.window: t, node_id].max())

        return {
            "node_id" : node_id,
            "t"       : t,
            "x_window": x_window,          # [W, 1]
            "graph"   : self.graph,        # shared static graph
            "label"   : label,
        }

    # ------------------------------------------------------------------
    def as_flat_list(self):
        """Return list of (node_id, t, label) for curriculum scheduler."""
        return [(v, t, int(self.labels[t - self.window: t, v].max()))
                for v, t in self.index]


# -----------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------

def build_graph_from_correlation(signals: np.ndarray,
                                 threshold: float = 0.5) -> Data:
    """
    Build a static graph from Pearson correlation between node signals.
    An edge (i, j) is added when |corr(i, j)| >= threshold.

    Args:
        signals   : np.ndarray [T, N] normalised signals
        threshold : float  correlation threshold (default 0.5)

    Returns:
        torch_geometric.data.Data with edge_index and edge_attr
    """
    T, N = signals.shape
    corr = np.corrcoef(signals.T)           # [N, N]

    src, dst, weights = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j and abs(corr[i, j]) >= threshold:
                src.append(i)
                dst.append(j)
                weights.append(abs(corr[i, j]))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float32)

    # Node degree (needed by H_struct)
    degree = torch.zeros(N, dtype=torch.long)
    if len(src) > 0:
        for s in src:
            degree[s] += 1

    graph = Data(
        num_nodes  = N,
        edge_index = edge_index,
        edge_attr  = edge_attr,
    )
    graph.degree = degree
    return graph