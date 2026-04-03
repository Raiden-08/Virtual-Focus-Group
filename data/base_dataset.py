"""
base_dataset.py — Sliding window generator with graph construction.
Owned by Person 1. Used by all teammates.

PERFORMANCE NOTE (15M+ samples):
  All index building, label computation, and window extraction are
  fully vectorized with numpy. No Python for-loops over samples.
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
                 graph: Data | None = None,
                 norm_mean: np.ndarray | None = None,
                 norm_std: np.ndarray | None = None):
        super().__init__()
        assert signals.shape == labels.shape, (
            f"signals {signals.shape} and labels {labels.shape} must match")

        T, N = signals.shape
        self.window = window
        self.stride = stride
        self.N = N
        self.T = T

        # --- normalise per node to zero mean / unit variance ---
        # Use provided stats (from training set) to prevent data leakage
        if norm_mean is not None and norm_std is not None:
            self.mean = norm_mean.reshape(1, -1)
            self.std  = norm_std.reshape(1, -1)
        else:
            self.mean = signals.mean(axis=0, keepdims=True)        # [1, N]
            self.std  = signals.std(axis=0, keepdims=True) + 1e-8  # [1, N]
        self.signals = (signals - self.mean) / self.std        # [T, N]
        self.labels  = labels                                  # [T, N]

        # --- build or store graph ---
        self.graph = graph if graph is not None else \
            build_graph_from_correlation(self.signals, threshold=0.5)

        # --- VECTORIZED index building ---
        # t_values = [window, window+stride, window+2*stride, ...]
        t_values = np.arange(window, T, stride, dtype=np.int32)
        n_times  = len(t_values)
        node_ids = np.arange(N, dtype=np.int32)

        # For each t, we have N node entries: (node_0,t), (node_1,t), ...
        # index_t: [n_times*N] with repeated t values
        # index_v: [n_times*N] with tiled node ids
        self._index_t = np.repeat(t_values, N)         # [n_times*N]
        self._index_v = np.tile(node_ids, n_times)     # [n_times*N]
        self._len = len(self._index_t)

        # --- VECTORIZED label pre-computation ---
        # For each (v, t), label = max(labels[t-W:t, v])
        # Use a rolling-max approach via cumsum for speed
        print(f"[Dataset] Pre-computing {self._len:,} window labels (vectorized)...")
        self._precomputed_labels = self._vectorized_window_labels()

        # --- Build as_tuples eagerly (just integer arrays, very fast) ---
        # This replaces the slow as_flat_list() Python loop
        self.as_tuples = list(zip(
            self._index_v.tolist(),
            self._index_t.tolist(),
            self._precomputed_labels.tolist()
        ))

        # Keep legacy index for compatibility
        self.index = self.as_tuples  # [(node_id, t, label), ...]

        # --- Pre-compute all windows as a contiguous tensor ---
        print(f"[Dataset] Pre-building {self._len:,} window tensors...")
        self._precomputed_windows = self._build_all_windows()

        # --- Pre-compute forecast targets: signals[t, v] (one step AFTER window) ---
        # Window covers signals[t-W : t], so target = signals[t] which model hasn't seen
        self._precomputed_targets = torch.from_numpy(
            self.signals[self._index_t, self._index_v].astype(np.float32)
        ).unsqueeze(-1)  # [n_samples, 1]

        print(f"[Dataset] Ready. {self._len:,} samples.")

    def _vectorized_window_labels(self) -> np.ndarray:
        """Compute max-label-in-window for every sample using vectorized ops."""
        T, N = self.labels.shape
        W = self.window

        # Compute cumulative sum of labels per node: [T+1, N]
        cum = np.zeros((T + 1, N), dtype=np.int64)
        cum[1:] = np.cumsum(self.labels, axis=0)

        # For each t in _index_t, window sum = cum[t] - cum[t-W]
        # If sum > 0, there's at least one anomaly in the window
        t_arr = self._index_t   # [n_samples]
        v_arr = self._index_v   # [n_samples]

        window_sums = cum[t_arr, v_arr] - cum[t_arr - W, v_arr]
        return (window_sums > 0).astype(np.int64)

    def _build_all_windows(self) -> torch.Tensor:
        """Pre-extract all windows into one contiguous tensor [n_samples, W, 1]."""
        T, N = self.signals.shape
        W = self.window

        # Use stride_tricks to create a view of all windows: [T-W+1, W, N]
        # Then index by (t, v) to get [n_samples, W, 1]
        from numpy.lib.stride_tricks import as_strided
        strides = self.signals.strides
        shape = (T - W + 1, W, N)
        new_strides = (strides[0], strides[0], strides[1])
        all_windows_view = as_strided(self.signals, shape=shape, strides=new_strides)
        # all_windows_view[t-W, :, v] = signals[t-W:t, v] for t >= W

        # Index: for sample i, window starts at t-W, node is v
        start_indices = self._index_t - W  # [n_samples]
        node_indices = self._index_v       # [n_samples]

        # Extract windows: [n_samples, W]
        windows = all_windows_view[start_indices, :, node_indices]

        # Convert to torch tensor [n_samples, W, 1]
        return torch.from_numpy(windows.copy().astype(np.float32)).unsqueeze(-1)

    # ------------------------------------------------------------------
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {
            "node_id" : int(self._index_v[idx]),
            "t"       : int(self._index_t[idx]),
            "x_window": self._precomputed_windows[idx],   # [W, 1] — zero-copy
            "graph"   : self.graph,                        # shared static graph
            "label"   : int(self._precomputed_labels[idx]),
        }

    # ------------------------------------------------------------------
    def as_flat_list(self):
        """Return list of (node_id, t, label) for curriculum scheduler."""
        # Already computed during __init__, just return as_tuples
        return self.as_tuples


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

    # Handle NaN in correlation (constant columns produce nan)
    corr = np.corrcoef(signals.T)           # [N, N]
    corr = np.nan_to_num(corr, nan=0.0)

    # Vectorized edge construction
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)  # no self-loops
    mask = abs_corr >= threshold

    src, dst = np.where(mask)
    weights = abs_corr[mask]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float32)

    # Node degree (needed by H_struct)
    degree = torch.zeros(N, dtype=torch.long)
    if len(src) > 0:
        degree = torch.tensor(np.bincount(src, minlength=N), dtype=torch.long)

    graph = Data(
        num_nodes  = N,
        edge_index = edge_index,
        edge_attr  = edge_attr,
    )
    graph.degree = degree
    return graph