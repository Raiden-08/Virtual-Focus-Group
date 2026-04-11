"""
test_backbone.py — Unit tests for Person 1's modules.
Run with: python -m pytest tests/test_backbone.py -v

Verifies shapes, interface contract, and integration readiness for Person 2.
"""

import torch
import numpy as np
import pytest
from torch_geometric.data import Data


# ------------------------------------------------------------------ helpers
def make_dummy_graph(N: int = 51) -> Data:
    """Fully connected graph for testing."""
    src = [i for i in range(N) for j in range(N) if i != j]
    dst = [j for i in range(N) for j in range(N) if i != j]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.ones(len(src))
    degree     = torch.full((N,), N - 1, dtype=torch.long)
    g = Data(num_nodes=N, edge_index=edge_index, edge_attr=edge_attr)
    g.degree = degree
    return g


# ------------------------------------------------------------------ data
class TestBaseDataset:
    def test_init_and_len(self):
        from data.base_dataset import BaseTimeSeriesDataset
        T, N, W = 200, 10, 30
        signals = np.random.randn(T, N).astype(np.float32)
        labels  = np.zeros((T, N), dtype=np.int64)
        ds = BaseTimeSeriesDataset(signals, labels, window=W, stride=1)
        assert len(ds) > 0

    def test_item_shapes(self):
        from data.base_dataset import BaseTimeSeriesDataset
        T, N, W = 200, 10, 30
        signals = np.random.randn(T, N).astype(np.float32)
        labels  = np.zeros((T, N), dtype=np.int64)
        ds = BaseTimeSeriesDataset(signals, labels, window=W, stride=1)
        item = ds[0]
        assert item["x_window"].shape == (W, 1), item["x_window"].shape
        assert item["node_id"] in range(N)
        assert item["label"] in (0, 1)

    def test_graph_built(self):
        from data.base_dataset import BaseTimeSeriesDataset
        T, N = 100, 5
        signals = np.random.randn(T, N).astype(np.float32)
        labels  = np.zeros((T, N), dtype=np.int64)
        ds = BaseTimeSeriesDataset(signals, labels, window=10)
        g = ds.graph
        assert g.edge_index is not None
        assert g.edge_index.shape[0] == 2

    def test_flat_list(self):
        from data.base_dataset import BaseTimeSeriesDataset
        T, N, W = 100, 5, 10
        signals = np.random.randn(T, N).astype(np.float32)
        labels  = np.zeros((T, N), dtype=np.int64)
        ds = BaseTimeSeriesDataset(signals, labels, window=W)
        flat = ds.as_flat_list()
        assert isinstance(flat, list)
        assert all(len(row) == 3 for row in flat)


# ------------------------------------------------------------------ lstm
class TestLSTMEncoder:
    def test_forward_shapes(self):
        from backbone.lstm_encoder import LSTMEncoder
        B, W, d_in, hidden = 8, 30, 1, 64
        model = LSTMEncoder(d_in=d_in, hidden_size=hidden)
        x = torch.randn(B, W, d_in)
        h, x_hat = model(x)
        assert h.shape     == (B, hidden), h.shape
        assert x_hat.shape == (B, d_in),   x_hat.shape

    def test_reconstruction_loss(self):
        from backbone.lstm_encoder import LSTMEncoder
        model = LSTMEncoder()
        x = torch.randn(4, 30, 1)
        _, x_hat = model(x)
        loss = model.reconstruction_loss(x, x_hat)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


# ------------------------------------------------------------------ gnn
class TestGNNReasoner:
    def test_forward_shapes(self):
        from backbone.gnn_reasoner import GNNReasoner
        N, in_dim, out_dim = 51, 64, 64
        g = make_dummy_graph(N)
        model = GNNReasoner(in_dim=in_dim, out_dim=out_dim)
        h = torch.randn(N, in_dim)
        z = model(h, g.edge_index, g.edge_attr)
        assert z.shape == (N, out_dim), z.shape

    def test_no_nan(self):
        from backbone.gnn_reasoner import GNNReasoner
        N = 10
        g = make_dummy_graph(N)
        model = GNNReasoner()
        h = torch.randn(N, 64)
        z = model(h, g.edge_index)
        assert not torch.isnan(z).any()


# ------------------------------------------------------------------ backbone
class TestBackbone:
    def _make_backbone(self, N=51):
        from backbone.backbone import Backbone
        return Backbone(num_nodes=N), N

    def test_forward_shapes(self):
        model, N = self._make_backbone()
        g = make_dummy_graph(N)
        x = torch.randn(N, 30, 1)
        z_all, x_hat_all = model(x, g)
        assert z_all.shape     == (N, 64), z_all.shape
        assert x_hat_all.shape == (N, 1),  x_hat_all.shape

    def test_get_embedding_interface(self):
        """Interface A contract: single (node_id, t) → (z [64], x_hat [1])"""
        from backbone.backbone import get_embedding, Backbone
        N, W = 10, 30
        model = Backbone(num_nodes=N)
        g = make_dummy_graph(N)
        x_window = torch.randn(W, 1)
        z, x_hat = get_embedding(x_window, g, node_id=0, t=50, backbone=model)
        assert z.shape     == (64,), f"Expected z [64], got {z.shape}"
        assert x_hat.shape == (1,),  f"Expected x_hat [1], got {x_hat.shape}"
        assert not torch.isnan(z).any()
        assert not torch.isnan(x_hat).any()
        print(f"\n[Interface A OK] z={z.shape}, x_hat={x_hat.shape}")

    def test_z_dtype_float32(self):
        """Person 2 FAISS expects float32."""
        from backbone.backbone import get_embedding, Backbone
        model = Backbone(num_nodes=5)
        g = make_dummy_graph(5)
        z, _ = get_embedding(torch.randn(30, 1), g, 0, 0, backbone=model)
        assert z.dtype == torch.float32, f"Got {z.dtype}, FAISS needs float32"


# ------------------------------------------------------------------ graph
class TestGraphConstruction:
    def test_threshold(self):
        from data.base_dataset import build_graph_from_correlation
        T, N = 500, 10
        signals = np.random.randn(T, N).astype(np.float32)
        g = build_graph_from_correlation(signals, threshold=0.5)
        if g.edge_index.numel() > 0:
            assert g.edge_index.shape[0] == 2

    def test_self_loops_excluded(self):
        from data.base_dataset import build_graph_from_correlation
        T, N = 500, 5
        signals = np.random.randn(T, N)
        g = build_graph_from_correlation(signals, threshold=0.0)
        src, dst = g.edge_index
        assert not (src == dst).any(), "Self-loops should be excluded"