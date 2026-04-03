"""
backbone.py — Unified entry point for LSTM + GNN backbone.
Owned by Person 1.

Exposes the Interface A contract:
    get_embedding(x_window, graph, node_id, t) -> (z, x_hat)

Also provides Backbone as an nn.Module for training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .lstm_encoder import LSTMEncoder
from .gnn_reasoner import GNNReasoner


# ---------------------------------------------------------------------------
# Backbone module
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """
    Combines LSTMEncoder (per-node) and GNNReasoner (cross-node).

    Dimensions (must be agreed with Person 2 on Day 1):
        LSTM hidden_size : 64
        GNN out_dim      : 64   ← FAISS index dim

    Args:
        d_in        : feature dim per node (1 for univariate signals)
        hidden_size : LSTM hidden dim (default 64)
        gnn_out_dim : GNN output / embedding dim d_z (default 64)
        num_nodes   : number of sensor nodes N
        window_size : sliding window length W (default 30)
        lstm_layers : depth of LSTM (default 2)
        gat_heads   : GAT attention heads (default 4)
        dropout     : dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_in        : int = 1,
        hidden_size : int = 64,
        gnn_out_dim : int = 64,
        num_nodes   : int = 51,
        window_size : int = 30,
        lstm_layers : int = 2,
        gat_heads   : int = 4,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.hidden_size = hidden_size
        self.gnn_out_dim = gnn_out_dim

        self.lstm = LSTMEncoder(
            d_in        = d_in,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            dropout     = dropout,
        )

        self.gnn = GNNReasoner(
            in_dim  = hidden_size,
            out_dim = gnn_out_dim,
            heads   = gat_heads,
            dropout = dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, x_windows: torch.Tensor,
                graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full batch of node windows.
        Supports single graph [N, W, d_in] or batched graphs [B, N, W, d_in].
        """
        is_batched = (x_windows.dim() == 4)
        if is_batched:
            B, N_dim, W, d_in = x_windows.shape
            x_windows = x_windows.view(B * N_dim, W, d_in)
        else:
            N_dim = x_windows.shape[0]

        # LSTM: process all nodes
        h_all, x_hat_all = self.lstm(x_windows)    # [B*N, hidden], [B*N, d_in]

        # Build edge_index for the GAT natively on this device
        device = x_windows.device
        base_edge_index = graph.edge_index.to(device)
        
        if is_batched:
            E = base_edge_index.shape[1]
            edge_index = base_edge_index.repeat(1, B)
            offsets = torch.arange(B, device=device).view(B, 1).repeat(1, E).view(-1) * N_dim
            edge_index = edge_index + offsets
            
            edge_attr = None
            if graph.edge_attr is not None:
                edge_attr = graph.edge_attr.to(device).repeat(B, 1)
        else:
            edge_index = base_edge_index
            edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

        # GNN: reason over node embeddings
        z_all = self.gnn(h_all, edge_index, edge_attr)   # [B*N, gnn_out_dim]

        if is_batched:
            z_all = z_all.view(B, N_dim, -1)
            x_hat_all = x_hat_all.view(B, N_dim, -1)

        return z_all, x_hat_all

    # ------------------------------------------------------------------
    def reconstruction_loss(self, x_windows: torch.Tensor,
                             x_hat_all: torch.Tensor) -> torch.Tensor:
        """MSE loss between last timestep and reconstruction for all nodes."""
        return self.lstm.reconstruction_loss(x_windows, x_hat_all)


# ---------------------------------------------------------------------------
# Interface A — fixed contract for Person 2
# ---------------------------------------------------------------------------

_global_backbone: Backbone | None = None


def load_backbone(checkpoint_path: str, device: str = "cpu",
                  **kwargs) -> Backbone:
    """Load a trained backbone from a checkpoint file."""
    global _global_backbone
    model = Backbone(**kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _global_backbone = model
    return model


@torch.no_grad()
def get_embedding(
    x_window  : torch.Tensor,
    graph     : Data,
    node_id   : int,
    t         : int,
    backbone  : Backbone | None = None,
    device    : str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Interface A — called by Person 2 (RAG scorer).

    Args:
        x_window : Tensor [W, d_in]   sliding window for node v at time t
        graph    : torch_geometric Data object (static, shared)
        node_id  : int                node index v
        t        : int                timestep (informational)
        backbone : Backbone | None    if None, uses global loaded backbone
        device   : str

    Returns:
        z     : Tensor [d_z]   joint LSTM+GNN embedding for node v at t
        x_hat : Tensor [d_in]  LSTM reconstructed signal for node v at t
    """
    model = backbone or _global_backbone
    if model is None:
        raise RuntimeError(
            "No backbone available. Call load_backbone() first or pass backbone=.")

    model = model.to(device).eval()

    # Build a single-node batch: [1, W, d_in]
    x = x_window.unsqueeze(0).to(device)   # [1, W, d_in]

    # For single-node queries we skip the GNN (no graph context) and
    # return only the LSTM embedding — Person 2 can still score it.
    # Full GNN pass requires all-node windows (used in training loop).
    h, x_hat = model.lstm(x)               # [1, hidden], [1, d_in]

    z     = h.squeeze(0)                   # [hidden_size] = [d_z]
    x_hat = x_hat.squeeze(0)               # [d_in]

    return z.cpu(), x_hat.cpu()


def get_embedding_full_graph(
    x_windows : torch.Tensor,
    graph     : Data,
    node_id   : int,
    t         : int,
    backbone  : Backbone | None = None,
    device    : str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full backbone pass using all nodes — returns embedding for node_id.
    Use this during training / evaluation for richer GNN context.

    Args:
        x_windows : Tensor [N, W, d_in]  windows for ALL nodes at time t
        graph     : torch_geometric Data
        node_id   : int
        t         : int

    Returns:
        z     : Tensor [d_z]   GNN-enhanced embedding for node_id
        x_hat : Tensor [d_in]  LSTM reconstruction for node_id
    """
    model = backbone or _global_backbone
    if model is None:
        raise RuntimeError("No backbone available.")

    model = model.to(device).eval()
    x_windows = x_windows.to(device)

    with torch.no_grad():
        z_all, x_hat_all = model(x_windows, graph)

    return z_all[node_id].cpu(), x_hat_all[node_id].cpu()