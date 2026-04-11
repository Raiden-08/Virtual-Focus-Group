"""
backbone.py — Unified entry point for LSTM + GNN backbone.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data

from .lstm_encoder import LSTMEncoder
from .gnn_reasoner import GNNReasoner

class Backbone(nn.Module):
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

    def forward(self, x_windows: torch.Tensor,
                graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        is_batched = (x_windows.dim() == 4)
        if is_batched:
            B, N_dim, W, d_in = x_windows.shape
            x_flat = x_windows.view(B * N_dim, W, d_in)
        else:
            B = 1
            N_dim, W, d_in = x_windows.shape
            x_flat = x_windows

        # 1. TEMPORAL PHASE (LSTM) - Now returns x_pred_all
        h_all, x_hat_all, x_pred_all = self.lstm(x_flat)    

        # 2. SPATIAL PHASE (GNN) 
        device = x_windows.device
        base_edge_index = graph.edge_index.to(device)
        
        if is_batched:
            E = base_edge_index.shape[1]
            edge_index = base_edge_index.repeat(1, B)
            offsets = torch.arange(B, device=device).repeat_interleave(E) * N_dim
            edge_index = edge_index + offsets
            edge_attr = graph.edge_attr.to(device).repeat(B, 1) if graph.edge_attr is not None else None
        else:
            edge_index = base_edge_index
            edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

        z_all = self.gnn(h_all, edge_index, edge_attr)

        if is_batched:
            z_all = z_all.view(B, N_dim, -1)
            x_hat_all = x_hat_all.view(B, N_dim, -1)
            x_pred_all = x_pred_all.view(B, N_dim, -1) # NEW

        return z_all, x_hat_all, x_pred_all

# Interface A
_global_backbone: Backbone | None = None

def load_backbone(checkpoint_path: str, device: str = "cpu", **kwargs) -> Backbone:
    global _global_backbone
    model = Backbone(**kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _global_backbone = model
    return model

@torch.no_grad()
def get_embedding(
    x_windows : torch.Tensor,
    graph     : Data,
    node_id   : int,
    t         : int,
    backbone  : Backbone | None = None,
    device    : str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    model = backbone or _global_backbone
    if model is None:
        raise RuntimeError("No backbone available.")

    model = model.to(device).eval()
    x_windows = x_windows.to(device)

    # ⚡ Unpack the dual loss components
    z_all, x_hat_all, x_pred_all = model(x_windows, graph)

    z      = z_all[node_id].cpu()     
    x_hat  = x_hat_all[node_id].cpu() 
    x_pred = x_pred_all[node_id].cpu() # NEW

    return z, x_hat, x_pred