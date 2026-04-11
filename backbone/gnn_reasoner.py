"""
gnn_reasoner.py — GAT-based GNN for cross-node reasoning (GDN-style).
Owned by Person 1.

Takes per-node LSTM embeddings [N, hidden_size] and a graph, and
produces refined joint embeddings [N, gnn_out_dim].

We use Graph Attention Network (GAT) as the backbone, which mirrors
the attention mechanism from Graph Deviation Networks (GDN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GNNReasoner(nn.Module):
    """
    Two-layer GAT over the node embedding graph.

    Args:
        in_dim      : input dim per node = LSTM hidden_size (64)
        out_dim     : output embedding dim d_z (64) — must match FAISS index
        heads       : number of attention heads per GAT layer (default 4)
        dropout     : attention dropout (default 0.1)
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 64,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0, \
            f"out_dim ({out_dim}) must be divisible by heads ({heads})"

        mid_dim = out_dim  # intermediate channel width

        self.gat1 = GATConv(
            in_channels  = in_dim,
            out_channels = mid_dim // heads,
            heads        = heads,
            dropout      = dropout,
            concat       = True,
        )
        # gat1 output: [N, mid_dim]

        self.gat2 = GATConv(
            in_channels  = mid_dim,
            out_channels = out_dim,
            heads        = 1,
            dropout      = dropout,
            concat       = False,
        )
        # gat2 output: [N, out_dim]

        self.norm1 = nn.LayerNorm(mid_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            h          : Tensor [N, in_dim]  per-node LSTM embeddings
            edge_index : Tensor [2, E]       graph connectivity
            edge_attr  : Tensor [E]          edge weights (optional)

        Returns:
            z : Tensor [N, out_dim]  refined joint embeddings
        """
        # Layer 1
        x = self.gat1(h, edge_index)    # [N, mid_dim]
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        z = self.gat2(x, edge_index)    # [N, out_dim]
        z = self.norm2(z)
        z = F.elu(z)

        return z