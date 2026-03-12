"""
lstm_encoder.py — Per-node LSTM with reconstruction head.
Owned by Person 1.

Each node is encoded independently. The LSTM reads a sliding window
[W, d_in] and outputs:
  h   : Tensor [hidden_size]   — final hidden state (used by GNN)
  x_hat : Tensor [d_in]        — reconstruction of the last timestep
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Args:
        d_in        : input feature dim per node (1 for univariate)
        hidden_size : LSTM hidden dim — must match gnn_reasoner input (64)
        num_layers  : LSTM depth (default 2)
        dropout     : dropout between LSTM layers (default 0.1)
    """

    def __init__(self, d_in: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = d_in,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # Reconstruction head: hidden_size → d_in
        self.recon_head = nn.Linear(hidden_size, d_in)

    # ------------------------------------------------------------------
    def forward(self, x_window: torch.Tensor):
        """
        Args:
            x_window : Tensor [B, W, d_in]  (batch of windows)

        Returns:
            h     : Tensor [B, hidden_size]  final hidden state
            x_hat : Tensor [B, d_in]         reconstruction of last step
        """
        # out: [B, W, hidden_size], (h_n, c_n): [num_layers, B, hidden_size]
        out, (h_n, _) = self.lstm(x_window)

        # Take the last layer's hidden state
        h = h_n[-1]                      # [B, hidden_size]
        x_hat = self.recon_head(h)       # [B, d_in]

        return h, x_hat

    # ------------------------------------------------------------------
    def reconstruction_loss(self, x_window: torch.Tensor,
                             x_hat: torch.Tensor) -> torch.Tensor:
        """
        MSE loss between the last timestep of x_window and x_hat.

        Args:
            x_window : Tensor [B, W, d_in]
            x_hat    : Tensor [B, d_in]

        Returns:
            loss : scalar Tensor
        """
        x_last = x_window[:, -1, :]     # [B, d_in]
        return nn.functional.mse_loss(x_hat, x_last)