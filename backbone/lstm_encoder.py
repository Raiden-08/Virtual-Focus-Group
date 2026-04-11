
"""
lstm_encoder.py — Per-node LSTM with dual head (Reconstruction + Prediction).
"""

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
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

        # Head 1: Reconstructs current timestep 't'
        self.recon_head = nn.Linear(hidden_size, d_in)
        
        # ⚡ NEW Head 2: Predicts future timestep 't+1'
        self.pred_head = nn.Linear(hidden_size, d_in)

    def forward(self, x_window: torch.Tensor):
        out, (h_n, _) = self.lstm(x_window)

        # Take the last layer's hidden state
        h = h_n[-1]                      # [B, hidden_size]
        
        x_hat = self.recon_head(h)       # [B, d_in]
        x_pred = self.pred_head(h)       # [B, d_in] (NEW)

        return h, x_hat, x_pred