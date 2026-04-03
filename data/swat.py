"""
swat.py — SWAT (Secure Water Treatment) dataset loader.
Owned by Person 1.

Download instructions:
  Request access from iTrust, Singapore University of Technology & Design:
  https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Expected files after download:
  data/raw/SWaT_Dataset_Normal_v1.csv
  data/raw/SWaT_Dataset_Attack_v0.csv

Data split (standard benchmark):
  Train: ALL normal data       — model learns normal patterns
  Val:   first val_ratio of attack data — has anomalies for model selection
  Test:  remaining attack data — final evaluation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from .base_dataset import BaseTimeSeriesDataset, build_graph_from_correlation

# Columns that are sensor readings (not timestamps / labels)
_LABEL_COL = "Normal/Attack"
_TIMESTAMP_COL = "Timestamp"


def load_swat(data_dir: str = "data/raw",
              window: int = 30,
              stride: int = 1,
              val_ratio: float = 0.15):
    """
    Load SWAT CSVs and return train / val / test BaseTimeSeriesDataset splits.

    Standard benchmark split:
      - Train: ALL normal data (learn normal patterns)
      - Val:   first val_ratio of attack data (anomalies for model selection)
      - Test:  remaining attack data (final evaluation)

    Returns:
        train_ds, val_ds, test_ds : BaseTimeSeriesDataset
        feature_names             : list[str]
    """
    data_dir = Path(data_dir)
    normal_path = data_dir / "SWaT_Dataset_Normal_v1.csv"
    attack_path = data_dir / "SWaT_Dataset_Attack_v0.csv"

    if not normal_path.exists() or not attack_path.exists():
        raise FileNotFoundError(
            f"SWAT CSV files not found in {data_dir}.\n"
            "Request access at https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/"
        )

    # --- load ---
    normal_df = pd.read_csv(normal_path, sep=",", header=1, low_memory=False)
    attack_df = pd.read_csv(attack_path, sep=",", header=1, low_memory=False)

    # strip whitespace from column names
    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()

    # sensor columns = everything except timestamp + label
    drop_cols = [c for c in [_TIMESTAMP_COL, _LABEL_COL]
                 if c in normal_df.columns]
    feature_cols = [c for c in normal_df.columns if c not in drop_cols]

    # ── normal data: all label = 0 ──────────────────────────────────────
    normal_signals = normal_df[feature_cols].values.astype(np.float32)
    normal_labels  = np.zeros_like(normal_signals, dtype=np.int64)

    # ── attack data: label column "Attack" / "Normal" ───────────────────
    attack_signals = attack_df[feature_cols].values.astype(np.float32)
    raw_labels     = attack_df[_LABEL_COL].str.strip().values
    row_labels     = (raw_labels != "Normal").astype(np.int64)
    attack_labels  = np.broadcast_to(
        row_labels[:, None], attack_signals.shape).copy()

    # ── Split: Train/Val/Test completely from contaminated Attack DB ────
    # Discard normal DB entirely. Curriculum requires noisy training sets!
    T_a = len(attack_signals)
    
    # 60% Train, 15% Val, 25% Test
    t_train_end = int(T_a * 0.60)
    t_val_end   = int(T_a * (0.60 + val_ratio))
    
    train_sig = attack_signals[:t_train_end]
    train_lbl = attack_labels[:t_train_end]
    
    val_sig   = attack_signals[t_train_end:t_val_end]
    val_lbl   = attack_labels[t_train_end:t_val_end]
    
    test_sig  = attack_signals[t_val_end:]
    test_lbl  = attack_labels[t_val_end:]

    # ── Compute normalization stats strictly from Train split ───────────
    train_mean = train_sig.mean(axis=0)         # [N]
    train_std  = train_sig.std(axis=0) + 1e-8   # [N]

    # ── Build graph from normalized training signals ────────────────────
    train_norm = (train_sig - train_mean) / train_std
    graph = build_graph_from_correlation(train_norm, threshold=0.5)

    print(f"[SWaT] Train: {len(train_sig):,} timesteps "
          f"({train_lbl.max(axis=1).sum():,} anomalous rows / {train_lbl.max(axis=1).mean()*100:.1f}%)")
    print(f"[SWaT] Val:   {len(val_sig):,} timesteps "
          f"({val_lbl.max(axis=1).sum():,} anomalous rows / {val_lbl.max(axis=1).mean()*100:.1f}%)")
    print(f"[SWaT] Test:  {len(test_sig):,} timesteps "
          f"({test_lbl.max(axis=1).sum():,} anomalous rows / {test_lbl.max(axis=1).mean()*100:.1f}%)")

    # ── Create datasets (val/test use training stats for normalization) ──
    train_ds = BaseTimeSeriesDataset(train_sig, train_lbl,
                                     window=window, stride=stride,
                                     graph=graph)
    val_ds   = BaseTimeSeriesDataset(val_sig,   val_lbl,
                                     window=window, stride=stride,
                                     graph=graph,
                                     norm_mean=train_mean,
                                     norm_std=train_std)
    test_ds  = BaseTimeSeriesDataset(test_sig,  test_lbl,
                                     window=window, stride=stride,
                                     graph=graph,
                                     norm_mean=train_mean,
                                     norm_std=train_std)

    return train_ds, val_ds, test_ds, feature_cols