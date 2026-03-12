"""
swat.py — SWAT (Secure Water Treatment) dataset loader.
Owned by Person 1.

Download instructions:
  Request access from iTrust, Singapore University of Technology & Design:
  https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Expected files after download:
  data/raw/SWaT_Dataset_Normal_v1.csv
  data/raw/SWaT_Dataset_Attack_v0.csv
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
              train_ratio: float = 0.6,
              val_ratio: float = 0.1):
    """
    Load SWAT CSVs and return train / val / test BaseTimeSeriesDataset splits.

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
    normal_df = pd.read_csv(normal_path, sep=";", low_memory=False)
    attack_df = pd.read_csv(attack_path, sep=";", low_memory=False)

    # strip whitespace from column names
    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()

    # sensor columns = everything except timestamp + label
    drop_cols = [c for c in [_TIMESTAMP_COL, _LABEL_COL]
                 if c in normal_df.columns]
    feature_cols = [c for c in normal_df.columns if c not in drop_cols]

    # normal data: all label = 0
    normal_signals = normal_df[feature_cols].values.astype(np.float32)
    normal_labels  = np.zeros_like(normal_signals, dtype=np.int64)

    # attack data: label column "Attack" / "Normal"
    attack_signals = attack_df[feature_cols].values.astype(np.float32)
    raw_labels     = attack_df[_LABEL_COL].str.strip().values
    # broadcast the row label to all nodes
    row_labels     = (raw_labels != "Normal").astype(np.int64)
    attack_labels  = np.broadcast_to(
        row_labels[:, None], attack_signals.shape).copy()

    # --- split normal into train / val ---
    T_n = len(normal_signals)
    t_train = int(T_n * train_ratio)
    t_val   = int(T_n * (train_ratio + val_ratio))

    train_sig = normal_signals[:t_train]
    train_lbl = normal_labels[:t_train]

    val_sig   = normal_signals[t_train:t_val]
    val_lbl   = normal_labels[t_train:t_val]

    # test = remaining normal + all attack
    test_sig  = np.concatenate([normal_signals[t_val:], attack_signals])
    test_lbl  = np.concatenate([normal_labels[t_val:],  attack_labels])

    # build graph from training signals (after normalisation inside dataset)
    # graph is shared across splits — built on train stats
    tmp_mean = train_sig.mean(0)
    tmp_std  = train_sig.std(0) + 1e-8
    train_norm = (train_sig - tmp_mean) / tmp_std
    graph = build_graph_from_correlation(train_norm, threshold=0.5)

    train_ds = BaseTimeSeriesDataset(train_sig, train_lbl,
                                     window=window, stride=stride,
                                     graph=graph)
    val_ds   = BaseTimeSeriesDataset(val_sig,   val_lbl,
                                     window=window, stride=stride,
                                     graph=graph)
    test_ds  = BaseTimeSeriesDataset(test_sig,  test_lbl,
                                     window=window, stride=stride,
                                     graph=graph)

    return train_ds, val_ds, test_ds, feature_cols