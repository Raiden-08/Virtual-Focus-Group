import numpy as np
import pandas as pd
from pathlib import Path
from .base_dataset import BaseTimeSeriesDataset, build_graph_from_correlation

MIN_STD = 0.01      # drop channels with near-zero variance
MIN_LEN = 1000      # drop channels that are too short

def _load_telemanom(data_dir: Path, dataset: str = "SMAP"):
    anomaly_csv = data_dir / "labeled_anomalies.csv"
    label_df = pd.read_csv(anomaly_csv)
    label_df = label_df[label_df["spacecraft"] == dataset]
    channel_ids = label_df["chan_id"].tolist()

    train_list, test_list, valid_ids = [], [], []
    for cid in channel_ids:
        tr_path = data_dir / "train" / f"{cid}.npy"
        te_path = data_dir / "test"  / f"{cid}.npy"
        if not tr_path.exists() or not te_path.exists():
            continue
        tr = np.load(tr_path)[:, 0]
        te = np.load(te_path)[:, 0]
        # filter out degenerate channels
        if tr.std() < MIN_STD or len(tr) < MIN_LEN or len(te) < MIN_LEN:
            print(f"  [skip] {cid}: std={tr.std():.4f}, len_train={len(tr)}, len_test={len(te)}")
            continue
        train_list.append(tr)
        test_list.append(te)
        valid_ids.append(cid)

    print(f"  Kept {len(valid_ids)}/{len(channel_ids)} channels for {dataset}")

    # Pad/truncate to median length instead of min (less data loss)
    def align(arrays):
        target = int(np.median([len(a) for a in arrays]))
        out = []
        for a in arrays:
            if len(a) >= target:
                out.append(a[:target])
            else:
                # pad with last value
                pad = np.full(target - len(a), a[-1])
                out.append(np.concatenate([a, pad]))
        return np.stack(out, axis=1).astype(np.float32)

    train_signals = align(train_list)
    test_signals  = align(test_list)

    T_test, N = test_signals.shape
    test_labels = np.zeros((T_test, N), dtype=np.int64)
    for ni, cid in enumerate(valid_ids):
        row = label_df[label_df["chan_id"] == cid].iloc[0]
        import ast
        for (start, end) in ast.literal_eval(row["anomaly_sequences"]):
            end = min(end + 1, T_test)
            if start < T_test:
                test_labels[start:end, ni] = 1

    print(f"  Train: {train_signals.shape}, Test: {test_signals.shape}, "
          f"Anomaly rate: {test_labels.mean()*100:.1f}%")
    return train_signals, test_signals, test_labels, valid_ids


def _load_split(data_dir, dataset, window, stride, val_ratio):
    data_dir = Path(data_dir)
    train_sig, test_sig, test_lbl, channel_ids = _load_telemanom(data_dir, dataset)

    T_n   = len(train_sig)
    t_val = int(T_n * (1 - val_ratio))
    val_sig   = train_sig[t_val:]
    val_lbl   = np.zeros_like(val_sig, dtype=np.int64)
    train_sig = train_sig[:t_val]
    train_lbl = np.zeros_like(train_sig, dtype=np.int64)

    tmp_norm = (train_sig - train_sig.mean(0)) / (train_sig.std(0) + 1e-8)
    graph    = build_graph_from_correlation(tmp_norm, threshold=0.5)

    train_ds = BaseTimeSeriesDataset(train_sig, train_lbl, window=window, stride=stride, graph=graph)
    val_ds   = BaseTimeSeriesDataset(val_sig,   val_lbl,   window=window, stride=stride, graph=graph)
    test_ds  = BaseTimeSeriesDataset(test_sig,  test_lbl,  window=window, stride=stride, graph=graph)
    return train_ds, val_ds, test_ds, channel_ids


def load_smap(data_dir="data/raw/smap", window=30, stride=1, val_ratio=0.15):
    return _load_split(data_dir, "SMAP", window, stride, val_ratio)

def load_msl(data_dir="data/raw/smap", window=30, stride=1, val_ratio=0.15):
    return _load_split(data_dir, "MSL", window, stride, val_ratio)
