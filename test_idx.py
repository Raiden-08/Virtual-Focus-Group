import numpy as np
import torch
from data.base_dataset import BaseTimeSeriesDataset

# Mock data
T = 100
N = 5
W = 30
S = 1
signals = np.random.randn(T, N)
labels = np.zeros((T, N))
labels[50, 0] = 1

ds = BaseTimeSeriesDataset(signals, labels, window=W, stride=S)

t_values = np.array([30, 31, 50])
k_values = (t_values - ds.window) // ds.stride
print("k_values:", k_values)
base_idx = (k_values * N)[:, None]
node_offsets = np.arange(N)[None, :]
full_t_indices = (base_idx + node_offsets).flatten()
print("full_t_indices:", full_t_indices)

# Validate it retrieves the right nodes
for i, idx in enumerate(full_t_indices):
    t_expected = t_values[i // N]
    node_expected = i % N
    
    t_actual = ds._index_t[idx]
    node_actual = ds._index_v[idx]
    
    assert t_actual == t_expected, f"{t_actual} != {t_expected}"
    assert node_actual == node_expected, f"{node_actual} != {node_expected}"
print("All indices match perfectly!")
