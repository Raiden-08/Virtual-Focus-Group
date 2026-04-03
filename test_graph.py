import torch
from torch_geometric.data import Data, Batch

# Mock graph: 3 nodes, 2 edges
edge_index = torch.tensor([[0, 1], [1, 2]])
graph = Data(edge_index=edge_index)

# Batch size = 2
B = 2
N_dim = 3

# Desired native PyG batch behavior:
pyg_batch = Batch.from_data_list([graph]*B)
print("PyG batched edges:\n", pyg_batch.edge_index)

# Our custom GPU batched edge construct:
device = torch.device('cpu')
E = edge_index.shape[1]
batched_edge_index = edge_index.repeat(1, B) # [2, B*E]
offsets = torch.arange(B, device=device).view(B, 1).repeat(1, E).view(-1) * N_dim
batched_edge_index = batched_edge_index + offsets

print("Our batched edges:\n", batched_edge_index)

assert torch.all(pyg_batch.edge_index == batched_edge_index), "Must be exactly equal"
print("Success! The graphs match.")
