"""
hardness.py — Three-component hardness scoring for RC-TGAD.
FIXED: 
1. h_temp direction (Low Error = Easy).
2. L2 Normalization for RAG retrieval.
"""

import numpy as np
import networkx as nx
import torch
from typing import Optional, List

# ======================================================================
# Component 1 — H_temp (REVERSED FIX)
# ======================================================================

def compute_h_temp(recon_target, x_recon, pred_target, x_pred, window_errors, eps=1e-8):
    recon_e = torch.norm(recon_target.reshape(-1).float() - x_recon.reshape(-1).float(), p=2).item()
    pred_e  = torch.norm(pred_target.reshape(-1).float() - x_pred.reshape(-1).float(), p=2).item()
    e = recon_e + pred_e
    
    if not window_errors:
        return 0.5
        
    e_min = np.percentile(window_errors, 5)
    e_max = np.percentile(window_errors, 95)
    
    # 🛡️ FIX: Direct mapping. Low error -> Low Hardness (Easy).
    h_temp = (e - e_min) / (e_max - e_min + eps)
    return float(np.clip(h_temp, 0.0, 1.0))

# ======================================================================
# Component 2 — H_struct (Remains standard)
# ======================================================================

def compute_h_struct(node_id, graph, anomaly_source_id=None, gamma=0.5):
    G = _pyg_to_nx(graph)
    degrees = dict(G.degree())
    deg = degrees.get(node_id, 0)
    max_deg = max(degrees.values()) if degrees else 1
    centrality_term = 1.0 - (deg / max_deg) 

    if anomaly_source_id is not None and G.has_node(anomaly_source_id):
        try:
            dist = nx.shortest_path_length(G, node_id, anomaly_source_id)
            diam = nx.diameter(G) if nx.is_connected(G) else _pseudo_diameter(G)
            depth_term = dist / max(diam, 1)
        except nx.NetworkXNoPath:
            depth_term = 1.0
    else:
        depth_term = 0.5

    h_struct = gamma * depth_term + (1 - gamma) * centrality_term
    return float(np.clip(h_struct, 0.0, 1.0))

# ======================================================================
# Component 3 — H_RAG (NORMALIZATION FIX)
# ======================================================================

def compute_h_rag(z, vector_store, k: int = 10) -> float:
    # 🛡️ FIX: L2 Normalize embedding before FAISS query
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    norm = np.linalg.norm(z_np)
    if norm > 1e-8:
        z_np = z_np / norm

    neighbors = vector_store.query(z_np, k=k)
    if not neighbors:
        return 0.0

    labels = [n["label"] for n in neighbors]
    p_hat = sum(labels) / len(labels)

    if p_hat == 0.0 or p_hat == 1.0:
        return 0.0

    entropy = -p_hat * np.log2(p_hat) - (1 - p_hat) * np.log2(1 - p_hat)
    return float(np.clip(entropy, 0.0, 1.0))

# (Internal helpers _pyg_to_nx and _pseudo_diameter remain unchanged)
# ======================================================================
# Internal Graph Helpers
# ======================================================================

def _pyg_to_nx(graph):
    """Converts PyTorch Geometric Data to NetworkX Graph."""
    G = nx.Graph()
    
    # Add nodes safely
    num_nodes = getattr(graph, 'num_nodes', 51)
    G.add_nodes_from(range(num_nodes))
    
    # Add edges safely
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        edges = graph.edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
    return G

def _pseudo_diameter(G):
    """Returns the diameter of the largest connected component if graph is disconnected."""
    if len(G) == 0:
        return 1
        
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_subgraph = G.subgraph(components[0])
    
    if len(largest_subgraph) <= 1:
        return 1
        
    return nx.diameter(largest_subgraph)