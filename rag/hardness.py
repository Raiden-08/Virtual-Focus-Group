"""
hardness.py — Three-component hardness scoring for RC-TGAD.
"""

import numpy as np
import networkx as nx
import torch
from typing import Optional, List

def compute_h_temp(x, x_hat, window_errors, eps=1e-8):
    x_flat = x.reshape(-1).float()
    x_hat_flat = x_hat.reshape(-1).float()
    
    e = torch.norm(x_flat - x_hat_flat, p=2).item()
    
    if not window_errors:
        return 0.5
        
    e_min = np.percentile(window_errors, 5)
    e_max = np.percentile(window_errors, 95)
    
    h_temp = (e - e_min) / (e_max - e_min + eps)
    return float(np.clip(h_temp, 0.0, 1.0))

_STRUCT_CACHE = {}
_NX_GRAPH_CACHE = None
_GRAPH_DIAMETER_CACHE = None
_MAX_DEG_CACHE = None

def compute_h_struct(node_id, graph, anomaly_source_id, gamma):
    global _STRUCT_CACHE, _NX_GRAPH_CACHE, _GRAPH_DIAMETER_CACHE, _MAX_DEG_CACHE
    
    cache_key = (node_id, anomaly_source_id, gamma)
    if cache_key in _STRUCT_CACHE:
        return _STRUCT_CACHE[cache_key]

    if _NX_GRAPH_CACHE is None:
        _NX_GRAPH_CACHE = _pyg_to_nx(graph)
        degrees = dict(_NX_GRAPH_CACHE.degree())
        _MAX_DEG_CACHE = max(degrees.values()) if degrees else 1
        
        if nx.is_connected(_NX_GRAPH_CACHE):
            _GRAPH_DIAMETER_CACHE = nx.diameter(_NX_GRAPH_CACHE)
        else:
            _GRAPH_DIAMETER_CACHE = _pseudo_diameter(_NX_GRAPH_CACHE)

    G = _NX_GRAPH_CACHE
    deg = dict(G.degree()).get(node_id, 0)
    centrality_term = 1.0 - (deg / _MAX_DEG_CACHE) 

    if anomaly_source_id is not None and G.has_node(anomaly_source_id):
        try:
            dist = nx.shortest_path_length(G, node_id, anomaly_source_id)
            depth_term = dist / max(_GRAPH_DIAMETER_CACHE, 1)
        except nx.NetworkXNoPath:
            depth_term = 1.0
    else:
        depth_term = 0.5

    h_struct = gamma * depth_term + (1 - gamma) * centrality_term
    result = float(np.clip(h_struct, 0.0, 1.0))
    
    _STRUCT_CACHE[cache_key] = result
    return result


def compute_h_rag(z, vector_store, k: int = 10) -> float:
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    norm = np.linalg.norm(z_np)
    if norm > 1e-8:
        z_np = z_np / norm

    z_np = z_np.reshape(1, -1).astype(np.float32)

    neighbors = vector_store.query(z_np, k=k)
    if not neighbors:
        return 0.5
        
    try:
        distances = [float(d) for d, _ in neighbors]
    except Exception:
        return 0.5
        
    if not distances:
        return 0.5
        
    h_rag = float(np.mean(distances))
    return float(np.clip(h_rag, 0.0, 1.0))


def _pyg_to_nx(graph):
    import networkx as nx
    G = nx.Graph()
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        edges = graph.edge_index.cpu().numpy()
        G.add_edges_from([(edges[0, i], edges[1, i]) for i in range(edges.shape[1])])
    return G

def _pseudo_diameter(G):
    import networkx as nx
    if not G.nodes: return 1
    return max([nx.diameter(G.subgraph(c)) for c in nx.connected_components(G)])