"""
rag_scorer.py — Unified entry point for Person 2's RAG hardness scoring.

Exposes score_hardness() — the Interface B contract used by Person 3's
curriculum scheduler.

Composite score:
    H = alpha_1 * H_temp + alpha_2 * H_struct + alpha_3 * H_RAG

All weights must sum to 1.0.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List

from rag.vector_store import VectorStore
from rag.hardness import compute_h_temp, compute_h_struct, compute_h_rag


# ======================================================================
# Public Interface — Interface B (Person 3 calls this)
# ======================================================================

def score_hardness(
    z: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    node_id: int,
    graph,                              # torch_geometric Data
    t: int,
    # ---- extras needed internally (not in the minimal interface) ----
    window_errors: List[float],         # running list of reconstruction errors
    vector_store: VectorStore,          # shared FAISS store
    ground_truth_label: int,            # 0 or 1 — for adding to store AFTER scoring
    alphas: Tuple[float, float, float] = (0.33, 0.33, 0.34),
    k_neighbors: int = 10,
    gamma: float = 0.5,
    anomaly_source_id: Optional[int] = None,
) -> float:
    """
    Compute composite hardness H in [0, 1] for one (node, timestep) sample.

    IMPORTANT ordering:
        1. Compute all three components USING the current store state.
        2. ADD z to the store AFTER scoring (so the current sample doesn't
           influence its own score).
        3. APPEND current reconstruction error to window_errors so future
           samples can normalize against it.

    Args:
        z:                   Embedding from backbone.get_embedding()  [d_z]
        x:                   Raw signal window                        [d]
        x_hat:               LSTM reconstruction                     [d]
        node_id:             Integer node index.
        graph:               torch_geometric Data object.
        t:                   Current timestep (integer).
        window_errors:       Mutable list; this function appends to it.
        vector_store:        Shared VectorStore instance.
        ground_truth_label:  0 = normal, 1 = anomaly.
        alphas:              (alpha_1, alpha_2, alpha_3) weights, must sum to ~1.
        k_neighbors:         Neighbors for H_RAG retrieval.
        gamma:               Weight inside H_struct.
        anomaly_source_id:   Optional source node for structural depth term.

    Returns:
        H — float in [0, 1].
    """
    # Validate alpha weights
    alpha_sum = sum(alphas)
    if abs(alpha_sum - 1.0) > 1e-3:
        raise ValueError(f"alphas must sum to 1.0, got {alpha_sum:.4f}")

    alpha_1, alpha_2, alpha_3 = alphas

    # ------------------------------------------------------------------
    # Step 1: Compute current reconstruction error & append to history
    # ------------------------------------------------------------------
    # Safe norm: flatten both tensors first to handle shape mismatches
    # (x_hat from Person 1's backbone is [1], x from x_window[-1] is also [1],
    #  but defensive flattening prevents silent NaN if shapes ever diverge)
    x_flat     = x.reshape(-1).float()
    x_hat_flat = x_hat.reshape(-1).float()
    if x_flat.shape == x_hat_flat.shape:
        e = torch.norm(x_flat - x_hat_flat, p=2).item()
    else:
        e = abs(x_flat[0].item() - x_hat_flat[0].item())
    window_errors.append(e)           # grow the running list

    # ------------------------------------------------------------------
    # Step 2: H_temp
    # ------------------------------------------------------------------
    h_temp = compute_h_temp(x, x_hat, window_errors)

    # ------------------------------------------------------------------
    # Step 3: H_struct
    # ------------------------------------------------------------------
    h_struct = compute_h_struct(
        node_id=node_id,
        graph=graph,
        anomaly_source_id=anomaly_source_id,
        gamma=gamma,
    )

    # ------------------------------------------------------------------
    # Step 4: H_RAG  (uses store BEFORE adding current sample)
    # ------------------------------------------------------------------
    h_rag = compute_h_rag(z, vector_store, k=k_neighbors)

    # ------------------------------------------------------------------
    # Step 5: Composite score
    # ------------------------------------------------------------------
    H = alpha_1 * h_temp + alpha_2 * h_struct + alpha_3 * h_rag

    # ------------------------------------------------------------------
    # Step 6: Add this embedding to store AFTER scoring
    # ------------------------------------------------------------------
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    vector_store.add(z_np, label=ground_truth_label)

    return float(np.clip(H, 0.0, 1.0))


# ======================================================================
# Convenience: batch scoring (optional — Person 3 may prefer per-sample)
# ======================================================================

def score_dataset(
    dataset,                        # list of (node_id, t, label) tuples
    get_embedding_fn,               # backbone.get_embedding callable
    x_windows,                      # dict {(node_id, t): Tensor [W, d]}
    graphs,                         # dict {t: torch_geometric Data} or single Data
    vector_store: VectorStore,
    alphas: Tuple[float, float, float] = (0.33, 0.33, 0.34),
    k_neighbors: int = 10,
    gamma: float = 0.5,
) -> dict:
    """
    Score all samples in the dataset in order.

    Returns:
        hardness_scores: dict {(node_id, t): float}
    """
    hardness_scores = {}
    window_errors: List[float] = []

    for node_id, t, label in dataset:
        x_window = x_windows[(node_id, t)]             # [W, d]
        graph = graphs[t] if isinstance(graphs, dict) else graphs

        z, x_hat = get_embedding_fn(x_window, graph, node_id, t)
        x = x_window[-1]                                # last timestep as "current" x

        H = score_hardness(
            z=z,
            x=x,
            x_hat=x_hat,
            node_id=node_id,
            graph=graph,
            t=t,
            window_errors=window_errors,
            vector_store=vector_store,
            ground_truth_label=label,
            alphas=alphas,
            k_neighbors=k_neighbors,
            gamma=gamma,
        )
        hardness_scores[(node_id, t)] = H

    return hardness_scores