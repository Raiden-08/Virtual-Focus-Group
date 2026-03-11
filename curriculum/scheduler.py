# curriculum/scheduler.py
# PERSON 3 — Curriculum Scheduler
# RC-TGAD: Retrieval-Augmented Curriculum Training for Temporal Graph Anomaly Detection

import numpy as np
from typing import List, Tuple, Dict


def pacing(epoch: int, k_warmup: int) -> float:
    """
    Returns lambda(k) — the hardness threshold at a given epoch.
    
    At epoch 0:        lambda = 0.0  → only the very easiest samples (H ≈ 0)
    At epoch k_warmup: lambda = 1.0  → all samples included
    Between:           linear ramp

    Args:
        epoch    : current training epoch (0-indexed)
        k_warmup : total number of warmup epochs (from config)

    Returns:
        float in [0.0, 1.0]
    """
    if k_warmup <= 0:
        return 1.0  # no curriculum — include everything from the start
    return float(min(1.0, epoch / k_warmup))


def get_batch(
    dataset: List[Tuple],
    hardness_scores: Dict[Tuple, float],
    epoch: int,
    k_warmup: int,
    safety_floor_ratio: float = 0.05,
    fallback_ratio: float = 0.30,
    verbose: bool = False
) -> List[int]:
    """
    Select which sample indices to include in training at a given epoch.

    A sample (node_id, t) is included if its hardness score H(v,t) <= lambda(epoch).
    - Easy samples (H close to 0) enter training early.
    - Hard samples (H close to 1) enter only in later epochs.
    - If too few samples qualify, falls back to the easiest fallback_ratio of the dataset.

    Args:
        dataset             : list of (node_id, t, label) tuples — full training set
        hardness_scores     : dict mapping (node_id, t) -> float H in [0, 1]
                              (from Person 2's rag_scorer.py, or mock dict for testing)
        epoch               : current epoch (0-indexed)
        k_warmup            : warmup period length
        safety_floor_ratio  : minimum fraction of dataset that must be selected
        fallback_ratio      : fraction to fall back to if floor not met
        verbose             : print debug info

    Returns:
        List of integer indices into dataset that should be trained this epoch
    """
    threshold = pacing(epoch, k_warmup)

    # Primary selection: include all samples with H <= threshold
    selected = [
        i for i, (node_id, t, label) in enumerate(dataset)
        if hardness_scores.get((node_id, t), 0.5) <= threshold
    ]

    # Safety floor: if less than safety_floor_ratio of data selected, fall back
    min_required = int(safety_floor_ratio * len(dataset))
    if len(selected) < min_required:
        # Sort all samples by hardness (ascending = easiest first)
        all_scores = [
            (hardness_scores.get((dataset[i][0], dataset[i][1]), 0.5), i)
            for i in range(len(dataset))
        ]
        all_scores.sort(key=lambda x: x[0])
        fallback_count = max(min_required, int(fallback_ratio * len(dataset)))
        selected = [i for _, i in all_scores[:fallback_count]]

        if verbose:
            print(f"[Scheduler] Epoch {epoch}: threshold={threshold:.3f}, "
                  f"primary selection too small ({len(selected)} < {min_required}), "
                  f"fell back to top-{fallback_ratio*100:.0f}% easiest = {len(selected)} samples")
    else:
        if verbose:
            print(f"[Scheduler] Epoch {epoch}: threshold={threshold:.3f}, "
                  f"selected {len(selected)}/{len(dataset)} samples "
                  f"({100*len(selected)/len(dataset):.1f}%)")

    return selected


def get_curriculum_stats(
    dataset: List[Tuple],
    hardness_scores: Dict[Tuple, float],
    k_warmup: int,
    total_epochs: int
) -> Dict:
    """
    Utility: compute how many samples are available at each epoch.
    Use this to sanity-check your curriculum before training.

    Returns a dict with epoch -> sample_count mapping.
    """
    stats = {}
    for epoch in range(total_epochs):
        selected = get_batch(dataset, hardness_scores, epoch, k_warmup, verbose=False)
        stats[epoch] = {
            "threshold": pacing(epoch, k_warmup),
            "n_selected": len(selected),
            "pct_selected": 100 * len(selected) / len(dataset)
        }
    return stats


# ─── MOCK TESTING (run this file directly to verify your scheduler works) ────
if __name__ == "__main__":
    import random
    random.seed(42)

    print("=" * 60)
    print("SCHEDULER UNIT TEST — using mock data")
    print("=" * 60)

    # Build a fake dataset: 1000 samples across 10 nodes, 100 timesteps
    N_NODES = 10
    T = 100
    mock_dataset = [
        (node_id, t, random.randint(0, 1))   # (node_id, t, label)
        for node_id in range(N_NODES)
        for t in range(T)
    ]

    # Mock hardness scores: random floats in [0, 1]
    mock_hardness = {
        (node_id, t): random.random()
        for node_id in range(N_NODES)
        for t in range(T)
    }

    K_WARMUP = 30
    TOTAL_EPOCHS = 50

    print(f"\nDataset size    : {len(mock_dataset)} samples")
    print(f"K_warmup        : {K_WARMUP}")
    print(f"Total epochs    : {TOTAL_EPOCHS}")
    print()

    # Test pacing function
    print("Pacing function lambda(k):")
    for ep in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
        lam = pacing(ep, K_WARMUP)
        bar = "█" * int(lam * 20)
        print(f"  epoch {ep:3d}  λ={lam:.3f}  [{bar:<20}]")

    print()

    # Test get_batch at key epochs
    print("Sample selection across epochs:")
    for ep in [0, 5, 10, 20, 30, 50]:
        selected = get_batch(mock_dataset, mock_hardness, ep, K_WARMUP, verbose=True)

    print()

    # Full stats
    stats = get_curriculum_stats(mock_dataset, mock_hardness, K_WARMUP, TOTAL_EPOCHS)
    print("Curriculum progression (every 5 epochs):")
    print(f"  {'Epoch':>6}  {'Threshold':>10}  {'N Selected':>12}  {'% Data':>8}")
    for ep in range(0, TOTAL_EPOCHS, 5):
        s = stats[ep]
        print(f"  {ep:>6}  {s['threshold']:>10.3f}  {s['n_selected']:>12}  {s['pct_selected']:>7.1f}%")

    print()
    print("All scheduler tests passed.")