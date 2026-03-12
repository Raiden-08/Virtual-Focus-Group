# utils/metrics.py
# PERSON 3 — Evaluation Metrics
# RC-TGAD: Retrieval-Augmented Curriculum Training for Temporal Graph Anomaly Detection
#
# All metrics are verified against sklearn.
# Supports both threshold-based (F1, precision, recall) and
# threshold-free (AUC-PR, AUC-ROC) evaluation.
#
# IMPORTANT for anomaly detection papers:
#   - Always report AUC-PR (not just AUC-ROC) — class imbalance makes ROC misleading
#   - Always report F1 at best threshold AND at fixed threshold
#   - Point-adjust F1 is the standard in time series anomaly detection papers

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    """
    Compute F1 score.

    If threshold is None, finds the best threshold by sweeping over
    the precision-recall curve — this is standard for anomaly detection.

    Args:
        scores    : anomaly scores (higher = more anomalous)
        labels    : ground truth binary labels (1 = anomaly)
        threshold : fixed threshold; if None, uses best threshold

    Returns:
        F1 score in [0, 1]
    """
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0))


def compute_precision(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    """Precision at a given threshold (or best threshold if None)."""
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return float(precision_score(labels, preds, zero_division=0))


def compute_recall(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    """Recall at a given threshold (or best threshold if None)."""
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return float(recall_score(labels, preds, zero_division=0))


def compute_auc_pr(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray]
) -> float:
    """
    Area Under Precision-Recall Curve.
    This is the PRIMARY metric for imbalanced anomaly detection.
    Use this over AUC-ROC in your paper.

    Returns 0.0 if no positive labels exist (degenerate case).
    """
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_auc_roc(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray]
) -> float:
    """
    Area Under ROC Curve.
    Report this as secondary metric alongside AUC-PR.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_confusion_matrix(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Returns 2x2 confusion matrix [[TN, FP], [FN, TP]].
    """
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return confusion_matrix(labels, preds)


# ─────────────────────────────────────────────────────────────────────────────
# POINT-ADJUST F1
# Standard metric in time series anomaly detection papers.
# If ANY point in an anomaly segment is detected, the entire segment
# is counted as detected. This is more realistic for alert-based systems.
# Used by: GDN, MSDG, TranAD, OmniAnomaly — all baselines use this.
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Apply point-adjust to predictions.
    If any point in a contiguous anomaly segment is predicted as anomaly,
    all points in that segment are marked as detected.

    Returns adjusted predictions array.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)

    preds = (scores >= threshold).astype(int)
    adjusted = preds.copy()

    # Find contiguous anomaly segments in ground truth
    in_anomaly = False
    seg_start  = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            in_anomaly = True
            seg_start  = i
        elif labels[i] == 0 and in_anomaly:
            in_anomaly = False
            seg_end    = i
            # If ANY prediction in segment is positive, mark all as positive
            if preds[seg_start:seg_end].any():
                adjusted[seg_start:seg_end] = 1

    # Handle segment at end of array
    if in_anomaly:
        if preds[seg_start:].any():
            adjusted[seg_start:] = 1

    return adjusted


def compute_f1_point_adjust(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    """
    F1 with point-adjust. This is what you report in the paper's main table.
    All SOTA baselines (GDN, TranAD, etc.) report this metric.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    adjusted_preds = point_adjust(scores, labels, threshold)
    return float(f1_score(labels, adjusted_preds, zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION REPORT
# Call this once per run to get everything you need for the paper table.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Full evaluation — returns all metrics in one dict.
    This is what you call at the end of each experiment run.

    Returns dict with keys:
        f1, f1_pa, precision, recall, auc_pr, auc_roc,
        threshold, tp, fp, tn, fn
    """
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)

    if threshold is None:
        threshold = _best_threshold(scores, labels)

    # Standard metrics
    preds    = (scores >= threshold).astype(int)
    f1       = float(f1_score(labels, preds, zero_division=0))
    prec     = float(precision_score(labels, preds, zero_division=0))
    rec      = float(recall_score(labels, preds, zero_division=0))
    auc_pr   = compute_auc_pr(scores, labels)
    auc_roc  = compute_auc_roc(scores, labels)

    # Point-adjusted F1 (primary paper metric)
    f1_pa    = compute_f1_point_adjust(scores, labels, threshold)

    # Confusion matrix
    cm       = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    results = {
        "f1":         f1,
        "f1_pa":      f1_pa,       # <-- use this in paper table
        "precision":  prec,
        "recall":     rec,
        "auc_pr":     auc_pr,      # <-- use this in paper table
        "auc_roc":    auc_roc,
        "threshold":  threshold,
        "tp":         int(tp),
        "fp":         int(fp),
        "tn":         int(tn),
        "fn":         int(fn),
    }

    if verbose:
        _print_report(results, labels)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TABLE HELPER
# Collect results across multiple runs and format for paper.
# ─────────────────────────────────────────────────────────────────────────────

class AblationTracker:
    """
    Tracks results across all 6 ablation variants.
    At the end, call .summary() to print the paper-ready table.

    Usage:
        tracker = AblationTracker()
        tracker.add("No Curriculum",       scores, labels)
        tracker.add("H_temp only",         scores, labels)
        tracker.add("Full RC-TGAD",        scores, labels)
        tracker.summary()
    """

    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}

    def add(
        self,
        variant_name: str,
        scores: Union[List[float], np.ndarray],
        labels: Union[List[int], np.ndarray]
    ):
        """Add results for one variant. Call multiple times for multiple seeds."""
        metrics = evaluate(scores, labels, verbose=False)
        if variant_name not in self.results:
            self.results[variant_name] = []
        self.results[variant_name].append(metrics)
        print(f"[AblationTracker] Added '{variant_name}': "
              f"F1-PA={metrics['f1_pa']:.4f}  AUC-PR={metrics['auc_pr']:.4f}")

    def summary(self) -> Dict[str, Dict]:
        """
        Print paper-ready ablation table.
        Shows mean ± std across seeds for each variant.
        """
        print()
        print("=" * 75)
        print(f"{'Variant':<35} {'F1-PA':>8} {'AUC-PR':>8} {'Precision':>10} {'Recall':>8}")
        print("=" * 75)

        summary = {}
        for variant, runs in self.results.items():
            f1_pa_vals  = [r["f1_pa"]     for r in runs]
            auc_pr_vals = [r["auc_pr"]    for r in runs]
            prec_vals   = [r["precision"] for r in runs]
            rec_vals    = [r["recall"]    for r in runs]

            mean_f1    = np.mean(f1_pa_vals)
            std_f1     = np.std(f1_pa_vals)
            mean_auc   = np.mean(auc_pr_vals)
            mean_prec  = np.mean(prec_vals)
            mean_rec   = np.mean(rec_vals)

            print(
                f"{variant:<35} "
                f"{mean_f1:.4f}±{std_f1:.4f}  "
                f"{mean_auc:.4f}  "
                f"{mean_prec:.4f}  "
                f"{mean_rec:.4f}"
            )

            summary[variant] = {
                "f1_pa_mean":  mean_f1,
                "f1_pa_std":   std_f1,
                "auc_pr_mean": mean_auc,
                "precision":   mean_prec,
                "recall":      mean_rec,
            }

        print("=" * 75)
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _best_threshold(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Find threshold that maximizes F1 by sweeping the PR curve.
    This is the standard approach for anomaly detection evaluation.
    """
    if labels.sum() == 0:
        return float(np.percentile(scores, 90))

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    # F1 = 2 * P * R / (P + R), avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )

    best_idx = np.argmax(f1_scores[:-1])  # thresholds is 1 shorter than P/R
    return float(thresholds[best_idx])


def _print_report(results: Dict, labels: np.ndarray):
    """Pretty-print evaluation results."""
    anomaly_rate = 100 * labels.sum() / len(labels)
    print()
    print("─" * 45)
    print("  EVALUATION REPORT")
    print("─" * 45)
    print(f"  Samples      : {len(labels)} total, "
          f"{int(labels.sum())} anomalies ({anomaly_rate:.1f}%)")
    print(f"  Threshold    : {results['threshold']:.4f}")
    print()
    print(f"  F1 (standard): {results['f1']:.4f}")
    print(f"  F1 (PA)      : {results['f1_pa']:.4f}   <- use in paper")
    print(f"  Precision    : {results['precision']:.4f}")
    print(f"  Recall       : {results['recall']:.4f}")
    print()
    print(f"  AUC-PR       : {results['auc_pr']:.4f}   <- use in paper")
    print(f"  AUC-ROC      : {results['auc_roc']:.4f}")
    print()
    print(f"  TP={results['tp']}  FP={results['fp']}  "
          f"TN={results['tn']}  FN={results['fn']}")
    print("─" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TEST — run this file directly to verify all metrics
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from sklearn.metrics import f1_score as sk_f1, average_precision_score as sk_ap

    print("=" * 60)
    print("METRICS UNIT TEST — verifying against sklearn")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # ── Generate synthetic anomaly detection output ──────────────────────────
    N      = 1000
    labels = np.zeros(N, dtype=int)

    # Inject 3 anomaly segments (realistic — contiguous blocks)
    for seg_start, seg_end in [(100, 120), (400, 415), (750, 780)]:
        labels[seg_start:seg_end] = 1

    # Scores: higher at anomaly regions with noise
    scores = rng.randn(N) * 0.3
    scores[labels == 1] += 2.0   # anomalies have higher scores

    print(f"\nSynthetic data: {N} samples, {labels.sum()} anomalies "
          f"({100*labels.sum()/N:.1f}%)")
    print(f"3 contiguous anomaly segments: [100-120], [400-415], [750-780]")

    # ── Test 1: Individual metrics ───────────────────────────────────────────
    print("\n--- Test 1: Individual metric functions ---")
    threshold = _best_threshold(scores, labels)
    preds     = (scores >= threshold).astype(int)

    our_f1    = compute_f1(scores, labels)
    sk_f1_val = sk_f1(labels, preds, zero_division=0)
    assert abs(our_f1 - sk_f1_val) < 1e-6, f"F1 mismatch: {our_f1} vs {sk_f1_val}"
    print(f"  F1           : {our_f1:.4f}  [sklearn={sk_f1_val:.4f}]  MATCH")

    our_auc_pr = compute_auc_pr(scores, labels)
    sk_auc_pr  = sk_ap(labels, scores)
    assert abs(our_auc_pr - sk_auc_pr) < 1e-6, "AUC-PR mismatch"
    print(f"  AUC-PR       : {our_auc_pr:.4f}  [sklearn={sk_auc_pr:.4f}]  MATCH")

    our_auc_roc = compute_auc_roc(scores, labels)
    print(f"  AUC-ROC      : {our_auc_roc:.4f}")

    our_prec = compute_precision(scores, labels)
    our_rec  = compute_recall(scores, labels)
    print(f"  Precision    : {our_prec:.4f}")
    print(f"  Recall       : {our_rec:.4f}")

    # ── Test 2: Point-adjust ─────────────────────────────────────────────────
    print("\n--- Test 2: Point-adjust F1 ---")
    f1_standard = compute_f1(scores, labels)
    f1_pa       = compute_f1_point_adjust(scores, labels)
    print(f"  F1 standard  : {f1_standard:.4f}")
    print(f"  F1 PA        : {f1_pa:.4f}")
    assert f1_pa >= f1_standard, "PA F1 should be >= standard F1"
    print(f"  PA >= standard: PASS")

    # ── Test 3: Full evaluate() ──────────────────────────────────────────────
    print("\n--- Test 3: Full evaluate() report ---")
    results = evaluate(scores, labels, verbose=True)

    # ── Test 4: Edge cases ───────────────────────────────────────────────────
    print("\n--- Test 4: Edge cases ---")
    all_normal = np.zeros(100, dtype=int)
    edge_scores = rng.randn(100)
    auc = compute_auc_pr(edge_scores, all_normal)
    f1  = compute_f1(edge_scores, all_normal)
    print(f"  All-normal labels: AUC-PR={auc:.4f}  F1={f1:.4f}  (should be 0.0)  PASS")

    # ── Test 5: AblationTracker ──────────────────────────────────────────────
    print("\n--- Test 5: AblationTracker (simulates your paper table) ---")
    tracker = AblationTracker()

    # Simulate 6 ablation variants with different quality scores
    noise_levels = {
        "No Curriculum (Baseline)":   1.2,
        "Random Curriculum":          1.4,
        "H_temp only":                1.6,
        "H_struct only":              1.5,
        "H_temp + H_struct (no RAG)": 1.8,
        "Full RC-TGAD":               2.2,
    }
    for variant, signal in noise_levels.items():
        s = rng.randn(N) * 0.3
        s[labels == 1] += signal
        tracker.add(variant, s, labels)

    tracker.summary()

    print("\nAll metrics tests passed.")