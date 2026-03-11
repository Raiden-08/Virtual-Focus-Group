# experiments/run_rctgad.py
# PERSON 3 — RC-TGAD Full Model Experiment Runner
# Curriculum ON — uses real hardness scores from Person 2's RAG scorer
#
# MODES:
#   Mock mode  (NOW)   : python experiments/run_rctgad.py --mock
#   Real mode  (Merge) : python experiments/run_rctgad.py --config configs/default.yaml
#
# Compare against baseline:
#   python experiments/run_rctgad.py --mock --compare results/baseline/swat/aggregate.json

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from curriculum.trainer    import Trainer, MockBackbone, MockRAGScorer, MockTemporalGraphDataset
from utils.metrics         import evaluate, AblationTracker

# Reuse helpers from run_baseline (same interface contracts)
from experiments.run_baseline import (
    load_dataset,
    load_backbone,
    evaluate_on_test
)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Full Model Runner")
    parser.add_argument("--config",   type=str,  default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model")
    parser.add_argument("--seeds",    type=int,  nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str,  default=None)
    parser.add_argument("--compare",  type=str,  default=None,
                        help="Path to baseline aggregate.json to print comparison table")
    parser.add_argument("--override", type=str,  nargs="*", default=[])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# RAG SCORER LOADER
# This is the key difference from run_baseline.py
# ─────────────────────────────────────────────────────────────────────────────

def load_rag_scorer_real(cfg):
    """
    Load Person 2's real RAG scorer post-merge.
    Until then, MockRAGScorer is used.
    """
    try:
        # ── REAL MODE: uncomment when Person 2 delivers ───────────────────
        # from rag.rag_scorer import RAGScorer
        # return RAGScorer(cfg)
        raise ImportError("Not merged yet")
    except ImportError:
        print("[run_rctgad] Person 2 module not found — using MockRAGScorer")
        return MockRAGScorer()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SEED RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(cfg, seed, mock, results_dir):
    """Run RC-TGAD (curriculum ON) for one seed. Returns test metrics dict."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*55}")
    print(f"  RC-TGAD RUN  |  seed={seed}  |  dataset={cfg['data']['dataset']}")
    print(f"  alphas: H_temp={cfg['rag']['alpha_1']}  "
          f"H_struct={cfg['rag']['alpha_2']}  "
          f"H_RAG={cfg['rag']['alpha_3']}")
    print(f"  k_warmup={cfg['curriculum']['k_warmup']}")
    print(f"{'='*55}")

    cfg["training"]["seed"]     = seed
    cfg["logging"]["run_name"]  = f"rctgad_seed{seed}"

    # Load components
    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone                        = load_backbone(cfg, mock)
    rag_scorer = MockRAGScorer() if mock else load_rag_scorer_real(cfg)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    # Build trainer — curriculum ON
    trainer = Trainer(
        backbone=backbone,
        rag_scorer=rag_scorer,
        dataset=train_data,
        config={
            "epochs":        cfg["training"]["epochs"],
            "k_warmup":      cfg["curriculum"]["k_warmup"],
            "batch_size":    cfg["training"]["batch_size"],
            "lr":            cfg["training"]["lr"],
            "weight_decay":  cfg["training"]["weight_decay"],
            "use_wandb":     cfg["logging"]["use_wandb"],
            "run_name":      cfg["logging"]["run_name"],
            "wandb_project": cfg["logging"]["wandb_project"],
        },
        use_curriculum=True,    # <-- RC-TGAD: curriculum ON
        device=device
    )

    # Train
    history = trainer.train(
        val_dataset=val_data,
        save_dir=os.path.join(results_dir, f"seed{seed}")
    )

    # Test evaluation
    print(f"\n[RC-TGAD] Evaluating on test set (seed={seed})...")
    test_results = evaluate_on_test(backbone, test_data, cfg, device)

    print(f"\n  Test Results (seed={seed}):")
    print(f"    F1-PA    : {test_results['f1_pa']:.4f}")
    print(f"    AUC-PR   : {test_results['auc_pr']:.4f}")
    print(f"    AUC-ROC  : {test_results['auc_roc']:.4f}")
    print(f"    Precision: {test_results['precision']:.4f}")
    print(f"    Recall   : {test_results['recall']:.4f}")

    # Save seed results
    seed_path = os.path.join(results_dir, f"seed{seed}", "test_results.json")
    os.makedirs(os.path.dirname(seed_path), exist_ok=True)
    with open(seed_path, "w") as f:
        json.dump({**test_results, "history": history}, f, indent=2)

    return test_results, history


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# Prints baseline vs RC-TGAD side by side
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(baseline_path, rctgad_results):
    """
    Prints a clean side-by-side comparison table.
    baseline_path: path to results/baseline/<dataset>/aggregate.json
    """
    if not os.path.exists(baseline_path):
        print(f"\n[Compare] Baseline file not found: {baseline_path}")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    rctgad_agg = {
        metric: {
            "mean": float(np.mean([r[metric] for r in rctgad_results])),
            "std":  float(np.std([r[metric]  for r in rctgad_results])),
        }
        for metric in ["f1_pa", "auc_pr", "precision", "recall"]
    }

    print(f"\n{'='*65}")
    print(f"  BASELINE  vs  RC-TGAD  COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Metric':<14} {'Baseline':>18} {'RC-TGAD':>18} {'Delta':>8}")
    print(f"  {'-'*60}")

    for metric in ["f1_pa", "auc_pr", "precision", "recall"]:
        b_mean = baseline[metric]["mean"]
        b_std  = baseline[metric]["std"]
        r_mean = rctgad_agg[metric]["mean"]
        r_std  = rctgad_agg[metric]["std"]
        delta  = r_mean - b_mean
        arrow  = "▲" if delta > 0 else "▼"
        print(f"  {metric:<14} "
              f"{b_mean:.4f}±{b_std:.4f}   "
              f"{r_mean:.4f}±{r_std:.4f}   "
              f"{arrow}{abs(delta):.4f}")

    print(f"{'='*65}")
    print(f"  F1-PA and AUC-PR are the primary paper metrics.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    overrides = args.override or []
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")
    if args.mock:
        overrides += ["training.epochs=20", "training.device=cpu"]

    cfg = load_config(args.config, overrides if overrides else None)

    dataset_name = cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "rctgad", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRC-TGAD — FULL MODEL EXPERIMENT")
    print(f"Dataset  : {dataset_name}")
    print(f"Seeds    : {args.seeds}")
    print(f"Epochs   : {cfg['training']['epochs']}")
    print(f"k_warmup : {cfg['curriculum']['k_warmup']}")
    print(f"Alphas   : {cfg['rag']['alpha_1']}, {cfg['rag']['alpha_2']}, {cfg['rag']['alpha_3']}")
    print(f"Mock mode: {args.mock}")

    # Run across all seeds
    all_results  = []
    all_histories = []
    for seed in args.seeds:
        result, history = run_single_seed(cfg, seed, args.mock, results_dir)
        all_results.append(result)
        all_histories.append(history)

    # Aggregate
    print(f"\n{'='*55}")
    print(f"  RC-TGAD FINAL RESULTS  ({dataset_name}, {len(args.seeds)} seeds)")
    print(f"{'='*55}")

    for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]:
        vals = [r[metric] for r in all_results]
        print(f"  {metric:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save aggregate
    agg = {
        metric: {
            "mean": float(np.mean([r[metric] for r in all_results])),
            "std":  float(np.std([r[metric]  for r in all_results])),
            "runs": [r[metric] for r in all_results]
        }
        for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]
    }
    agg_path = os.path.join(results_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Saved to {agg_path}")

    # Print comparison if baseline exists
    baseline_path = args.compare or os.path.join(
        "results", "baseline", dataset_name, "aggregate.json"
    )
    print_comparison(baseline_path, all_results)


if __name__ == "__main__":
    main()