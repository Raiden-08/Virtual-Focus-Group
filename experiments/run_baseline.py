# experiments/run_baseline.py
# PERSON 3 — Baseline Experiment Runner
# RC-TGAD: Runs the NO-CURRICULUM baseline (vanilla LSTM+GNN with no scheduling)
#
# MODES:
#   Mock mode  (NOW)   : python experiments/run_baseline.py --mock
#   Real mode  (Week2) : python experiments/run_baseline.py --config configs/default.yaml
#
# Runs 3 seeds, prints mean ± std, saves results to results/baseline/

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from curriculum.trainer    import Trainer, MockBackbone, MockRAGScorer, MockTemporalGraphDataset
from utils.metrics         import evaluate, AblationTracker


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Baseline Runner")
    parser.add_argument("--config",   type=str,  default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model (no real dataset needed)")
    parser.add_argument("--seeds",    type=int,  nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str,  default=None, help="Override data.dataset in config")
    parser.add_argument("--override", type=str,  nargs="*", default=[])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# Swap mock → real when Person 1 delivers data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg, seed, mock=False):
    """
    Returns (train_dataset, val_dataset, test_dataset).
    Mock mode: synthetic data — no files needed.
    Real mode: loads SWAT / SMAP / MSL from data/raw/
    """
    if mock:
        d_in       = cfg["model"]["d_in"]
        win        = cfg["model"]["window_size"]
        train_data = MockTemporalGraphDataset(n_nodes=10, T=500, window_size=win, d_in=d_in, seed=seed)
        val_data   = MockTemporalGraphDataset(n_nodes=10, T=150, window_size=win, d_in=d_in, seed=seed+100)
        test_data  = MockTemporalGraphDataset(n_nodes=10, T=200, window_size=win, d_in=d_in, seed=seed+200)
        return train_data, val_data, test_data

    # ── REAL MODE: swap these imports when Person 1 is ready ─────────────────
    dataset_name = cfg["data"]["dataset"]
    if dataset_name == "swat":
        from data.swat import SWATDataset
        train_data = SWATDataset(cfg, split="train")
        val_data   = SWATDataset(cfg, split="val")
        test_data  = SWATDataset(cfg, split="test")
    elif dataset_name == "smap":
        from data.smap import SMAPDataset
        train_data = SMAPDataset(cfg, split="train")
        val_data   = SMAPDataset(cfg, split="val")
        test_data  = SMAPDataset(cfg, split="test")
    elif dataset_name == "msl":
        from data.msl import MSLDataset
        train_data = MSLDataset(cfg, split="train")
        val_data   = MSLDataset(cfg, split="val")
        test_data  = MSLDataset(cfg, split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_data, val_data, test_data


def load_backbone(cfg, mock=False):
    """
    Returns backbone model.
    Mock mode: random linear layers.
    Real mode: Person 1's LSTM+GNN backbone.
    """
    if mock:
        return MockBackbone(
            d_in=cfg["model"]["d_in"],
            d_z=cfg["model"]["gnn_out_dim"]
        )
    # ── REAL MODE: swap when Person 1 is ready ────────────────────────────
    from backbone.backbone import Backbone
    return Backbone(cfg)


def load_rag_scorer(cfg, mock=False):
    """
    Returns RAG scorer.
    Baseline always uses mock scorer (hardness scores not used — curriculum is off).
    """
    return MockRAGScorer()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(backbone, test_dataset, cfg, device) -> dict:
    """
    Run inference on test set, compute all metrics.
    Returns results dict from utils/metrics.py evaluate().
    """
    import torch
    from torch.utils.data import DataLoader

    backbone.eval()
    all_scores = []
    all_labels = []

    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            x_window = batch["x_window"].to(device)
            labels   = batch["label"]

            for b in range(x_window.shape[0]):
                z, x_hat = backbone.get_embedding(x_window[b])
                score    = torch.norm(x_hat - x_window[b].mean(dim=0)).item()
                all_scores.append(score)
                all_labels.append(int(labels[b]))

    return evaluate(all_scores, all_labels, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SEED RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(cfg, seed, mock, results_dir):
    """Run baseline for one seed. Returns test metrics dict."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*55}")
    print(f"  BASELINE RUN  |  seed={seed}  |  dataset={cfg['data']['dataset']}")
    print(f"{'='*55}")

    # Override seed in config
    cfg["training"]["seed"] = seed
    cfg["logging"]["run_name"] = f"baseline_seed{seed}"

    # Load components
    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone                        = load_backbone(cfg, mock)
    rag_scorer                      = load_rag_scorer(cfg, mock)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    # Build trainer — curriculum OFF for baseline
    trainer = Trainer(
        backbone=backbone,
        rag_scorer=rag_scorer,
        dataset=train_data,
        config={
            "epochs":       cfg["training"]["epochs"],
            "k_warmup":     cfg["curriculum"]["k_warmup"],
            "batch_size":   cfg["training"]["batch_size"],
            "lr":           cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "use_wandb":    cfg["logging"]["use_wandb"],
            "run_name":     cfg["logging"]["run_name"],
            "wandb_project":cfg["logging"]["wandb_project"],
        },
        use_curriculum=False,   # <-- BASELINE: no curriculum
        device=device
    )

    # Train
    history = trainer.train(
        val_dataset=val_data,
        save_dir=os.path.join(results_dir, f"seed{seed}")
    )

    # Test evaluation
    print(f"\n[Baseline] Evaluating on test set (seed={seed})...")
    test_results = evaluate_on_test(backbone, test_data, cfg, device)

    # Print results
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

    return test_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    overrides = args.override or []
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")
    if args.mock:
        # Shorter run for mock testing
        overrides += ["training.epochs=20", "training.device=cpu"]

    cfg = load_config(args.config, overrides if overrides else None)

    # Setup results directory
    dataset_name = cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "baseline", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRC-TGAD — BASELINE EXPERIMENT")
    print(f"Dataset  : {dataset_name}")
    print(f"Seeds    : {args.seeds}")
    print(f"Epochs   : {cfg['training']['epochs']}")
    print(f"Mock mode: {args.mock}")
    print(f"Results  : {results_dir}")

    # Run across all seeds
    all_results = []
    for seed in args.seeds:
        result = run_single_seed(cfg, seed, args.mock, results_dir)
        all_results.append(result)

    # Aggregate across seeds
    print(f"\n{'='*55}")
    print(f"  BASELINE FINAL RESULTS  ({dataset_name}, {len(args.seeds)} seeds)")
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


if __name__ == "__main__":
    main()