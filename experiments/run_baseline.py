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
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg, seed, mock=False):
    """
    Returns (train_dataset, val_dataset, test_dataset).

    Mock mode : synthetic data, no files needed.
    Real mode : calls Person 1's load_smap() / load_msl() / load_swat().

    KEY FACTS about Person 1's datasets (from base_dataset.py):
      - __getitem__ returns dict with keys:
            x_window : Tensor [W, 1]   (d_in=1, univariate per node)
            node_id  : int
            t        : int
            graph    : torch_geometric Data  (shared static graph)
            label    : int  (1 if ANY step in window is anomalous)
      - as_flat_list() returns [(node_id, t, label), ...]
        → we attach this as .as_tuples for the curriculum scheduler
    """
    if mock:
        d_in       = cfg["model"]["d_in"]
        win        = cfg["model"]["window_size"]
        train_data = MockTemporalGraphDataset(n_nodes=10, T=500, window_size=win, d_in=d_in, seed=seed)
        val_data   = MockTemporalGraphDataset(n_nodes=10, T=150, window_size=win, d_in=d_in, seed=seed+100)
        test_data  = MockTemporalGraphDataset(n_nodes=10, T=200, window_size=win, d_in=d_in, seed=seed+200)
        return train_data, val_data, test_data

    # ── REAL MODE ─────────────────────────────────────────────────────────────
    # Person 1 exposes plain functions, not class constructors.
    # load_smap / load_msl return (train_ds, val_ds, test_ds, channel_ids)
    # load_swat  returns (train_ds, val_ds, test_ds, channel_ids)  [same pattern]
    dataset_name = cfg["data"]["dataset"]
    win    = cfg["model"]["window_size"]
    stride = cfg["data"].get("stride", 1)

    if dataset_name == "swat":
        from data.swat import load_swat
        train_data, val_data, test_data, _ = load_swat(
            data_dir  = cfg["data"]["data_dir"],
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"]["val_split"],
        )
    elif dataset_name == "smap":
        from data.smap import load_smap
        train_data, val_data, test_data, _ = load_smap(
            data_dir  = cfg["data"].get("data_dir", "data/raw/smap"),
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"]["val_split"],
        )
    elif dataset_name == "msl":
        from data.smap import load_msl      # load_msl lives inside smap.py
        train_data, val_data, test_data, _ = load_msl(
            data_dir  = cfg["data"].get("data_dir", "data/raw/smap"),
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"]["val_split"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Auto-detect num_nodes from actual data (prevents SMAP/SWAT mismatch)
    cfg["model"]["num_nodes"] = train_data.N
    print(f"[Dataset] Auto-detected num_nodes = {train_data.N}")

    # Attach .as_tuples — curriculum scheduler needs (node_id, t, label) list.
    # Person 1's dataset exposes as_flat_list() for exactly this purpose.
    train_data.as_tuples = train_data.as_flat_list()
    val_data.as_tuples   = val_data.as_flat_list()
    test_data.as_tuples  = test_data.as_flat_list()

    return train_data, val_data, test_data


def load_backbone(cfg, mock=False):
    """
    Returns backbone model.
    Mock mode : random linear layers (no torch_geometric needed).
    Real mode : Person 1's Backbone — takes individual kwargs, NOT a config dict.
    """
    if mock:
        return MockBackbone(
            d_in=cfg["model"]["d_in"],
            d_z=cfg["model"]["gnn_out_dim"],
            num_nodes=10   # matches MockTemporalGraphDataset default
        )
    # ── REAL MODE ─────────────────────────────────────────────────────────────
    from backbone.backbone import Backbone
    return Backbone(
        d_in        = cfg["model"]["d_in"],           # 1 (univariate per node)
        hidden_size = cfg["model"]["lstm_hidden"],    # 64
        gnn_out_dim = cfg["model"]["gnn_out_dim"],    # 64
        num_nodes   = cfg["model"]["num_nodes"],      # 51 SWAT / 55 SMAP,MSL
        window_size = cfg["model"]["window_size"],    # 30
        lstm_layers = cfg["model"]["lstm_layers"],    # 2
        gat_heads   = cfg["model"]["gnn_heads"],      # 4
        dropout     = cfg["model"]["dropout"],        # 0.1
    )


def load_rag_scorer(cfg, mock=False):
    """Baseline always uses mock scorer — curriculum is OFF, scores are unused."""
    return MockRAGScorer()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(backbone, test_dataset, cfg, device) -> dict:
    """
    Run inference on test set, compute all metrics.

    IMPORTANT — how Person 1's backbone works:
      - backbone.forward(x_windows, graph) expects x_windows: [N, W, d_in]
        for ALL nodes at a single timestep, returns z_all [N,d_z], x_hat_all [N,d_in]
      - We group by timestep t, stack all node windows, run one forward pass,
        then collect per-node scores.
    """
    import torch
    from torch.utils.data import DataLoader
    from collections import defaultdict

    backbone.eval()
    all_scores = []
    all_labels = []

    # ───────────────── MOCK PATH ─────────────────
    if isinstance(backbone, MockBackbone):

        def mock_collate(batch):
            return {
                "x_window": torch.stack([b["x_window"] for b in batch]),
                "target":   torch.stack([b["target"] for b in batch]) if "target" in batch[0] else torch.stack([b["x_window"][:, -1, :] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch]),
                "node_id": [b["node_id"] for b in batch],
                "t": [b["t"] for b in batch],
                "graph": None,
            }
        loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=mock_collate,
        )

        with torch.no_grad():
            for batch in loader:

                x_window = batch["x_window"].to(device, non_blocking=True)   # [B,W,d_in]
                target   = batch["target"].to(device, non_blocking=True) # [B,d_in]
                labels   = batch["label"]

                # Batched GPU forward pass
                _, x_hat_all = backbone(x_window, None)                      # [B,d_in]

                scores = torch.norm(x_hat_all - target, dim=1)                # [B]

                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.tolist())

    # ───────────────── REAL DATASET PATH ─────────────────
    else:

        with torch.no_grad():
            from torch_geometric.data import Batch, Data

            # Use raw module if wrapped in DataParallel
            backbone_module = backbone.module if isinstance(backbone, torch.nn.DataParallel) else backbone
            N = backbone_module.num_nodes
            
            n_samples = len(test_dataset)
            n_times = n_samples // N
            batch_size = cfg["training"].get("batch_size", 512)
            graph = test_dataset.graph if test_dataset.graph is not None else Data(edge_index=torch.empty((2, 0), dtype=torch.long))
            use_amp = (device != "cpu")

            # Data is ordered by timestep — slice directly
            for ti_start in range(0, n_times, batch_size):
                ti_end = min(ti_start + batch_size, n_times)
                B = ti_end - ti_start

                idx_start = ti_start * N
                idx_end = ti_end * N

                x_all = test_dataset._precomputed_windows[idx_start:idx_end].to(
                    device, non_blocking=True)
                chunk_labels = test_dataset._precomputed_labels[idx_start:idx_end]

                x_all_reshaped = x_all.view(B, N, -1, x_all.shape[-1])

                with torch.amp.autocast('cuda', enabled=use_amp):
                    z_all, x_hat_all = backbone(x_all_reshaped, graph)

                x_hat_all = x_hat_all.view(B * N, -1)
                
                # Forecast target: signals[t] — one step AFTER the window
                target = test_dataset._precomputed_targets[idx_start:idx_end].to(
                    device, non_blocking=True)
                
                scores = torch.norm(x_hat_all - target, dim=1)

                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(chunk_labels.tolist())

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

    cfg["training"]["seed"]    = seed
    cfg["logging"]["run_name"] = f"baseline_seed{seed}"

    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone                        = load_backbone(cfg, mock)
    rag_scorer                      = load_rag_scorer(cfg, mock)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

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
        use_curriculum=False,
        device=device
    )

    history = trainer.train(
        val_dataset=val_data,
        save_dir=os.path.join(results_dir, f"seed{seed}")
    )

    print(f"\n[Baseline] Evaluating on test set (seed={seed})...")
    test_results = evaluate_on_test(backbone, test_data, cfg, device)

    print(f"\n  Test Results (seed={seed}):")
    print(f"    F1-PA    : {test_results['f1_pa']:.4f}")
    print(f"    AUC-PR   : {test_results['auc_pr']:.4f}")
    print(f"    AUC-ROC  : {test_results['auc_roc']:.4f}")
    print(f"    Precision: {test_results['precision']:.4f}")
    print(f"    Recall   : {test_results['recall']:.4f}")

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

    overrides = args.override or []
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")
    if args.mock:
        overrides += ["training.epochs=20", "training.device=cpu"]

    cfg = load_config(args.config, overrides if overrides else None)

    dataset_name = cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "baseline", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRC-TGAD — BASELINE EXPERIMENT")
    print(f"Dataset  : {dataset_name}")
    print(f"Seeds    : {args.seeds}")
    print(f"Epochs   : {cfg['training']['epochs']}")
    print(f"Mock mode: {args.mock}")
    print(f"Results  : {results_dir}")

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(cfg, seed, args.mock, results_dir)
        all_results.append(result)

    print(f"\n{'='*55}")
    print(f"  BASELINE FINAL RESULTS  ({dataset_name}, {len(args.seeds)} seeds)")
    print(f"{'='*55}")

    for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]:
        vals = [r[metric] for r in all_results]
        print(f"  {metric:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

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
