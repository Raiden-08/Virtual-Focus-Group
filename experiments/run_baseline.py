"""
run_baseline.py — Train backbone (no curriculum) and log baseline F1.
Owned by Person 3, but Person 1 runs this in Week 2 to get baseline numbers.

Usage:
    python -m experiments.run_baseline --dataset swat --epochs 100
"""

import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.backbone import Backbone
from utils.metrics import evaluate


def collate_fn(batch):
    """Custom collate — graph is shared so just return first."""
    return {
        "node_id" : [b["node_id"]  for b in batch],
        "t"       : [b["t"]        for b in batch],
        "x_window": torch.stack([b["x_window"] for b in batch]),
        "graph"   : batch[0]["graph"],
        "label"   : torch.tensor([b["label"]   for b in batch]),
    }


def run_baseline(cfg: dict, dataset_name: str = "swat"):
    device = torch.device(cfg["training"]["device"]
                          if torch.cuda.is_available() else "cpu")

    # --- load dataset ---
    if dataset_name == "swat":
        from data.swat import load_swat
        train_ds, val_ds, test_ds, _ = load_swat(
            data_dir   = cfg["data"]["swat_dir"],
            window     = cfg["model"]["window_size"],
            stride     = cfg["data"]["stride"],
            train_ratio= cfg["data"]["train_ratio"],
            val_ratio  = cfg["data"]["val_ratio"],
        )
    elif dataset_name in ("smap", "msl"):
        from data.smap import load_smap, load_msl
        loader = load_smap if dataset_name == "smap" else load_msl
        train_ds, val_ds, test_ds, _ = loader(
            data_dir = cfg["data"]["smap_dir"],
            window   = cfg["model"]["window_size"],
            stride   = cfg["data"]["stride"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # --- build model ---
    model = Backbone(
        d_in        = cfg["model"]["d_in"],
        hidden_size = cfg["model"]["lstm_hidden"],
        gnn_out_dim = cfg["model"]["gnn_out_dim"],
        num_nodes   = train_ds.N,
        window_size = cfg["model"]["window_size"],
        lstm_layers = cfg["model"]["lstm_layers"],
        gat_heads   = cfg["model"]["gat_heads"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    best_val_f1 = 0.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # --- train ---
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]",
                          leave=False):
            x = batch["x_window"].to(device)    # [B, W, d_in]

            h, x_hat = model.lstm(x)
            loss = model.reconstruction_loss(x, x_hat)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"])
            optimiser.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- validate ---
        if epoch % cfg["logging"]["log_interval"] == 0 or epoch == 1:
            model.eval()
            all_scores, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x_window"].to(device)
                    _, x_hat = model.lstm(x)
                    errors = torch.norm(x[:, -1, :] - x_hat, dim=-1)
                    all_scores.extend(errors.cpu().numpy())
                    all_labels.extend(batch["label"].numpy())

            scores  = np.array(all_scores)
            labels  = np.array(all_labels)
            metrics = evaluate(scores, labels, verbose=False)

            print(f"[Epoch {epoch:3d}] loss={avg_loss:.4f}  "
                  f"val_F1={metrics['f1']:.4f}  "
                  f"val_F1-PA={metrics['f1_pa']:.4f}  "
                  f"val_AUC-PR={metrics['auc_pr']:.4f}")

            ckpt = os.path.join(cfg["training"]["checkpoint_dir"],
                                f"backbone_best_{dataset_name}.pt")
            torch.save(model.state_dict(), ckpt)

    # --- test ---
    print("\n=== Test Evaluation ===")
    model.load_state_dict(torch.load(
        os.path.join(cfg["training"]["checkpoint_dir"],
                     f"backbone_best_{dataset_name}.pt"),
        map_location=device))
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_window"].to(device)
            _, x_hat = model.lstm(x)
            errors = torch.norm(x[:, -1, :] - x_hat, dim=-1)
            all_scores.extend(errors.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    scores  = np.array(all_scores)
    labels  = np.array(all_labels)
    metrics = evaluate(scores, labels, verbose=True)

    print(f"\nBaseline Results on {dataset_name.upper()}:")
    print(f"  F1 (standard) = {metrics['f1']:.4f}")
    print(f"  F1 (PA)       = {metrics['f1_pa']:.4f}  <- report this in paper")
    print(f"  AUC-PR        = {metrics['auc_pr']:.4f}  <- report this in paper")
    print(f"  AUC-ROC       = {metrics['auc_roc']:.4f}")
    print(f"  Precision     = {metrics['precision']:.4f}")
    print(f"  Recall        = {metrics['recall']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--dataset", default="swat",
                        choices=["swat", "smap", "msl"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_baseline(cfg, args.dataset)