# curriculum/trainer.py
# PERSON 3 — Full Training Loop
# RC-TGAD: Retrieval-Augmented Curriculum Training for Temporal Graph Anomaly Detection

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from typing import Dict, List, Tuple, Optional
from curriculum.scheduler import get_batch, pacing


# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    def __init__(self, d_in: int = 10, d_z: int = 64):
        super().__init__()
        self.d_in = d_in
        self.d_z = d_z
        self.num_nodes = 51
        self.lstm_mock  = nn.Linear(d_in, d_z)
        self.recon_mock = nn.Linear(d_z, d_in)

    def get_embedding(self, x_window, graph=None, node_id=None, t=None):
        x_flat = x_window.mean(dim=0)
        z      = torch.relu(self.lstm_mock(x_flat))
        x_hat  = self.recon_mock(z)
        return z, x_hat

    def forward(self, x_windows, graph=None):
        """
        Batched forward for GPU efficiency.
        x_windows : [B, W, d_in]
        returns   : z_all [B, d_z],  x_hat_all [B, d_in]
        """
        x_flat    = x_windows.mean(dim=1)                        # [B, d_in]
        z_all     = torch.relu(self.lstm_mock(x_flat))           # [B, d_z]
        x_hat_all = self.recon_mock(z_all)                       # [B, d_in]
        return z_all, x_hat_all


class MockRAGScorer:
    def __init__(self, seed: int = 42):
        self.rng    = np.random.RandomState(seed)
        self._cache: Dict[Tuple, float] = {}

    def score_hardness(self, z, x, x_hat, node_id, graph, t,
                       window_errors=None, label: int = 0) -> float:
        key = (node_id, t)
        if key not in self._cache:
            self._cache[key] = float(self.rng.random())
        return self._cache[key]

    def reset(self):
        """Clear cache between rescore passes."""
        self._cache.clear()

    def get_all_scores(self, dataset) -> Dict[Tuple, float]:
        return {
            (node_id, t): self.score_hardness(None, None, None, node_id, None, t)
            for (node_id, t, _) in dataset
        }


class MockTemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, n_nodes: int = 10, T: int = 200,
                 window_size: int = 30, d_in: int = 10,
                 anomaly_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.window_size = window_size
        self.d_in = d_in

        raw    = rng.randn(n_nodes, T + window_size, d_in).astype(np.float32)
        labels = np.zeros((n_nodes, T), dtype=np.int64)
        for n in range(n_nodes):
            anomaly_times = rng.choice(T, size=int(T * anomaly_ratio), replace=False)
            raw[n, anomaly_times] += rng.randn(len(anomaly_times), d_in) * 5
            labels[n, anomaly_times] = 1

        self.samples = []
        for n in range(n_nodes):
            for t in range(T):
                window = raw[n, t: t + window_size]
                self.samples.append({
                    "x_window": torch.tensor(window),
                    "node_id":  n,
                    "t":        t,
                    "label":    int(labels[n, t]),
                    "graph":    None
                })

        self.as_tuples = [
            (s["node_id"], s["t"], s["label"]) for s in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_graph(N: int, device: str) -> Data:
    """
    Fully-connected graph for a batch of N nodes.
    Used only when no real graph is available (mock mode).
    """
    edge_index = torch.combinations(torch.arange(N), r=2).T      # [2, E]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(edge_index=edge_index.to(device))


def _collate_graph(batch_list):
    """
    Custom collate: stacks tensors normally but handles
    torch_geometric Data objects by returning the first one
    (the graph is shared/static — same for all samples in a batch).
    """
    x_windows = torch.stack([b["x_window"] for b in batch_list])
    node_ids  = torch.tensor([b["node_id"]  for b in batch_list])
    ts        = torch.tensor([b["t"]        for b in batch_list])
    labels    = torch.tensor([b["label"]    for b in batch_list])

    # Graph: shared static object — just pass the first one through
    graph = batch_list[0].get("graph", None)

    return {
        "x_window": x_windows,
        "node_id":  node_ids,
        "t":        ts,
        "label":    labels,
        "graph":    graph,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(
        self,
        backbone,
        rag_scorer,
        dataset,
        config: Dict,
        use_curriculum: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.backbone       = backbone.to(device)
        self.rag_scorer     = rag_scorer
        self.dataset        = dataset
        self.config         = config
        self.use_curriculum = use_curriculum
        self.device         = device

        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        self.recon_loss_fn = nn.MSELoss()
        self.cls_loss_fn   = nn.BCEWithLogitsLoss()

        self.history = {
            "train_loss": [],
            "val_f1":     [],
            "val_auc_pr": [],
            "n_samples":  [],
            "pct_data":   [],
        }

        self.wandb = None
        if config.get("use_wandb", False):
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "rc-tgad"),
                    config=config,
                    name=config.get("run_name", "run")
                )
                self.wandb = wandb
                print("[Trainer] wandb initialized.")
            except ImportError:
                print("[Trainer] wandb not installed — skipping.")

        print(f"[Trainer] Device    : {self.device}")
        print(f"[Trainer] Curriculum: {'ON' if use_curriculum else 'OFF (baseline mode)'}")
        print(f"[Trainer] Dataset   : {len(dataset)} samples")

    # ── HARDNESS SCORING ─────────────────────────────────────────────────────

    def _compute_all_hardness_scores(self) -> Dict[Tuple, float]:
        """
        Score every sample by running a real forward pass through the backbone
        and calling score_hardness() with real embeddings + ground-truth labels.

        This ensures:
          1. H_temp uses real reconstruction errors (not 0.5 placeholder)
          2. H_RAG store is populated with real embeddings AND real labels
             so entropy is meaningful from the very first epoch
          3. Scores refresh as backbone improves (called every k_rescore epochs)
        """
        from collections import defaultdict

        print("[Trainer] Pre-computing hardness scores (real forward pass)...")
        t0      = time.time()
        scores  = {}

        self.backbone.eval()

        # Group by timestep so we can do batched N-node forward passes
        t_buckets = defaultdict(list)
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            t_buckets[int(sample["t"])].append((idx, sample))

        with torch.no_grad():
            for t, idx_samples in t_buckets.items():
                samples = [s for _, s in idx_samples]
                idxs    = [i for i, _ in idx_samples]

                N    = self.backbone.num_nodes
                W    = samples[0]["x_window"].shape[0]
                d_in = samples[0]["x_window"].shape[1]

                x_all  = torch.zeros(N, W, d_in)
                labels = torch.zeros(N, dtype=torch.long)
                graph  = samples[0].get("graph", None)

                for s in samples:
                    nid = int(s["node_id"])
                    if nid < N:
                        x_all[nid]  = s["x_window"]
                        labels[nid] = int(s["label"])

                x_all = x_all.to(self.device)
                if graph is not None:
                    graph = graph.to(self.device)

                z_all, x_hat_all = self.backbone(x_all, graph)

                # Score each node that appeared in this timestep bucket
                node_ids_in_bucket = {int(s["node_id"]) for s in samples}
                for nid in node_ids_in_bucket:
                    if nid >= N:
                        continue
                    z    = z_all[nid].cpu()
                    x    = x_all[nid, -1, :].cpu()   # last timestep
                    xhat = x_hat_all[nid].cpu()
                    lbl  = int(labels[nid].item())

                    if hasattr(self.rag_scorer, "score_hardness"):
                        H = self.rag_scorer.score_hardness(
                            z, x, xhat, nid, graph, t, label=lbl
                        )
                    else:
                        H = 0.5
                    scores[(nid, t)] = H

        self.backbone.train()

        vals = list(scores.values())
        print(f"[Trainer] Scored {len(scores)} samples in {time.time()-t0:.2f}s  "
              f"mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")
        return scores

    # ── SINGLE EPOCH ─────────────────────────────────────────────────────────

    def _train_epoch(self, indices: List[int], batch_size: int) -> float:

        from collections import defaultdict

        self.backbone.train()

        total_loss = 0.0
        n_steps = 0

        # --------------------------------------------------
        # Group samples by timestep
        # --------------------------------------------------

        t_buckets = defaultdict(list)

        for idx in indices:
            sample = self.dataset[idx]

            t_buckets[int(sample["t"])].append(sample)

        # --------------------------------------------------
        # Train one timestep at a time
        # --------------------------------------------------

        for t, samples in t_buckets.items():

            N = self.backbone.num_nodes
            W = samples[0]["x_window"].shape[0]
            d_in = samples[0]["x_window"].shape[1]

            x_all = torch.zeros(N, W, d_in)
            labels = torch.zeros(N)

            graph = samples[0]["graph"]
            if graph is not None:
                graph = graph.to(self.device)

            for s in samples:
                nid = int(s["node_id"])

                if nid >= N:
                    continue

                x_all[nid] = s["x_window"]
                labels[nid] = s["label"]

            x_all = x_all.to(self.device)
            labels = labels.to(self.device)

            if graph is not None: graph = graph.to(self.device)

            self.optimizer.zero_grad()

            # -----------------------------
            # Forward pass
            # -----------------------------

            z_all, x_hat_all = self.backbone(x_all, graph)

            # Reconstruct last timestep (not mean) — matches Person 1's LSTM
            # which predicts x[:, -1, :] via the recon_head
            target = x_all[:, -1, :]          # [N, d_in]  last step

            recon_loss = self.recon_loss_fn(x_hat_all, target)

            residuals = x_hat_all - target
            anomaly_logit = torch.norm(residuals, dim=1)

            cls_loss = self.cls_loss_fn(anomaly_logit, labels)

            loss = recon_loss + 0.5 * cls_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)
    # ── VALIDATION ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:

        from collections import defaultdict
        from utils.metrics import compute_f1, compute_auc_pr

        self.backbone.eval()

        all_scores = []
        all_labels = []

        # Group samples by timestep
        t_buckets = defaultdict(list)

        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            t_buckets[int(sample["t"])].append(sample)

        for t, samples in t_buckets.items():

            N = self.backbone.num_nodes
            W = samples[0]["x_window"].shape[0]
            d_in = samples[0]["x_window"].shape[1]

            x_all = torch.zeros(N, W, d_in)
            labels = torch.zeros(N)

            graph = samples[0]["graph"]
            if graph is not None:
                graph = graph.to(self.device)

            for s in samples:
                nid = int(s["node_id"])
                if nid >= N:
                    continue

                x_all[nid] = s["x_window"]
                labels[nid] = s["label"]

            x_all = x_all.to(self.device)

            _, x_hat_all = self.backbone(x_all, graph)

            target = x_all[:, -1, :]   # last timestep — matches training target
            scores = torch.norm(x_hat_all - target, dim=1)

            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

        return compute_f1(all_scores, all_labels), compute_auc_pr(all_scores, all_labels)
    

        # ── MAIN TRAIN LOOP ──────────────────────────────────────────────────────

    def train(self, val_dataset=None, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)

        epochs     = self.config.get("epochs", 100)
        k_warmup   = self.config.get("k_warmup", 30)
        batch_size = self.config.get("batch_size", 64)
        best_f1    = -1.0
        best_epoch = -1

        # How often to refresh hardness scores (every 10 epochs by default)
        # Refreshing lets H_RAG improve as the backbone gets better trained
        k_rescore = self.config.get("k_rescore", 10)

        if self.use_curriculum:
            hardness_scores = self._compute_all_hardness_scores()
        else:
            hardness_scores = {
                (node_id, t): 0.0
                for (node_id, t, _) in self.dataset.as_tuples
            }

        print(f"\n[Trainer] Starting training for {epochs} epochs...")
        print(f"[Trainer] Hardness scores refreshed every {k_rescore} epochs")
        print("-" * 60)

        for epoch in range(epochs):

            t_start = time.time()

            # ── Re-score hardness periodically so H_RAG improves with backbone ──
            if self.use_curriculum and epoch > 0 and epoch % k_rescore == 0:
                # Reset store so stale embeddings from early training don't persist
                if hasattr(self.rag_scorer, "reset"):
                    self.rag_scorer.reset()
                hardness_scores = self._compute_all_hardness_scores()
                vals = list(hardness_scores.values())
                print(f"[Trainer] Epoch {epoch}: Rescored hardness  "
                      f"mean={float(np.mean(vals)):.3f}  std={float(np.std(vals)):.3f}")

            if self.use_curriculum:
                indices = get_batch(
                    self.dataset.as_tuples,
                    hardness_scores,
                    epoch,
                    k_warmup,
                    verbose=False
                )
            else:
                indices = list(range(len(self.dataset)))

            train_loss = self._train_epoch(indices, batch_size)

            epoch_time = time.time() - t_start

            f1, auc_pr = 0.0, 0.0

            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):

                f1, auc_pr = self._validate(val_dataset)

                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch

                    ckpt_path = os.path.join(save_dir, "best_model.pt")

                    torch.save({
                        "epoch": epoch,
                        "model_state": self.backbone.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "f1": f1,
                        "auc_pr": auc_pr,
                        "config": self.config
                    }, ckpt_path)

            pct = 100 * len(indices) / len(self.dataset)
            lam = pacing(epoch, k_warmup)

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc_pr"].append(auc_pr)
            self.history["n_samples"].append(len(indices))
            self.history["pct_data"].append(pct)

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"λ={lam:.3f} | "
                    f"data={pct:5.1f}% | "
                    f"loss={train_loss:.4f} | "
                    f"F1={f1:.4f} AUC-PR={auc_pr:.4f} | "
                    f"{epoch_time:.1f}s"
                )

        print("-" * 60)
        print(f"[Trainer] Done. Best F1={best_f1:.4f} at epoch {best_epoch}")

        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    os.makedirs("utils", exist_ok=True)
    if not os.path.exists("utils/metrics.py"):
        with open("utils/metrics.py", "w") as f:
            f.write("""
from sklearn.metrics import f1_score, average_precision_score
import numpy as np

def compute_f1(scores, labels, threshold=None):
    scores = np.array(scores); labels = np.array(labels)
    if threshold is None: threshold = np.percentile(scores, 90)
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0))

def compute_auc_pr(scores, labels):
    scores = np.array(scores); labels = np.array(labels)
    if labels.sum() == 0: return 0.0
    return float(average_precision_score(labels, scores))
""")

    for d in ["curriculum", "utils"]:
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()

    print("=" * 60)
    print("TRAINER SMOKE TEST")
    print("=" * 60)

    CONFIG = {
        "epochs": 20, "k_warmup": 10, "batch_size": 32,
        "lr": 1e-3, "weight_decay": 1e-5, "use_wandb": False, "run_name": "smoke"
    }

    train_ds = MockTemporalGraphDataset(n_nodes=5, T=100, window_size=10, d_in=8)
    val_ds   = MockTemporalGraphDataset(n_nodes=5, T=50,  window_size=10, d_in=8, seed=99)
    scorer   = MockRAGScorer(seed=42)

    print("\n--- Baseline (no curriculum) ---")
    t1 = Trainer(MockBackbone(d_in=8, d_z=32), scorer, train_ds, CONFIG, use_curriculum=False)
    h1 = t1.train(val_dataset=val_ds, save_dir="checkpoints/baseline")

    print("\n--- RC-TGAD (curriculum ON) ---")
    t2 = Trainer(MockBackbone(d_in=8, d_z=32), scorer, train_ds, CONFIG, use_curriculum=True)
    h2 = t2.train(val_dataset=val_ds, save_dir="checkpoints/rctgad")

    print(f"\nBaseline F1 : {h1['val_f1'][-1]:.4f}")
    print(f"RC-TGAD  F1 : {h2['val_f1'][-1]:.4f}")
    print("\nSmoke test passed ✅")