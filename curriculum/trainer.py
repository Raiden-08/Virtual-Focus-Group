# curriculum/trainer.py
# PERSON 3 — Full Training Loop
# RC-TGAD: Retrieval-Augmented Curriculum Training for Temporal Graph Anomaly Detection
#
# This trainer is designed to work in THREE modes:
#   Mode 1 (NOW)     : fully mocked backbone + scorer — you can run this TODAY
#   Mode 2 (Week 2)  : real backbone (Person 1) + mocked scorer
#   Mode 3 (Merge)   : real backbone (Person 1) + real scorer (Person 2)
#
# You switch modes by changing the imports at the bottom — nothing else changes.

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
from curriculum.scheduler import get_batch, pacing


# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLASSES
# These mimic the exact interfaces Person 1 and Person 2 will deliver.
# Replace with real imports at merge time — the Trainer class never changes.
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    """
    Mimics Person 1's backbone.py interface.
    Returns random embeddings z and reconstruction x_hat.
    Replace with: from backbone.backbone import Backbone
    """
    def __init__(self, d_in: int = 10, d_z: int = 64):
        super().__init__()
        self.d_in = d_in
        self.d_z = d_z
        # Simple linear layers to make it trainable (so optimizer has params)
        self.lstm_mock = nn.Linear(d_in, d_z)
        self.recon_mock = nn.Linear(d_z, d_in)

    def get_embedding(self, x_window, graph=None, node_id=None, t=None):
        """
        Interface A (fixed contract with Person 1):
        Input:  x_window — Tensor [W, d_in]
        Output: z        — Tensor [d_z]
                x_hat    — Tensor [d_in]
        """
        x_flat = x_window.mean(dim=0)               # [d_in]
        z = torch.relu(self.lstm_mock(x_flat))       # [d_z]
        x_hat = self.recon_mock(z)                   # [d_in]
        return z, x_hat

    def forward(self, x_window, graph=None, node_id=None, t=None):
        return self.get_embedding(x_window, graph, node_id, t)


class MockRAGScorer:
    """
    Mimics Person 2's rag_scorer.py interface.
    Returns random hardness scores in [0, 1].
    Replace with: from rag.rag_scorer import RAGScorer
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._cache: Dict[Tuple, float] = {}

    def score_hardness(self, z, x, x_hat, node_id, graph, t,
                       window_errors=None) -> float:
        """
        Interface B (fixed contract with Person 2):
        Input:  z        — Tensor [d_z]
                x        — Tensor [d_in]
                x_hat    — Tensor [d_in]
                node_id  — int
                graph    — torch_geometric Data (or None in mock mode)
                t        — int
        Output: H        — float in [0, 1]
        """
        key = (node_id, t)
        if key not in self._cache:
            self._cache[key] = float(self.rng.random())
        return self._cache[key]

    def get_all_scores(self, dataset) -> Dict[Tuple, float]:
        """Pre-compute scores for entire dataset. Used by scheduler."""
        return {
            (node_id, t): self.score_hardness(None, None, None, node_id, None, t)
            for (node_id, t, _) in dataset
        }


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATASET
# Mimics what Person 1's data/ pipeline will return.
# Returns sliding windows of shape [W, d_in] with binary anomaly labels.
# ─────────────────────────────────────────────────────────────────────────────

class MockTemporalGraphDataset(torch.utils.data.Dataset):
    """
    Mock dataset that returns (x_window, node_id, t, label) tuples.
    Replace with: from data.swat import SWATDataset
    """
    def __init__(self, n_nodes: int = 10, T: int = 200,
                 window_size: int = 30, d_in: int = 10,
                 anomaly_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.window_size = window_size
        self.d_in = d_in

        # Generate synthetic time series for each node
        raw = rng.randn(n_nodes, T + window_size, d_in).astype(np.float32)

        # Inject anomalies: random spikes in 10% of timesteps
        labels = np.zeros((n_nodes, T), dtype=np.int64)
        for n in range(n_nodes):
            anomaly_times = rng.choice(T, size=int(T * anomaly_ratio), replace=False)
            raw[n, anomaly_times] += rng.randn(len(anomaly_times), d_in) * 5
            labels[n, anomaly_times] = 1

        # Build flat list of (node_id, t, label, window)
        self.samples = []
        for n in range(n_nodes):
            for t in range(T):
                window = raw[n, t: t + window_size]  # [W, d_in]
                self.samples.append({
                    "x_window": torch.tensor(window),
                    "node_id":  n,
                    "t":        t,
                    "label":    int(labels[n, t])
                })

        # Also expose as flat tuples for scheduler
        self.as_tuples = [
            (s["node_id"], s["t"], s["label"]) for s in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop for RC-TGAD.

    Modes (set use_curriculum=True/False, real or mock backbone/scorer):
      - Baseline:    use_curriculum=False, MockBackbone, MockRAGScorer
      - RC-TGAD:     use_curriculum=True,  MockBackbone, MockRAGScorer
      - Post-merge:  use_curriculum=True,  RealBackbone, RealRAGScorer
    """

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

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        # Loss: MSE for reconstruction + BCE for anomaly classification
        self.recon_loss_fn = nn.MSELoss()
        self.cls_loss_fn   = nn.BCEWithLogitsLoss()

        # Logging
        self.history = {
            "train_loss":   [],
            "val_f1":       [],
            "val_auc_pr":   [],
            "n_samples":    [],   # how many samples trained per epoch
            "pct_data":     [],   # % of dataset used per epoch
        }

        # Try to init wandb — skip gracefully if not installed
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
                print("[Trainer] wandb not installed — logging to console only.")

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Curriculum: {'ON' if use_curriculum else 'OFF (baseline mode)'}")
        print(f"[Trainer] Dataset size: {len(dataset)} samples")

    # ── HARDNESS SCORING ─────────────────────────────────────────────────────

    def _compute_all_hardness_scores(self) -> Dict[Tuple, float]:
        """
        Pre-compute hardness scores for all samples before training starts.
        In mock mode: random scores from MockRAGScorer.
        Post-merge:   real scores from Person 2's RAGScorer.

        Note: H_RAG depends on the vector store which grows during training.
        For simplicity in v1, we pre-compute once before epoch 1.
        In v2 (paper extension), recompute every N epochs.
        """
        print("[Trainer] Pre-computing hardness scores...")
        t0 = time.time()
        scores = self.rag_scorer.get_all_scores(self.dataset.as_tuples)
        print(f"[Trainer] Scored {len(scores)} samples in {time.time()-t0:.2f}s")

        # Log score distribution
        vals = list(scores.values())
        print(f"[Trainer] Hardness distribution — "
              f"mean={np.mean(vals):.3f}  "
              f"std={np.std(vals):.3f}  "
              f"min={np.min(vals):.3f}  "
              f"max={np.max(vals):.3f}")
        return scores

    # ── SINGLE EPOCH ─────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        indices: List[int],
        batch_size: int
    ) -> float:
        """
        Train on the subset of samples specified by indices.
        Returns mean loss for this epoch.
        """
        self.backbone.train()

        # Build subset DataLoader from curriculum-selected indices
        subset  = Subset(self.dataset, indices)
        loader  = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == "cuda")
        )

        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            x_window = batch["x_window"].to(self.device)   # [B, W, d_in]
            labels   = batch["label"].float().to(self.device)  # [B]

            self.optimizer.zero_grad()

            # Process each sample in the batch
            # (In real version Person 1's backbone handles batching natively)
            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for b in range(x_window.shape[0]):
                z, x_hat = self.backbone.get_embedding(
                    x_window[b],                      # [W, d_in]
                    graph=None,
                    node_id=int(batch["node_id"][b]),
                    t=int(batch["t"][b])
                )

                # Reconstruction loss (unsupervised signal)
                recon_loss = self.recon_loss_fn(x_hat, x_window[b].mean(dim=0))

                # Anomaly score: use reconstruction error magnitude as logit
                anomaly_logit = torch.norm(x_hat - x_window[b].mean(dim=0)).unsqueeze(0)
                cls_loss = self.cls_loss_fn(anomaly_logit, labels[b].unsqueeze(0))

                batch_loss = batch_loss + recon_loss + 0.5 * cls_loss

            batch_loss = batch_loss / x_window.shape[0]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── VALIDATION ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:
        """
        Run validation. Returns (F1, AUC-PR).
        Imports metrics from utils/metrics.py (Person 3 also owns this).
        """
        from utils.metrics import compute_f1, compute_auc_pr

        self.backbone.eval()
        all_scores = []
        all_labels = []

        loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

        for batch in loader:
            x_window = batch["x_window"].to(self.device)
            labels   = batch["label"]

            for b in range(x_window.shape[0]):
                z, x_hat = self.backbone.get_embedding(x_window[b])
                score = torch.norm(x_hat - x_window[b].mean(dim=0)).item()
                all_scores.append(score)
                all_labels.append(int(labels[b]))

        f1     = compute_f1(all_scores, all_labels)
        auc_pr = compute_auc_pr(all_scores, all_labels)
        return f1, auc_pr

    # ── MAIN TRAIN LOOP ──────────────────────────────────────────────────────

    def train(
        self,
        val_dataset=None,
        save_dir: str = "checkpoints"
    ):
        """
        Main training loop.

        Args:
            val_dataset : dataset to validate on (if None, skips validation)
            save_dir    : where to save best model checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)

        epochs     = self.config.get("epochs", 100)
        k_warmup   = self.config.get("k_warmup", 30)
        batch_size = self.config.get("batch_size", 64)
        best_f1    = -1.0
        best_epoch = -1

        # Pre-compute hardness scores once before training
        if self.use_curriculum:
            hardness_scores = self._compute_all_hardness_scores()
        else:
            # Baseline: uniform scores — scheduler will include everything
            hardness_scores = {
                (node_id, t): 0.0
                for (node_id, t, _) in self.dataset.as_tuples
            }

        print(f"\n[Trainer] Starting training for {epochs} epochs...")
        print("-" * 60)

        for epoch in range(epochs):
            t_start = time.time()

            # ── Curriculum sample selection ─────────────────────────────────
            if self.use_curriculum:
                indices = get_batch(
                    self.dataset.as_tuples,
                    hardness_scores,
                    epoch,
                    k_warmup,
                    verbose=False
                )
            else:
                indices = list(range(len(self.dataset)))  # all samples

            # ── Train one epoch ─────────────────────────────────────────────
            train_loss = self._train_epoch(indices, batch_size)
            epoch_time = time.time() - t_start

            # ── Validation (every 5 epochs) ─────────────────────────────────
            f1, auc_pr = 0.0, 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                f1, auc_pr = self._validate(val_dataset)

                # Save best checkpoint
                if f1 > best_f1:
                    best_f1    = f1
                    best_epoch = epoch
                    ckpt_path  = os.path.join(save_dir, "best_model.pt")
                    torch.save({
                        "epoch":      epoch,
                        "model_state": self.backbone.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "f1":         f1,
                        "auc_pr":     auc_pr,
                        "config":     self.config
                    }, ckpt_path)

            # ── Logging ─────────────────────────────────────────────────────
            pct = 100 * len(indices) / len(self.dataset)
            lam = pacing(epoch, k_warmup)

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc_pr"].append(auc_pr)
            self.history["n_samples"].append(len(indices))
            self.history["pct_data"].append(pct)

            if self.wandb:
                self.wandb.log({
                    "epoch":       epoch,
                    "train_loss":  train_loss,
                    "val_f1":      f1,
                    "val_auc_pr":  auc_pr,
                    "n_samples":   len(indices),
                    "pct_data":    pct,
                    "lambda":      lam,
                })

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch:4d}/{epochs}  |  "
                    f"λ={lam:.3f}  |  "
                    f"data={pct:5.1f}%  |  "
                    f"loss={train_loss:.4f}  |  "
                    f"F1={f1:.4f}  AUC-PR={auc_pr:.4f}  |  "
                    f"time={epoch_time:.1f}s"
                )

        print("-" * 60)
        print(f"[Trainer] Training complete.")
        if val_dataset is not None:
            print(f"[Trainer] Best F1={best_f1:.4f} at epoch {best_epoch}")
        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST — run this file directly to verify trainer works
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Create a minimal utils/metrics.py if it doesn't exist yet
    os.makedirs("utils", exist_ok=True)
    if not os.path.exists("utils/metrics.py"):
        with open("utils/metrics.py", "w") as f:
            f.write("""
# Placeholder metrics — replace with full metrics.py from Day 5-6
from sklearn.metrics import f1_score, average_precision_score
import numpy as np

def compute_f1(scores, labels, threshold=None):
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = np.percentile(scores, 90)
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0))

def compute_auc_pr(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))
""")
        print("[Setup] Created placeholder utils/metrics.py")

    os.makedirs("curriculum", exist_ok=True)
    if not os.path.exists("curriculum/__init__.py"):
        open("curriculum/__init__.py", "w").close()

    if not os.path.exists("utils/__init__.py"):
        open("utils/__init__.py", "w").close()

    print("=" * 60)
    print("TRAINER SMOKE TEST")
    print("=" * 60)

    CONFIG = {
        "epochs":       20,        # short run for testing
        "k_warmup":     10,
        "batch_size":   32,
        "lr":           1e-3,
        "weight_decay": 1e-5,
        "use_wandb":    False,
        "run_name":     "smoke_test"
    }

    # Build mock components
    train_dataset = MockTemporalGraphDataset(n_nodes=5, T=100, window_size=10, d_in=8)
    val_dataset   = MockTemporalGraphDataset(n_nodes=5, T=50,  window_size=10, d_in=8, seed=99)
    backbone      = MockBackbone(d_in=8, d_z=32)
    rag_scorer    = MockRAGScorer(seed=42)

    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # ── Test 1: Baseline (no curriculum) ────────────────────────────────────
    print("\n--- Test 1: Baseline mode (no curriculum) ---")
    trainer_baseline = Trainer(
        backbone=MockBackbone(d_in=8, d_z=32),
        rag_scorer=rag_scorer,
        dataset=train_dataset,
        config=CONFIG,
        use_curriculum=False
    )
    history_baseline = trainer_baseline.train(val_dataset=val_dataset, save_dir="checkpoints/baseline")

    # ── Test 2: RC-TGAD (with curriculum) ───────────────────────────────────
    print("\n--- Test 2: RC-TGAD mode (curriculum ON) ---")
    trainer_rctgad = Trainer(
        backbone=MockBackbone(d_in=8, d_z=32),
        rag_scorer=rag_scorer,
        dataset=train_dataset,
        config=CONFIG,
        use_curriculum=True
    )
    history_rctgad = trainer_rctgad.train(val_dataset=val_dataset, save_dir="checkpoints/rctgad")

    print("\n--- Final Comparison ---")
    print(f"Baseline  final F1: {history_baseline['val_f1'][-1]:.4f}")
    print(f"RC-TGAD   final F1: {history_rctgad['val_f1'][-1]:.4f}")
    print("\nSmoke test passed — trainer.py is working correctly.")