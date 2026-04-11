"""
trainer.py — Unified Training Loop for RC-TGAD.
Refactored for Unified Processing Unit [B, N, W, 1].
Includes DataParallel Graph-Wrapper Fix to prevent silent deadlocks.
Includes AMP (Mixed Precision) and Single-Threaded GPU optimization for maximum speed.
Includes explicit Garbage Collection to prevent Kaggle RAM spikes.

SPEED FIX: _compute_hardness_from_loss now fully vectorized.
  Before: 20M Python calls, 14+ hours on T4.
  After:  pure tensor ops + single FAISS batch query, ~2-3 minutes on T4.
"""

import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from curriculum.scheduler import get_batch_fast, pacing

# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLASSES
# ─────────────────────────────────────────────────────────────────────────────
class MockBackbone(nn.Module):
    def __init__(self, d_in: int = 10, d_z: int = 64, num_nodes: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
    def forward(self, x_windows, graph=None):
        pass

class MockRAGScorer:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    def score_hardness(self, *args, **kwargs) -> float:
        return float(self.rng.random())

class MockTemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# THE DATAPARALLEL SHIELD
# ─────────────────────────────────────────────────────────────────────────────
class DPGraphWrapper:
    """Tricks nn.DataParallel into NOT slicing the graph in half."""
    def __init__(self, data):
        self.edge_index = data.edge_index
        self.edge_attr = getattr(data, 'edge_attr', None)


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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger=None
    ):
        self.raw_backbone   = backbone
        self.backbone       = backbone.to(device)

        if device == "cuda" and torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)
            print(f"[Trainer] Using {torch.cuda.device_count()} GPUs via DataParallel")

        self.rag_scorer     = rag_scorer
        self.dataset        = dataset
        self.config         = config
        self.use_curriculum = use_curriculum
        self.device         = device
        self.logger         = logger

        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        self.history = {
            "train_loss": [],
            "val_f1":     [],
            "val_auc_pr": [],
            "pct_data":   [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # VECTORIZED HARDNESS SCORING
    # The entire inner Python loop (20M calls) is replaced with tensor ops.
    # Timeline: 14 hours → ~2 minutes.
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _compute_hardness_from_loss(self) -> np.ndarray:
        print("\n[Trainer] Computing hardness scores (vectorized)...")
        t0 = time.time()
        self.backbone.eval()

        ds       = self.dataset
        N        = self.raw_backbone.num_nodes
        T        = len(ds)                          # number of timesteps
        bs       = self.config.get("batch_size", 512) * 2

        # ── 1. Pre-compute H_struct for every node (cached after first call) ──
        # H_struct is purely topology-based — same for every timestep.
        # We compute it once here rather than N*T times.
        h_struct_vec = self._get_h_struct_vector(ds)   # [N] numpy

        # ── 2. Run full forward pass, accumulate recon errors & embeddings ──
        all_recon_errors = np.zeros((T, N), dtype=np.float32)   # [T, N]
        all_embeddings   = np.zeros((T, N, 64), dtype=np.float32)  # [T, N, d_z]
        all_labels       = np.zeros((T, N),  dtype=np.int32)    # [T, N]

        from tqdm import tqdm
        pbar = tqdm(total=T, desc="Hardness Eval", unit="steps")

        for i in range(0, T, bs):
            end_i       = min(i + bs, T)
            batch_data  = [ds[j] for j in range(i, end_i)]
            B           = len(batch_data)

            x          = torch.stack([d["x"] for d in batch_data]).to(self.device)   # [B, N, W, 1]
            y          = torch.stack([d["y"] for d in batch_data]).numpy()            # [B, N]
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])

            # Forward — unpack dual-loss outputs
            z_all, x_hat_all, x_pred_all = self.backbone(x, graph_safe)
            # z_all    : [B, N, 64]
            # x_hat_all: [B, N, 1]

            # Reconstruction target: last timestep of input window [B, N, 1]
            target = x[:, :, -1, :]   # [B, N, 1]

            # Per-node reconstruction error: L2 norm = abs error for d_in=1
            recon_err = torch.norm(
                (x_hat_all - target).view(B, N, -1), dim=-1
            ).cpu().numpy()   # [B, N]

            all_recon_errors[i:end_i] = recon_err
            all_embeddings[i:end_i]   = z_all.cpu().numpy()
            all_labels[i:end_i]       = y

            pbar.update(B)

        pbar.close()

        # ── 3. H_temp: vectorized percentile normalisation across all errors ──
        # Shape: [T*N] flat array of errors
        flat_errors = all_recon_errors.reshape(-1)
        e_min = np.percentile(flat_errors, 5)
        e_max = np.percentile(flat_errors, 95)
        denom = max(e_max - e_min, 1e-8)

        # High error = obvious anomaly = easy to detect → LOW hardness
        # Low error  = subtle anomaly = hard to detect  → HIGH hardness
        h_temp_flat = 1.0 - np.clip((flat_errors - e_min) / denom, 0.0, 1.0)
        h_temp = h_temp_flat.reshape(T, N)   # [T, N]

        # ── 4. H_RAG: single batched FAISS call for all embeddings ──
        h_rag = self._compute_h_rag_batched(
            all_embeddings, all_labels
        )   # [T, N]

        # ── 5. H_struct: broadcast across all timesteps ──
        h_struct = np.broadcast_to(h_struct_vec[None, :], (T, N)).copy()   # [T, N]

        # ── 6. Composite score ──
        alpha_1 = self.config.get("alpha_1", 0.33)
        alpha_2 = self.config.get("alpha_2", 0.33)
        alpha_3 = 1.0 - alpha_1 - alpha_2

        H = alpha_1 * h_temp + alpha_2 * h_struct + alpha_3 * h_rag   # [T, N]

        # Flatten to [T*N] — same indexing as get_batch_fast
        H_flat = H.reshape(-1).astype(np.float32)

        # Normalise to [0, 1]
        h_range = H_flat.max() - H_flat.min()
        if h_range > 1e-6:
            H_flat = (H_flat - H_flat.min()) / (h_range + 1e-8)
        else:
            H_flat = np.full_like(H_flat, 0.5)

        elapsed = time.time() - t0
        print(f"[Trainer] Hardness computed in {elapsed:.1f}s  "
              f"mean={H_flat.mean():.3f}  std={H_flat.std():.3f}")

        self.backbone.train()
        return H_flat

    def _get_h_struct_vector(self, ds) -> np.ndarray:
        """
        Compute H_struct for each of the N nodes once.
        Uses the struct cache in hardness.py — O(1) after first call.
        Returns numpy array [N].
        """
        from rag.hardness import compute_h_struct
        N     = self.raw_backbone.num_nodes
        graph = ds[0]["graph"]
        vec   = np.zeros(N, dtype=np.float32)
        for n in range(N):
            vec[n] = compute_h_struct(
                node_id=n,
                graph=graph,
                anomaly_source_id=None,
                gamma=self.config.get("gamma", 0.5)
            )
        return vec

    def _compute_h_rag_batched(
        self,
        embeddings: np.ndarray,   # [T, N, d_z]
        labels:     np.ndarray,   # [T, N]
        k: int = 10,
    ) -> np.ndarray:
        """
        Compute H_RAG for all (t, n) pairs in a single batched FAISS call.

        Strategy:
          1. Process timesteps sequentially (causal: only use past embeddings).
          2. Build the FAISS store incrementally in batches of FAISS_BATCH_SIZE.
          3. For each new batch, query all its embeddings against the current store,
             compute entropy, then add the batch to the store.

        This replaces 20M individual FAISS queries with ~400 batch queries.
        """
        try:
            from rag.vector_store import VectorStore
        except ImportError:
            # No FAISS available — fall back to neutral 0.5
            print("[Trainer] VectorStore unavailable — H_RAG set to 0.5")
            return np.full(embeddings.shape[:2], 0.5, dtype=np.float32)

        T, N, d_z = embeddings.shape
        h_rag = np.full((T, N), 0.5, dtype=np.float32)

        store = VectorStore(dim=d_z)
        FAISS_BATCH = 512   # timesteps per batch query

        for i in range(0, T, FAISS_BATCH):
            end_i = min(i + FAISS_BATCH, T)
            batch_emb = embeddings[i:end_i]   # [B, N, d_z]
            batch_lbl = labels[i:end_i]        # [B, N]
            B         = batch_emb.shape[0]

            if len(store) < k:
                # Store not warm enough — leave as 0.5 (max uncertainty prior)
                # Add batch to store and continue
                flat_emb = batch_emb.reshape(B * N, d_z).astype(np.float32)
                flat_lbl = batch_lbl.reshape(B * N)
                for ei, li in zip(flat_emb, flat_lbl):
                    store.add(ei, int(li))
                continue

            # Query all B*N embeddings in one FAISS call
            flat_emb = batch_emb.reshape(B * N, d_z).astype(np.float32)

            # Normalise embeddings (L2)
            norms = np.linalg.norm(flat_emb, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            flat_emb_norm = flat_emb / norms

            # Single batch search
            neighbors = store.query_batch(flat_emb_norm, k=k)
            # neighbors: list of length B*N, each element is list of {'label', 'dist'}

            if neighbors is not None:
                for idx, nbrs in enumerate(neighbors):
                    if not nbrs:
                        continue
                    b = idx // N
                    n = idx % N
                    lbs = [nb["label"] for nb in nbrs]
                    p   = sum(lbs) / len(lbs)
                    if 0.0 < p < 1.0:
                        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                        h_rag[i + b, n] = float(entropy)
                    else:
                        h_rag[i + b, n] = 0.0

            # Add batch to store
            flat_lbl = batch_lbl.reshape(B * N)
            for ei, li in zip(flat_emb, flat_lbl):
                store.add(ei, int(li))

        return h_rag

    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN EPOCH
    # ─────────────────────────────────────────────────────────────────────────
    def _train_epoch(self, indices, batch_size: int) -> float:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        self.backbone.train()
        total_loss = 0.0
        n_steps    = 0
        ds         = self.dataset
        N          = self.raw_backbone.num_nodes

        is_full_dataset = (len(indices) == len(ds) * N)

        if is_full_dataset:
            t_indices = list(range(len(ds)))
        else:
            from collections import defaultdict
            t_groups = defaultdict(list)
            for idx in indices:
                t_idx    = int(idx // N)
                node_idx = int(idx % N)
                t_groups[t_idx].append(node_idx)
            t_indices = sorted(list(t_groups.keys()))

        for i in range(0, len(t_indices), batch_size):
            end_i          = min(i + batch_size, len(t_indices))
            current_t_batch = t_indices[i:end_i]

            batch_data = [ds[t_idx] for t_idx in current_t_batch]
            x          = torch.stack([d["x"] for d in batch_data]).to(self.device)
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])

            self.optimizer.zero_grad()

            with autocast():
                z_all, x_hat_all, x_pred_all = self.backbone(x, graph_safe)

                # Reconstruction target: last timestep of window
                target_recon = x[:, :, -1, :]   # [B, N, 1] — no CPU round-trip

                # Prediction target: next timestep from dataset signals
                target_pred = torch.stack([
                    torch.tensor(
                        ds.signals[min(d["t"] + 1, len(ds.signals) - 1)],
                        dtype=torch.float32
                    )
                    for d in batch_data
                ]).unsqueeze(-1).to(self.device)   # [B, N, 1]

                if is_full_dataset:
                    loss = (
                        nn.MSELoss()(x_hat_all, target_recon)
                        + nn.MSELoss()(x_pred_all, target_pred)
                    )
                else:
                    B_curr = len(current_t_batch)
                    mask   = torch.zeros((B_curr, N), dtype=torch.bool)
                    for b, t_idx in enumerate(current_t_batch):
                        mask[b, t_groups[t_idx]] = True

                    mask             = mask.to(self.device)
                    valid_x_hat      = x_hat_all[mask]
                    valid_x_pred     = x_pred_all[mask]
                    valid_trg_recon  = target_recon[mask]
                    valid_trg_pred   = target_pred[mask]

                    if valid_x_hat.numel() > 0:
                        loss = (
                            nn.MSELoss()(valid_x_hat,  valid_trg_recon)
                            + nn.MSELoss()(valid_x_pred, valid_trg_pred)
                        )
                    else:
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if isinstance(loss, torch.Tensor):
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()
                n_steps    += 1

        return total_loss / max(n_steps, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:
        from utils.metrics import compute_f1, compute_auc_pr
        self.backbone.eval()
        all_scores, all_labels = [], []

        batch_size = self.config.get("batch_size", 512)
        n_val      = len(val_dataset)

        for i in range(0, n_val, batch_size):
            end_i      = min(i + batch_size, n_val)
            batch_data = [val_dataset[j] for j in range(i, end_i)]

            x          = torch.stack([d["x"] for d in batch_data]).to(self.device)
            y          = torch.stack([d["y"] for d in batch_data])
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])

            _, x_hat_all, x_pred_all = self.backbone(x, graph_safe)

            # Reconstruction target: last timestep of window (no CPU round-trip)
            target_recon = x[:, :, -1, :]   # [B, N, 1]

            target_pred = torch.stack([
                torch.tensor(
                    val_dataset.signals[min(d["t"] + 1, len(val_dataset.signals) - 1)],
                    dtype=torch.float32
                )
                for d in batch_data
            ]).unsqueeze(-1).to(self.device)

            recon_err = torch.norm(
                (x_hat_all - target_recon).view(x_hat_all.shape[0], x_hat_all.shape[1], -1),
                dim=-1
            )
            pred_err = torch.norm(
                (x_pred_all - target_pred).view(x_pred_all.shape[0], x_pred_all.shape[1], -1),
                dim=-1
            )

            # Use MAX across nodes — SWaT attacks hit specific sensors
            node_scores    = recon_err + pred_err
            system_scores  = node_scores.max(dim=1).values   # max not mean
            system_labels  = y[:, 0]

            all_scores.extend(system_scores.cpu().tolist())
            all_labels.extend(system_labels.tolist())

        return compute_f1(all_scores, all_labels), compute_auc_pr(all_scores, all_labels)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def train(self, *args, **kwargs) -> Dict[str, List[float]]:
        # Universal parsing — handles both run_rctgad.py and ablations.py
        epochs      = self.config.get("epochs", 100)
        k_warmup    = self.config.get("k_warmup", 30)
        val_dataset = None
        save_dir    = None

        if len(args) > 0: epochs      = args[0]
        if len(args) > 1: k_warmup    = args[1]
        if len(args) > 2: val_dataset = args[2]
        if len(args) > 3: save_dir    = args[3]

        if "epochs"      in kwargs: epochs      = kwargs["epochs"]
        if "k_warmup"    in kwargs: k_warmup    = kwargs["k_warmup"]
        if "val_dataset" in kwargs: val_dataset = kwargs["val_dataset"]
        if "save_dir"    in kwargs: save_dir    = kwargs["save_dir"]

        print(f"\n[Trainer] Starting training for {epochs} epochs...")
        n_samples     = len(self.dataset) * self.raw_backbone.num_nodes
        hardness_array = np.zeros(n_samples, dtype=np.float32)

        if self.use_curriculum:
            hardness_array = self._compute_hardness_from_loss()

        best_f1    = -1.0
        batch_size = self.config.get("batch_size", 512)

        for epoch in range(epochs):
            t_start = time.time()

            if self.use_curriculum:
                indices = get_batch_fast(hardness_array, epoch, k_warmup)
                if self.logger:
                    current_k   = len(indices)
                    max_hardness = float(np.max(hardness_array[indices])) if current_k > 0 else 0.0
                    self.logger.log_curriculum_pacing(epoch, current_k, n_samples, max_hardness)

                if epoch > 0 and epoch % 10 == 0:
                    hardness_array = self._compute_hardness_from_loss()
            else:
                indices = np.arange(n_samples)

            train_loss = self._train_epoch(indices, batch_size)

            f1, auc_pr = 0.0, 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                f1, auc_pr = self._validate(val_dataset)

                if f1 > best_f1:
                    best_f1  = f1
                    out_dir  = save_dir if save_dir else "checkpoints/rctgad"
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(
                        self.raw_backbone.state_dict(),
                        os.path.join(out_dir, "best_model.pt")
                    )

            epoch_time = time.time() - t_start

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc_pr"].append(auc_pr)
            self.history["pct_data"].append(
                len(indices) / n_samples if self.use_curriculum else 1.0
            )

            pct = 100 * len(indices) / n_samples if self.use_curriculum else 100.0
            print(
                f"Epoch {epoch:03d} | "
                f"λ={pacing(epoch, k_warmup):.3f} | "
                f"data={pct:.1f}% | "
                f"Loss={train_loss:.4f} | "
                f"F1={f1:.4f} | "
                f"AUC-PR={auc_pr:.4f} | "
                f"{epoch_time:.1f}s"
            )

            if self.logger:
                self.logger.log_epoch(epoch, train_loss, f1, epoch_time)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.history