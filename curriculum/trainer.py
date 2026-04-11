"""
trainer.py — Unified Training Loop for RC-TGAD.
Refactored for Unified Processing Unit [B, N, W, 1].
Includes DataParallel Graph-Wrapper Fix to prevent silent deadlocks.
Includes AMP (Mixed Precision) and Single-Threaded GPU optimization for maximum speed.
Includes explicit Garbage Collection to prevent Kaggle RAM spikes.
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
    """
    Tricks nn.DataParallel into NOT slicing the graph in half.
    Since it's a custom object, DataParallel will pass references safely to all GPUs.
    """
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
        logger=None  # 🛡️ IEEE PAPER LOGGER
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

    @torch.no_grad()
    def _compute_hardness_from_loss(self) -> np.ndarray:
        print("\n[Trainer] Computing hardness scores (Single-threaded GPU)...")
        self.backbone.eval()
        ds = self.dataset
        n_timesteps = len(ds)
        N = self.raw_backbone.num_nodes
        batch_size = self.config.get("batch_size", 32) * 2 
        
        all_scores = np.zeros(n_timesteps * N, dtype=np.float32)
        
        from tqdm import tqdm
        pbar = tqdm(total=n_timesteps, desc="Hardness Eval", unit="steps")
        
        for i in range(0, n_timesteps, batch_size):
            end_i = min(i + batch_size, n_timesteps)
            batch_data = [ds[j] for j in range(i, end_i)]
            
            x = torch.stack([d["x"] for d in batch_data]).to(self.device)
            y = torch.stack([d["y"] for d in batch_data])
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            
            # ⚡ Unpack all 3
            z_all, x_hat_all, x_pred_all = self.backbone(x, graph_safe) 
            
            target = torch.stack([
                torch.tensor(ds.signals[d["t"]], dtype=torch.float32) 
                for d in batch_data
            ]).unsqueeze(-1).to(self.device)

            z_all_cpu = z_all.cpu()
            x_hat_all_cpu = x_hat_all.cpu()
            target_cpu = target.cpu()

            B_real = x.shape[0]
            
            for b in range(B_real):
                t = batch_data[b]["t"]
                t_base_idx = (t - ds.window) // getattr(ds, 'stride', 1)
                
                for n in range(N):
                    global_idx = t_base_idx * N + n
                    
                    # RAG scorer still uses reconstruction error for curriculum pacing (stable)
                    h_val = self.rag_scorer.score_hardness(
                        z=z_all_cpu[b, n],
                        x=target_cpu[b, n],
                        x_hat=x_hat_all_cpu[b, n],
                        node_id=n,
                        graph=batch_data[0]["graph"],
                        t=t,
                        ground_truth_label=int(y[b, n])
                    )
                    
                    all_scores[global_idx] = h_val
            
            pbar.update(B_real)
            
        pbar.close()

        score_min, score_max = all_scores.min(), all_scores.max()
        score_range = score_max - score_min
        if score_range < 1e-6:
            all_scores = np.full_like(all_scores, 0.5)
        else:
            all_scores = np.clip((all_scores - score_min) / (score_range + 1e-8), 0, 1)
            
        self.backbone.train()
        return all_scores

    def _train_epoch(self, indices, batch_size: int) -> float:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        
        self.backbone.train()
        total_loss = 0.0
        n_steps = 0
        ds = self.dataset
        N = self.raw_backbone.num_nodes
        
        is_full_dataset = (len(indices) == len(ds) * N)
        
        if is_full_dataset:
            t_indices = list(range(len(ds)))
        else:
            from collections import defaultdict
            t_groups = defaultdict(list)
            for idx in indices:
                t_idx = int(idx // N)
                node_idx = int(idx % N)
                t_groups[t_idx].append(node_idx)
            t_indices = sorted(list(t_groups.keys()))
        
        for i in range(0, len(t_indices), batch_size):
            end_i = min(i + batch_size, len(t_indices))
            current_t_batch = t_indices[i:end_i]
            
            batch_data = [ds[t_idx] for t_idx in current_t_batch]
            x = torch.stack([d["x"] for d in batch_data]).to(self.device)
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            
            self.optimizer.zero_grad()
            
            with autocast():
                # ⚡ Unpack all 3
                z_all, x_hat_all, x_pred_all = self.backbone(x, graph_safe)
                
                # Target for t
                target_recon = torch.stack([
                    torch.tensor(ds.signals[d["t"]], dtype=torch.float32) 
                    for d in batch_data
                ]).unsqueeze(-1).to(self.device)
                
                # Target for t+1 (Prediction)
                target_pred = torch.stack([
                    torch.tensor(ds.signals[min(d["t"] + 1, len(ds.signals)-1)], dtype=torch.float32) 
                    for d in batch_data
                ]).unsqueeze(-1).to(self.device)
                
                if is_full_dataset:
                    loss = nn.MSELoss()(x_hat_all, target_recon) + nn.MSELoss()(x_pred_all, target_pred)
                else:
                    B_curr = len(current_t_batch)
                    mask = torch.zeros((B_curr, N), dtype=torch.bool)
                    for b, t_idx in enumerate(current_t_batch):
                        mask[b, t_groups[t_idx]] = True
                        
                    mask = mask.to(self.device)
                    valid_x_hat = x_hat_all[mask]
                    valid_x_pred = x_pred_all[mask]
                    
                    valid_target_recon = target_recon[mask]
                    valid_target_pred = target_pred[mask]
                    
                    if valid_x_hat.numel() > 0:
                        # ⚡ THE DUAL LOSS
                        loss = nn.MSELoss()(valid_x_hat, valid_target_recon) + nn.MSELoss()(valid_x_pred, valid_target_pred)
                    else:
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            if isinstance(loss, torch.Tensor):
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_steps += 1

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:
        from utils.metrics import compute_f1, compute_auc_pr
        self.backbone.eval()
        all_scores, all_labels = [], []
        
        batch_size = self.config.get("batch_size", 32)
        n_val = len(val_dataset)
        
        for i in range(0, n_val, batch_size):
            end_i = min(i + batch_size, n_val)
            batch_data = [val_dataset[j] for j in range(i, end_i)]
            
            x = torch.stack([d["x"] for d in batch_data]).to(self.device) 
            y = torch.stack([d["y"] for d in batch_data]) 
            
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            
            # ⚡ Unpack all 3
            _, x_hat_all, x_pred_all = self.backbone(x, graph_safe)
            
            target_recon = torch.stack([
                torch.tensor(val_dataset.signals[d["t"]], dtype=torch.float32) 
                for d in batch_data
            ]).unsqueeze(-1).to(self.device)

            target_pred = torch.stack([
                torch.tensor(val_dataset.signals[min(d["t"] + 1, len(val_dataset.signals)-1)], dtype=torch.float32) 
                for d in batch_data
            ]).unsqueeze(-1).to(self.device)
            
            # ⚡ COMBINE ERRORS FOR MASSIVE ANOMALY SIGNAL BOOST
            recon_err = torch.norm(x_hat_all.view(x_hat_all.shape[0], x_hat_all.shape[1], -1) - target_recon.view(target_recon.shape[0], target_recon.shape[1], -1), dim=-1)
            pred_err = torch.norm(x_pred_all.view(x_pred_all.shape[0], x_pred_all.shape[1], -1) - target_pred.view(target_pred.shape[0], target_pred.shape[1], -1), dim=-1)
            
            node_scores = recon_err + pred_err
            system_scores = node_scores.mean(dim=1)
            system_labels = y[:, 0]
            
            all_scores.extend(system_scores.cpu().tolist())
            all_labels.extend(system_labels.tolist())
            
        return compute_f1(all_scores, all_labels), compute_auc_pr(all_scores, all_labels)

    def train(self, epochs: int, k_warmup: int, val_dataset=None, save_dir: Optional[str] = None, **kwargs) -> Dict[str, List[float]]:
        print(f"\n[Trainer] Starting training for {epochs} epochs...")
        n_samples = len(self.dataset) * self.raw_backbone.num_nodes
        hardness_array = np.zeros(n_samples, dtype=np.float32)

        # Initial Hardness Calculation for Epoch 0
        if self.use_curriculum:
            hardness_array = self._compute_hardness_from_loss()

        best_f1 = -1.0
        batch_size = self.config.get("batch_size", 512)

        for epoch in range(epochs):
            t_start = time.time()

            # 1. Curriculum Data Selection
            if self.use_curriculum:
                indices = get_batch_fast(hardness_array, epoch, k_warmup)
                
                # IEEE Logger: Record pacing details
                if self.logger:
                    current_k = len(indices)
                    max_hardness = float(np.max(hardness_array[indices])) if current_k > 0 else 0.0
                    self.logger.log_curriculum_pacing(epoch, current_k, n_samples, max_hardness)

                # Recompute Hardness every 10 epochs as the model gets smarter
                if epoch > 0 and epoch % 10 == 0:
                    hardness_array = self._compute_hardness_from_loss()
            else:
                indices = np.arange(n_samples)

            # 2. Run the Dual-Loss Epoch
            train_loss = self._train_epoch(indices, batch_size)
            
            # 3. Validation
            f1, auc_pr = 0.0, 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                f1, auc_pr = self._validate(val_dataset)
                
                # ⚡ BULLETPROOF SAVING LOGIC
                if f1 > best_f1:
                    best_f1 = f1
                    out_dir = save_dir if save_dir else "checkpoints/rctgad"
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(self.raw_backbone.state_dict(), os.path.join(out_dir, "best_model.pt"))

            epoch_time = time.time() - t_start
            
            # 4. Update History
            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc_pr"].append(auc_pr)
            
            if self.use_curriculum:
                self.history["pct_data"].append(len(indices) / n_samples)
            else:
                self.history["pct_data"].append(1.0)
                
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | F1: {f1:.4f} | Time: {epoch_time:.1f}s")

            # IEEE Logger
            if self.logger:
                self.logger.log_epoch(epoch, train_loss, f1, epoch_time)

            # 5. Garbage Collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.history