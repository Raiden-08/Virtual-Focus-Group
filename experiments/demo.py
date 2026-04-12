"""
experiments/demo.py — Live Replay Demo for RC-TGAD
"""

import time
import torch
import warnings
import numpy as np

from data.swat import load_swat
from backbone.backbone import load_backbone, get_embedding
from rag.vector_store import VectorStore
from rag.rag_scorer import score_hardness
from rag.explainer import generate_incident_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PRESENTATION CONFIGURATION
# ─────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/rctgad/best_model.pt"
VECTOR_STORE_PATH = "checkpoints/rctgad/faiss_index"

ANOMALY_THRESHOLD = 2.5   # Adjust based on your model's normal MSE
SLEEP_TIME = 0.005        # Make the terminal output go fast (matrix-style)
FAST_FORWARD_INDEX = 500  # Skip the first N timesteps to hit an attack faster
K_NEIGHBORS = 3

def run_live_demo():
    print("\n[INIT] Booting RC-TGAD Demo System...")
    print("[INIT] Loading SWaT Dataset...")
    train_ds, val_ds, test_ds, feature_cols = load_swat(data_dir="data/raw", window=30)
    
    print(f"[INIT] Loading Neural Backbone from {CHECKPOINT_PATH}...")
    try:
        from backbone.backbone import Backbone
        # 1. Load the dictionary Person 3 saved
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
        # 2. Initialize a blank backbone
        backbone = Backbone(d_in=1, hidden_size=64, gnn_out_dim=64, num_nodes=51)
        
        # 3. Inject ONLY the model weights
        backbone.load_state_dict(checkpoint["model_state"])
        backbone.eval()
    except Exception as e:
        print(f"[ERROR] Could not load backbone: {e}")
        return

    print("[INIT] Loading FAISS RAG Vector Store...")
    vector_store = VectorStore(dim=64)
    try:
        vector_store.load(VECTOR_STORE_PATH)
        print(f"[INIT] Vector Store loaded with {len(vector_store)} historical signatures.")
    except:
        print("[WARNING] Could not load FAISS index from disk. Starting with empty store.")

    graph = test_ds.graph
    window_errors = []

    print("\n=======================================================")
    print("      RC-TGAD LIVE MONITORING INTERFACE ONLINE         ")
    print("=======================================================\n")
    time.sleep(1)

    for i, sample in enumerate(test_ds):
        
        # FAST FORWARD TO THE ACTION FOR THE DEMO
        if i < FAST_FORWARD_INDEX:
            continue
            
        x_window = sample["x_window"]
        label = sample["label"]
        node_id = sample["node_id"]
        t = sample["t"]
        
        target_pred = torch.tensor(test_ds.signals[t], dtype=torch.float32)
        target_recon = x_window[-1]
            
        # Fix: Dual Head embedding return 
        z, x_recon, x_pred = get_embedding(x_window, graph, node_id, t, backbone=backbone)
        
        # Calculate Dual Error
        recon_e = torch.norm(target_recon - x_recon, p=2).item()
        pred_e = torch.norm(target_pred - x_pred, p=2).item()
        error = recon_e + pred_e
        
        # --- NORMAL OPERATION ---
        if error < ANOMALY_THRESHOLD:
            print(f"[t={t:06d}] Sensor {node_id:02d} | Status: NORMAL   | MSE: {error:.4f}", end="\r")
            time.sleep(SLEEP_TIME)
            continue
            
        # --- ANOMALY DETECTED ---
        print(f"\n\n>> [t={t:06d}] ⚠️ ANOMALY DETECTED ON SENSOR {node_id:02d} | MSE: {error:.4f}")
        print(">> Triggering RC-TGAD Hardness Scorer...")
        
        # Calculate Hardness
        h_score = score_hardness(
            z=z, 
            x_recon_target=target_recon, 
            x_recon=x_recon, 
            x_pred_target=target_pred, 
            x_pred=x_pred, 
            node_id=node_id, 
            graph=graph, 
            t=t,
            window_errors=window_errors, 
            vector_store=vector_store,
            ground_truth_label=label, 
            k_neighbors=K_NEIGHBORS
        )
        
        # Retrieve context for LLM
        print(">> Retrieving historical context from FAISS Vector Store...")
        z_np = z.numpy() if hasattr(z, "numpy") else np.asarray(z)
        neighbors = vector_store.query(z_np, k=K_NEIGHBORS)
        
        # Generate LLM Report
        print(">> Handing off to Groq GenAI Explainer Agent...\n")
        
        # Using approximations for sub-components for the demo display
        report = generate_incident_report(
            node_id=node_id,
            reconstruction_error=error,
            h_score=h_score,
            h_temp=h_score * 0.33,   # Approximation for demo prompt
            h_struct=h_score * 0.33, # Approximation for demo prompt
            h_rag=h_score * 0.34,    # Approximation for demo prompt
            neighbors=neighbors
        )
        
        print("-------------------------------------------------------")
        print("                 INCIDENT REPORT                       ")
        print("-------------------------------------------------------")
        print(report)
        print("-------------------------------------------------------\n")
        
        # THIS IS WHERE IT PAUSES. 
        # Change this to `break` if you want the script to end completely here.
        input("Press Enter to acknowledge and resume monitoring...")
        print("Resuming feed...\n")

if __name__ == "__main__":
    run_live_demo()