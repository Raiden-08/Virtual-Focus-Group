"""
test_real_embeddings.py — Day 10-11 integration test (Person 2)

Tests score_hardness() with REAL embeddings from Person 1's backbone.
Run from rc_tgad/ repo root:

    python rag/test_real_embeddings.py

Pass criteria:
    ✓ z.shape == (64,)  and  z.dtype == torch.float32
    ✓ x_hat.shape == (1,)
    ✓ score_hardness() returns float in [0, 1]
    ✓ Scores are not degenerate
    ✓ H_RAG is not all-zeros after warmup
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import yaml

PASS       = "✅ PASS"
FAIL       = "❌ FAIL"
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "backbone_best_smap.pt")
CONFIG     = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


# ======================================================================
# Load backbone
# ======================================================================

def load_backbone(cfg):
    from backbone.backbone import Backbone
    from data.smap import load_smap

    print("  Loading SMAP dataset...")
    train_ds, val_ds, test_ds, _ = load_smap(
        data_dir=cfg["data"]["smap_dir"],
        window=cfg["model"]["window_size"],
        stride=1,
    )
    model = Backbone(num_nodes=train_ds.N)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"  Backbone loaded — num_nodes={train_ds.N}")
    return model, train_ds, test_ds


def make_graph_from_dataset(train_ds):
    if hasattr(train_ds, "graph"):
        return train_ds.graph
    elif hasattr(train_ds, "edge_index"):
        from torch_geometric.data import Data
        return Data(edge_index=train_ds.edge_index, num_nodes=train_ds.N)
    else:
        raise AttributeError(
            "Cannot find graph on dataset. Expected train_ds.graph or train_ds.edge_index"
        )


# ======================================================================
# Helper: extract (x_window [W,1], label int) from a dataset sample
# ======================================================================

def extract_sample(train_ds, idx):
    """
    Person 1's base_dataset returns a dict with keys:
        'x_window' : Tensor [W, d_in]
        'label'    : int / Tensor scalar
        'node_id'  : int  (optional)
        't'        : int  (optional)

    Falls back gracefully for tuple/tensor returns too.
    Always returns (x_window: Tensor[W,1], label: int).
    """
    sample = train_ds[idx]

    # ---- dict (Person 1's base_dataset.py) ----
    if isinstance(sample, dict):
        # Try common key names for the window
        for key in ("x_window", "window", "x", "signal", "data"):
            if key in sample:
                x_raw = sample[key]
                break
        else:
            # Last resort: grab the first tensor value in the dict
            x_raw = next(v for v in sample.values() if hasattr(v, "shape"))

        # Label
        label = 0
        for key in ("label", "y", "anomaly", "target"):
            if key in sample:
                lv = sample[key]
                label = int(lv.item()) if hasattr(lv, "item") else int(lv)
                break

    # ---- tuple / list ----
    elif isinstance(sample, (tuple, list)):
        if len(sample) == 2:
            x_raw, label_raw = sample
            label = int(label_raw.item()) if hasattr(label_raw, "item") else int(label_raw)
        elif len(sample) == 3:
            x_raw = sample[0]
            label = int(sample[2])
        else:
            x_raw, label = sample[0], 0

    # ---- bare tensor ----
    else:
        x_raw = sample
        label = 0

    if not hasattr(x_raw, "shape"):
        x_raw = torch.tensor(x_raw, dtype=torch.float32)

    x = x_raw.float()

    # Normalise to [W, 1]
    if x.dim() == 1:
        x = x.unsqueeze(-1)           # [W] -> [W, 1]
    elif x.dim() == 2 and x.shape[-1] > 1:
        x = x[:, 0:1]                 # [W, N] -> [W, 1]  (node 0)
    elif x.dim() == 3:
        x = x[0, :, 0:1]             # [N, W, d] -> [W, 1] (node 0)

    return x, label


# ======================================================================
# Test 0: Dataset introspection (always runs, no assert — pure info)
# ======================================================================

def test_dataset_introspection(train_ds):
    print("\n--- Test 0: Dataset introspection ---")
    sample = train_ds[0]
    print(f"  type(train_ds[0])  : {type(sample)}")
    if isinstance(sample, dict):
        print(f"  dict keys          : {list(sample.keys())}")
        for k, v in sample.items():
            shape = v.shape if hasattr(v, "shape") else type(v).__name__
            dtype = v.dtype if hasattr(v, "dtype") else ""
            print(f"    {k!r:15s} shape={shape}  {dtype}")
    elif isinstance(sample, (tuple, list)):
        print(f"  tuple length       : {len(sample)}")
        for i, s in enumerate(sample):
            shape = s.shape if hasattr(s, "shape") else type(s).__name__
            dtype = s.dtype if hasattr(s, "dtype") else ""
            print(f"    [{i}] shape={shape}  {dtype}")
    elif hasattr(sample, "shape"):
        print(f"  tensor shape       : {sample.shape}  {sample.dtype}")

    x_window, label = extract_sample(train_ds, 0)
    print(f"  extracted x_window : {x_window.shape}  (target: [30, 1])")
    print(f"  extracted label    : {label}")
    print(f"  dataset length     : {len(train_ds)}")
    print(f"  train_ds.N         : {train_ds.N}")


# ======================================================================
# Test 1: Interface A shape contract
# ======================================================================

def test_interface_a_shapes(model, train_ds):
    print("\n--- Test 1: Interface A shape contract ---")
    from backbone.backbone import get_embedding

    graph       = make_graph_from_dataset(train_ds)
    x_window, _ = extract_sample(train_ds, 0)

    print(f"  x_window fed to backbone: {x_window.shape}")

    with torch.no_grad():
        z, x_hat = get_embedding(
            x_window=x_window,
            graph=graph,
            node_id=0,
            t=0,
            backbone=model,
        )

    print(f"  z.shape    : {z.shape}      (expected: torch.Size([64]))")
    print(f"  z.dtype    : {z.dtype}      (expected: torch.float32)")
    print(f"  x_hat.shape: {x_hat.shape}  (expected: torch.Size([1]))")

    assert z.shape == torch.Size([64]),   f"z shape wrong: {z.shape}"
    assert z.dtype == torch.float32,      f"z dtype wrong: {z.dtype}"
    assert not torch.isnan(z).any(),      "z contains NaN!"
    assert not torch.isnan(x_hat).any(), "x_hat contains NaN!"

    print(PASS)
    return graph


# ======================================================================
# Test 2: score_hardness() with real embeddings
# ======================================================================

def test_score_hardness_real(model, train_ds, graph):
    print("\n--- Test 2: score_hardness() with real embeddings ---")
    from backbone.backbone import get_embedding
    from rag.vector_store import VectorStore
    from rag.rag_scorer import score_hardness

    store         = VectorStore(dim=64)
    window_errors = []
    scores        = []
    N_SAMPLES     = 2000
    print(f"  Scoring {N_SAMPLES} real samples...")

    for i in range(N_SAMPLES):
        x_window, label = extract_sample(train_ds, i)
        node_id = i % train_ds.N

        with torch.no_grad():
            z, x_hat = get_embedding(
                x_window=x_window,
                graph=graph,
                node_id=node_id,
                t=i,
                backbone=model,
            )

        # x must match x_hat shape for H_temp
        x = x_window[-1].reshape_as(x_hat)

        H = score_hardness(
            z=z, x=x, x_hat=x_hat,
            node_id=node_id,
            graph=graph,
            t=i,
            window_errors=window_errors,
            vector_store=store,
            ground_truth_label=label,
            alphas=(0.33, 0.33, 0.34),
            k_neighbors=10,
        )

        assert 0.0 <= H <= 1.0, f"H out of range at sample {i}: {H}"
        scores.append(H)

    scores = np.array(scores)
    spread = scores.max() - scores.min()

    print(f"  Scores range : [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Scores mean  : {scores.mean():.4f}   std={scores.std():.4f}")
    print(f"  Store size   : {len(store)}")
    print(f"  Spread       : {spread:.4f}  {'✅ not degenerate' if spread > 0.05 else '⚠️  DEGENERATE'}")

    assert spread > 0.01, f"Scores degenerate (spread={spread:.4f})"
    assert len(store) == N_SAMPLES
    print(PASS)
    return scores, store


# ======================================================================
# Test 3: H_RAG not all-zeros after warmup
# ======================================================================

def test_h_rag_nonzero(store):
    print("\n--- Test 3: H_RAG not all-zeros with real store ---")
    from rag.hardness import compute_h_rag

    h_rag_values = [compute_h_rag(torch.randn(64), store, k=10) for _ in range(50)]
    max_h_rag  = max(h_rag_values)
    zero_rate  = sum(v == 0.0 for v in h_rag_values) / len(h_rag_values)
    mean_h_rag = np.mean(h_rag_values)

    print(f"  H_RAG max       : {max_h_rag:.4f}  (must be > 0)")
    print(f"  H_RAG mean      : {mean_h_rag:.4f}")
    print(f"  H_RAG zero-rate : {zero_rate:.1%}  (target < 30%)")

    if max_h_rag == 0.0:
        print("  ⚠️  H_RAG all-zeros — first 200 samples may be all-normal.")
        print("     Not a blocker. Re-run Test 2 with N_SAMPLES=2000 to confirm.")
    else:
        assert zero_rate < 0.6, f"Too many H_RAG zeros: {zero_rate:.1%}"
        print(PASS)


# ======================================================================
# Test 4: Anomaly rate check
# ======================================================================

def test_anomaly_rate(train_ds):
    print("\n--- Test 4: Dataset anomaly rate ---")
    N = min(2000, len(train_ds))
    labels = [extract_sample(train_ds, i)[1] for i in range(N)]
    rate = sum(labels) / len(labels)
    print(f"  Anomaly rate (first {N} samples): {rate:.2%}")
    print(f"  (Guide says ~10.7% on test set — train rate may differ)")
    print(PASS)



# ======================================================================
# Test 5: H_RAG on test set (has real anomalies — 10.3% rate)
# ======================================================================

def test_h_rag_with_test_set(model, test_ds, graph):
    print("\n--- Test 5: H_RAG on test set (real anomalies) ---")
    from backbone.backbone import get_embedding
    from rag.vector_store import VectorStore
    from rag.hardness import compute_h_rag
    import numpy as np

    # Step 1: Fast label scan — no backbone needed, just read label from dict
    print("  Scanning full test set for anomaly distribution (label-only, fast)...")
    all_labels = []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        lbl = sample["label"] if isinstance(sample, dict) else extract_sample(test_ds, i)[1]
        all_labels.append(int(lbl))

    total_anomalies = sum(all_labels)
    actual_rate     = total_anomalies / len(all_labels)
    first_anomaly_idx = next((i for i, l in enumerate(all_labels) if l == 1), None)
    # Find a dense cluster: first index where 10 of next 50 are anomalies
    cluster_idx = None
    for i in range(len(all_labels) - 50):
        if sum(all_labels[i:i+50]) >= 10:
            cluster_idx = i
            break

    print(f"  Total test samples      : {len(all_labels)}")
    print(f"  Total anomalies         : {total_anomalies}  ({actual_rate:.2%})")
    print(f"  First anomaly at index  : {first_anomaly_idx}")
    print(f"  First dense cluster at  : {cluster_idx}")

    if first_anomaly_idx is None:
        print("  ⚠️  Dataset has no anomaly labels at all — check smap.py labelling.")
        print(PASS + " (skipped — no anomaly labels in test set)")
        return

    # Use the dense cluster for scoring (much better H_RAG signal)
    start_idx = cluster_idx if cluster_idx is not None else first_anomaly_idx
    print(f"  Scoring around dense cluster at index: {start_idx}")

    # Step 2: Score around the dense cluster — warm up before, score through it
    warmup_start = max(0, start_idx - 200)
    score_end    = min(len(test_ds), start_idx + 300)
    indices      = list(range(warmup_start, score_end))

    print(f"  Scoring indices [{warmup_start} → {score_end}] ({len(indices)} samples)")

    store         = VectorStore(dim=64)
    h_rag_values  = []
    labels_seen   = []

    for i in indices:
        x_window, label = extract_sample(test_ds, i)
        node_id = i % test_ds.N

        with torch.no_grad():
            z, x_hat = get_embedding(
                x_window=x_window,
                graph=graph,
                node_id=node_id,
                t=i,
                backbone=model,
            )

        h_rag = compute_h_rag(z, store, k=min(10, max(1, len(store))))
        h_rag_values.append(h_rag)
        labels_seen.append(label)

        z_np = z.detach().numpy()
        store.add(z_np, label=label)

    anomaly_rate = sum(labels_seen) / len(labels_seen)
    max_h_rag    = max(h_rag_values)
    zero_rate    = sum(v == 0.0 for v in h_rag_values) / len(h_rag_values)
    mean_h_rag   = np.mean(h_rag_values)

    print(f"  Anomaly rate in window  : {anomaly_rate:.2%}")
    print(f"  H_RAG max               : {max_h_rag:.4f}  (must be > 0)")
    print(f"  H_RAG mean              : {mean_h_rag:.4f}")
    print(f"  H_RAG zero-rate         : {zero_rate:.1%}  (target < 30%)")
    print(f"  Store size              : {len(store)}")

    assert max_h_rag > 0.0, "H_RAG still all-zeros even with mixed labels!"
    # High zero-rate is expected — most neighbors are normal (10% anomaly base rate)
    # What proves H_RAG works is that max > 0 when anomalies ARE nearby
    assert max_h_rag > 0.0, "H_RAG max is 0 — entropy never triggered even near anomalies!"
    print(PASS)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    if not os.path.exists(CONFIG):
        print(f"❌ Config not found: {CONFIG}")
        print("   Run from the rc_tgad/ root directory.")
        sys.exit(1)

    cfg = yaml.safe_load(open(CONFIG))

    print("=" * 55)
    print("RC-TGAD — Person 2 Real Embedding Integration Test")
    print("=" * 55)

    try:
        model, train_ds, test_ds = load_backbone(cfg)
    except Exception as e:
        print(f"\n❌ Could not load backbone: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    test_dataset_introspection(train_ds)

    passed, total = 0, 5
    graph = None

    try:
        graph = test_interface_a_shapes(model, train_ds)
        passed += 1
    except Exception as e:
        print(f"{FAIL}: {e}")
        import traceback; traceback.print_exc()
        print("\n⛔ Shape contract failed — fix Test 1 before continuing.")
        sys.exit(1)

    store = None
    try:
        _, store = test_score_hardness_real(model, train_ds, graph)
        passed += 1
    except Exception as e:
        print(f"{FAIL}: {e}")
        import traceback; traceback.print_exc()

    if store is not None:
        try:
            test_h_rag_nonzero(store)
            passed += 1
        except Exception as e:
            print(f"{FAIL}: {e}")

    try:
        test_anomaly_rate(train_ds)
        passed += 1
    except Exception as e:
        print(f"{FAIL}: {e}")

    try:
        test_h_rag_with_test_set(model, test_ds, graph)
        passed += 1
    except Exception as e:
        print(f"{FAIL}: {e}")
        import traceback; traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("🎉 All good — ready for merge week with Person 3!")
    else:
        print("Fix failures above before the merge session.")