# configs/config_loader.py
# Loads default.yaml and applies any command-line overrides.
# Usage:
#   cfg = load_config("configs/default.yaml")
#   cfg = load_config("configs/default.yaml", overrides=["curriculum.k_warmup=20", "training.lr=0.0005"])

import yaml
import copy
import argparse
from typing import Dict, List, Optional


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict:
    """
    Load config from yaml file and apply any dot-notation overrides.

    Args:
        path      : path to yaml config file
        overrides : list of "section.key=value" strings (from CLI or ablation scripts)

    Returns:
        Flat-ish dict — access as cfg["curriculum"]["k_warmup"] or cfg.get(...)
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            _apply_override(cfg, override)

    _validate(cfg)
    return cfg


def _apply_override(cfg: Dict, override: str):
    """Apply a single 'section.key=value' override string."""
    assert "=" in override, f"Override must be 'key=value', got: {override}"
    key_path, value_str = override.split("=", 1)
    keys  = key_path.strip().split(".")
    value = _cast(value_str.strip())

    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    print(f"[Config] Override: {key_path} = {value}")


def _cast(value: str):
    """Cast string to int, float, bool, or None as appropriate."""
    if value.lower() == "null" or value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value  # keep as string


def _validate(cfg: Dict):
    """Basic sanity checks — catch dimension mismatches early."""
    assert cfg["model"]["lstm_hidden"] == cfg["rag"]["vector_dim"], \
        f"model.lstm_hidden ({cfg['model']['lstm_hidden']}) must equal rag.vector_dim ({cfg['rag']['vector_dim']})"

    assert cfg["model"]["gnn_out_dim"] == cfg["rag"]["vector_dim"], \
        f"model.gnn_out_dim ({cfg['model']['gnn_out_dim']}) must equal rag.vector_dim ({cfg['rag']['vector_dim']})"

    alpha_sum = cfg["rag"]["alpha_1"] + cfg["rag"]["alpha_2"] + cfg["rag"]["alpha_3"]
    assert abs(alpha_sum - 1.0) < 1e-4, \
        f"rag alphas must sum to 1.0, got {alpha_sum:.4f}"

    assert cfg["training"]["seed"] is not None, "training.seed must be set for reproducibility"


def get_ablation_configs(base_config_path: str) -> Dict[str, Dict]:
    """
    Returns all 6 ablation configs as a dict.
    Person 3 calls this in experiments/ablations.py.

    Returns:
        dict mapping variant_name -> config dict
    """
    ablations = {
        "No Curriculum (Baseline)": [
            "curriculum.enabled=false"
        ],
        "Random Curriculum": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.33",
            "rag.alpha_2=0.33",
            "rag.alpha_3=0.34",
            # Note: random curriculum is implemented in trainer by ignoring scores
        ],
        "H_temp only": [
            "curriculum.enabled=true",
            "rag.alpha_1=1.0",
            "rag.alpha_2=0.0",
            "rag.alpha_3=0.0",
        ],
        "H_struct only": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.0",
            "rag.alpha_2=1.0",
            "rag.alpha_3=0.0",
        ],
        "H_temp + H_struct (no RAG)": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.5",
            "rag.alpha_2=0.5",
            "rag.alpha_3=0.0",
        ],
        "Full RC-TGAD": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.33",
            "rag.alpha_2=0.33",
            "rag.alpha_3=0.34",
        ],
    }

    configs = {}
    for name, overrides in ablations.items():
        configs[name] = load_config(base_config_path, overrides)
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser — used by experiment scripts
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Standard CLI parser for experiment scripts.
    Usage:
        python experiments/run_rctgad.py --config configs/default.yaml
        python experiments/run_rctgad.py --config configs/default.yaml --override curriculum.k_warmup=20 training.lr=0.0005
    """
    parser = argparse.ArgumentParser(description="RC-TGAD Experiment Runner")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to yaml config file"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values: section.key=value"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("configs", exist_ok=True)

    print("=" * 55)
    print("CONFIG LOADER TEST")
    print("=" * 55)

    cfg = load_config("configs/default.yaml")
    print(f"\nLoaded config successfully.")
    print(f"  model.lstm_hidden : {cfg['model']['lstm_hidden']}")
    print(f"  model.window_size : {cfg['model']['window_size']}")
    print(f"  rag.k_neighbors   : {cfg['rag']['k_neighbors']}")
    print(f"  rag.alphas        : {cfg['rag']['alpha_1']}, {cfg['rag']['alpha_2']}, {cfg['rag']['alpha_3']}")
    print(f"  curriculum.enabled: {cfg['curriculum']['enabled']}")
    print(f"  curriculum.k_warmup: {cfg['curriculum']['k_warmup']}")
    print(f"  training.epochs   : {cfg['training']['epochs']}")
    print(f"  data.dataset      : {cfg['data']['dataset']}")

    print("\n--- Testing CLI overrides ---")
    cfg2 = load_config("configs/default.yaml", overrides=[
        "curriculum.k_warmup=20",
        "training.lr=0.0005",
        "rag.alpha_1=1.0",
        "rag.alpha_2=0.0",
        "rag.alpha_3=0.0",
        "curriculum.enabled=false"
    ])
    assert cfg2["curriculum"]["k_warmup"] == 20
    assert cfg2["training"]["lr"] == 0.0005
    assert cfg2["curriculum"]["enabled"] == False
    print("All overrides applied correctly.")

    print("\n--- Testing ablation config generator ---")
    ablation_cfgs = get_ablation_configs("configs/default.yaml")
    for name, c in ablation_cfgs.items():
        enabled = c["curriculum"]["enabled"]
        a1, a2, a3 = c["rag"]["alpha_1"], c["rag"]["alpha_2"], c["rag"]["alpha_3"]
        print(f"  {name:<35} enabled={str(enabled):<5} alphas=({a1:.2f}, {a2:.2f}, {a3:.2f})")

    print("\nAll config tests passed.")
