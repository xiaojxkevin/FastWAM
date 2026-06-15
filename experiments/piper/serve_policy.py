#!/usr/bin/env python
"""FastWAM synchronous inference server — entry point.

Usage:
    python experiments/piper/serve_policy.py --config experiments/piper/serve_cloth_folding.yaml
    bash experiments/piper/serve_policy.sh
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure the FastWAM project root is on sys.path so that `fastwam.*` imports work.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.piper.fastwam_server import FastWAMModelWrapper, FastWAMPolicyServer

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.  Leaf values in *override* win."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_config(config_path: str) -> dict:
    """Load YAML config with optional ``base_config`` inheritance.

    When ``base_config`` points to a training-run ``config.yaml`` (e.g.
    ``pretrained/config.yaml``), the resolved ``model`` block is reused
    verbatim and ``data.train.*`` fields are mapped to the flat ``data.*``
    structure the server expects.  The serve config only needs to specify
    deployment overrides (host, port, checkpoint, acceleration, …).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = cfg.pop("base_config", None)
    if base_path is None:
        return cfg

    # Resolve base config path (relative to serve config, then relative to project root).
    bp = Path(base_path)
    if not bp.is_absolute():
        candidate = path.parent / bp
        bp = candidate if candidate.exists() else _PROJECT_ROOT / bp
    if not bp.exists():
        raise FileNotFoundError(f"base_config not found: {base_path} (tried {bp})")
    with open(bp, "r") as f:
        base_cfg = yaml.safe_load(f)

    # ---- model: reuse training model block, then apply serve overrides -----
    base_model = base_cfg.get("model", {})
    serve_model = cfg.pop("model", {})
    cfg["model"] = _deep_merge(base_model, serve_model)

    # ---- data: map training data.train.* → flat data.* structure ----------
    train_data = base_cfg.get("data", {}).get("train", {})
    processor_cfg = train_data.get("processor", {})

    mapped_data: dict = {
        "shape_meta": train_data.get("shape_meta", {}),
        "num_obs_steps": train_data.get("num_frames", processor_cfg.get("num_obs_steps", 33)),
        "num_output_cameras": processor_cfg.get("num_output_cameras", 3),
        "action_output_dim": processor_cfg.get("action_output_dim", 14),
        "proprio_output_dim": processor_cfg.get("proprio_output_dim", 14),
        "norm_default_mode": processor_cfg.get("norm_default_mode", "z-score"),
        "text_embedding_cache_dir": train_data.get("text_embedding_cache_dir", ""),
        "dataset_stats_path": train_data.get("pretrained_norm_stats", ""),
    }
    # Apply serve-level data overrides (e.g. absolute paths).
    serve_data = cfg.pop("data", {})
    cfg["data"] = _deep_merge(mapped_data, serve_data)

    logger.info(
        "Loaded base config from %s | model overrides=%s",
        bp,
        sorted(serve_model.keys()),
    )
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="FastWAM websocket inference server")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/piper/serve_cloth_folding.yaml",
        help="Path to server YAML config (default: experiments/piper/serve_cloth_folding.yaml)",
    )
    args = parser.parse_args()

    # Resolve config path relative to project root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path

    logger.info("Loading config from: %s", config_path)
    cfg = _load_config(str(config_path))

    host = str(cfg.get("host", "0.0.0.0"))
    port = int(cfg.get("port", 8765))

    metadata = {
        "action_dim": cfg["data"]["action_output_dim"],
        "action_horizon": cfg["inference"]["action_horizon"],
        "rtc_supported": True,
        "rtc_mode": "training_time_prefix",
    }

    logger.info("Initialising FastWAM model wrapper …")
    wrapper = FastWAMModelWrapper(cfg)

    server = FastWAMPolicyServer(
        wrapper=wrapper,
        host=host,
        port=port,
        metadata=metadata,
    )
    try:
        server.serve_forever()
    finally:
        wrapper.shutdown()


if __name__ == "__main__":
    main()
