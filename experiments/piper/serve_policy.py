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


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    }

    logger.info("Initialising FastWAM model wrapper …")
    wrapper = FastWAMModelWrapper(cfg)

    server = FastWAMPolicyServer(
        wrapper=wrapper,
        host=host,
        port=port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
