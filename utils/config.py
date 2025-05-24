import argparse
import yaml
from typing import Any, Dict

def load_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Connect-N RL")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
