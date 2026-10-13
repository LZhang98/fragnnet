"""Create a W&B sweep from a local config file.

Example:
    python scripts/create_local_sweep.py \
        --sweep-config config/fragnnet/sweeps/my_sweep.yaml \
        --project my-wandb-project \
        --entity my-wandb-entity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def _load_sweep_config(config_path: Path) -> dict[str, Any]:
    """Load a sweep configuration from YAML or JSON.

    Args:
        config_path: Path to the sweep configuration file.

    Returns:
        Parsed sweep configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file extension is unsupported or config is invalid.
        ImportError: If YAML parsing is requested but PyYAML is unavailable.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML configs. Install with: pip install pyyaml"
            ) from exc

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError(f"Unsupported config format '{suffix}'. Use .json, .yaml, or .yml")

    if not isinstance(config, dict):
        raise ValueError("Sweep config must be a dictionary at the top level")

    if "method" not in config:
        raise ValueError("Sweep config is missing required key: 'method'")

    if "parameters" not in config:
        raise ValueError("Sweep config is missing required key: 'parameters'")

    return config


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for creating a sweep.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Create a W&B sweep from local config")
    parser.add_argument(
        "--sweep-config",
        type=Path,
        required=True,
        help="Path to sweep config file (.yaml/.yml/.json)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="frag-gnn",
        help="W&B project ID/name. Defaults to frag-gnn.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="frag-gnn",
        help="W&B entity Defaults to frag-gnn.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a W&B sweep and log the generated sweep ID."""
    args = _parse_args()

    sweep_config = _load_sweep_config(args.sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    print(f"Created sweep_id: {sweep_id}")


if __name__ == "__main__":
    main()
