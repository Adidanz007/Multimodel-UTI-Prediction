"""Shared utility helpers for the multimodal UTI project."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a consistent log format for all scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow is optional in some scripts.
        pass


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML config and resolve relative file paths to absolute paths."""
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # When config is in <project_root>/config/config.yaml, resolve paths from project root.
    if config_file.parent.name.lower() == "config":
        base_dir = config_file.parent.parent
    else:
        base_dir = config_file.parent

    def _resolve(value: Any) -> Any:
        if isinstance(value, str):
            # Keep env-style placeholders untouched.
            if value.startswith("${") and value.endswith("}"):
                return value
            path_obj = Path(value)
            if path_obj.is_absolute():
                return str(path_obj)
            return str((base_dir / path_obj).resolve())
        return value

    if "paths" in config and isinstance(config["paths"], dict):
        for key, value in list(config["paths"].items()):
            config["paths"][key] = _resolve(value)

    for section, key in [
        ("clinical", "model_output"),
        ("ultrasound", "model_output"),
        ("fusion", "model_output"),
    ]:
        if section in config and isinstance(config[section], dict) and key in config[section]:
            config[section][key] = _resolve(config[section][key])

    return config


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def start_run_timer() -> float:
    """Return start timestamp for duration tracking."""
    return time.time()


def end_run_timer(start_time: float) -> float:
    """Compute elapsed duration in seconds."""
    return time.time() - start_time


def save_json(payload: Dict[str, Any], output_path: str) -> None:
    """Save dictionary as json."""
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
