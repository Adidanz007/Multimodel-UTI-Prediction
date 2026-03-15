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
    """Load YAML config into a python dictionary."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
