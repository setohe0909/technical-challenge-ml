from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from typing import Any, Iterable, List

import numpy as np

from . import config


def set_global_seed(random_state: int) -> None:
    """Set seeds for Python and NumPy for reproducibility."""
    random.seed(random_state)
    np.random.seed(random_state)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def timestamped_dir(base_dir: str, suffix: str = "_model") -> str:
    """Create a timestamped directory inside base_dir and return its path."""
    ensure_dir(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir, f"{stamp}{suffix}")
    ensure_dir(out)
    return out


_SPLIT_RE = re.compile(config.SEPARATORS_REGEX)


def parse_groups(value: Any) -> List[str]:
    """Parse a raw group string into a list of normalized labels.

    Accepts separators: ',', '|', ';', '/'. Returns only labels present in TARGET_LABELS.
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        candidates = [str(v).strip() for v in value]
    else:
        candidates = [s.strip() for s in _SPLIT_RE.split(str(value)) if s.strip()]
    normalized = []
    for c in candidates:
        # Simple normalization: capitalize first letter, lower the rest
        label = c.strip()
        normalized.append(label)
    # Keep only known labels, preserve order without duplicates
    seen = set()
    filtered: List[str] = []
    for label in normalized:
        if label in config.TARGET_LABELS and label not in seen:
            seen.add(label)
            filtered.append(label)
    return filtered


def join_labels(labels: Iterable[str]) -> str:
    """Join labels with '|' for CSV persistence."""
    return "|".join(labels)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

