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
_CANONICAL_LABEL_BY_LOWER = {label.lower(): label for label in config.TARGET_LABELS}


def parse_groups(value: Any) -> List[str]:
    """Parse `group` into canonical labels defined in TARGET_LABELS.

    - Case-insensitive matching (e.g., 'neurological' → 'Neurological')
    - Accepts separators: ',', '|', ';', '/'
    - Deduplicates while preserving order
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value]
    else:
        tokens = [s.strip() for s in _SPLIT_RE.split(str(value)) if s.strip()]

    seen = set()
    parsed: List[str] = []
    for tok in tokens:
        key = tok.lower()
        if key in _CANONICAL_LABEL_BY_LOWER:
            canon = _CANONICAL_LABEL_BY_LOWER[key]
            if canon not in seen:
                seen.add(canon)
                parsed.append(canon)
    return parsed


def join_labels(labels: Iterable[str]) -> str:
    """Join labels with '|' for CSV persistence."""
    return "|".join(labels)


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

