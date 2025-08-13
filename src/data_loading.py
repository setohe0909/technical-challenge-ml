from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from . import config, utils


def load_dataset(csv_path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load dataset CSV and ensure required columns exist.

    Attempts multiple parsing strategies to handle common CSV issues:
    - Auto-detect delimiter (engine='python')
    - Explicit separators: ',', ';', '\t'
    - Robust quoting and escaping
    - Optionally, user-provided separator via `sep`
    """
    candidates = []
    if sep is not None:
        candidates.append({"sep": sep, "engine": "python"})
    # Try default fast engine first
    candidates.append({})
    # Then robust strategies
    candidates.extend([
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
    ])

    last_err: Exception | None = None
    for kw in candidates:
        try:
            df = pd.read_csv(
                csv_path,
                **kw,
                quotechar='"',
                escapechar='\\',
                on_bad_lines='error',
            )
            missing = [c for c in [*config.TEXT_COLUMNS, config.GROUP_COLUMN] if c not in df.columns]
            if missing:
                # Columns not found; continue trying alternative strategies
                last_err = ValueError(f"Missing required columns: {missing}")
                continue
            return df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to parse CSV. Try specifying a separator with --sep (',' ';' or '\t').\n"
        f"Last error: {last_err}"
    )


def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create combined text column from title and abstract."""
    df = df.copy()
    df[config.TEXT_COLUMNS] = df[config.TEXT_COLUMNS].fillna("")
    df[config.COMBINED_TEXT_COLUMN] = (
        df["title"].astype(str).str.strip() + " \n " + df["abstract"].astype(str).str.strip()
    )
    return df


def extract_features_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Return text Series X and DataFrame with parsed labels (list[str]) in column 'labels'."""
    df = df.copy()
    df["labels"] = df[config.GROUP_COLUMN].apply(utils.parse_groups)
    X = df[config.COMBINED_TEXT_COLUMN]
    return X, df[["labels"]]

