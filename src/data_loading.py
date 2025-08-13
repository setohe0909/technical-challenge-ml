from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from . import config, utils


def read_csv_robust(csv_path: str, sep: Optional[str] = None, require_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read CSV handling irregular delimiters/quotes. Optionally validate required columns."""
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
            if require_cols is not None:
                missing = [c for c in require_cols if c not in df.columns]
                if missing:
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


def load_dataset(csv_path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load dataset and ensure `title`, `abstract`, `group` exist."""
    return read_csv_robust(csv_path, sep=sep, require_cols=[*config.TEXT_COLUMNS, config.GROUP_COLUMN])


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

