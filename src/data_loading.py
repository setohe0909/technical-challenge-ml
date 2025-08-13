from __future__ import annotations

from typing import Tuple

import pandas as pd

from . import config, utils


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset CSV and ensure required columns exist."""
    df = pd.read_csv(csv_path)
    missing = [c for c in [*config.TEXT_COLUMNS, config.GROUP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


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

