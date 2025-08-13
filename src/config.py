"""Project-wide configuration constants."""
from typing import List


TARGET_LABELS: List[str] = [
    "Cardiovascular",
    "Neurological",
    "Hepatorenal",
    "Oncological",
]

TEXT_COLUMNS: List[str] = ["title", "abstract"]
GROUP_COLUMN: str = "group"
COMBINED_TEXT_COLUMN: str = "text"
PREDICTION_COLUMN: str = "predicted_groups"

SEPARATORS_REGEX: str = r"[\,\|;/]+"

RANDOM_STATE_DEFAULT: int = 42
STOPWORDS_LANGUAGE: str = "english"

DEFAULT_OUTPUT_DIR: str = "outputs"
DEFAULT_REPORTS_DIR: str = "reports"

