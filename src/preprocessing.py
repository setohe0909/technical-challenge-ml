from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


def _ensure_nltk_resources() -> None:
    try:
        _ = stopwords.words(config.STOPWORDS_LANGUAGE)
    except LookupError:
        nltk.download("stopwords")


class SimpleTextCleaner(BaseEstimator, TransformerMixin):
    """Lowercase, basic punctuation removal and stopword filtering."""

    def __init__(self, remove_digits: bool = True):
        _ensure_nltk_resources()
        self.remove_digits = remove_digits
        self._token_re = re.compile(r"[A-Za-z]+" if remove_digits else r"[A-Za-z0-9]+")
        self._stop = set(stopwords.words(config.STOPWORDS_LANGUAGE))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean_text(text) for text in X]

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        tokens = [t.lower() for t in self._token_re.findall(text)]
        tokens = [t for t in tokens if t not in self._stop and len(t) > 1]
        return " ".join(tokens)

