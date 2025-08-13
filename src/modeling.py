from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from .preprocessing import SimpleTextCleaner


def build_pipeline(model_type: str = "logistic") -> Pipeline:
    """Create a text classification pipeline with TF-IDF + OneVsRest classifier."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
    if model_type == "logistic":
        base = LogisticRegression(max_iter=200, n_jobs=None)
    elif model_type == "linearsvc":
        base = LinearSVC()
    else:
        raise ValueError("model_type must be 'logistic' or 'linearsvc'")

    clf = OneVsRestClassifier(base)

    pipe = Pipeline(
        steps=[
            ("clean", SimpleTextCleaner()),
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )
    return pipe


def hyperparameter_grid(model_type: str = "logistic") -> Dict:
    if model_type == "logistic":
        return {
            "tfidf__max_features": [20000, 40000],
            "clf__estimator__C": [0.5, 1.0, 2.0],
            "clf__estimator__class_weight": ["balanced", None],
        }
    else:  # linearsvc
        return {
            "tfidf__max_features": [20000, 40000],
            "clf__estimator__C": [0.5, 1.0, 2.0],
        }


def fit_with_search(
    X_train,
    y_train,
    model_type: str = "logistic",
    cv: int = 3,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict]:
    pipe = build_pipeline(model_type=model_type)
    grid = hyperparameter_grid(model_type=model_type)
    search = GridSearchCV(pipe, grid, scoring="f1_micro", cv=cv, n_jobs=n_jobs, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score_cv_f1_micro": float(search.best_score_),
    }


def get_label_binarizer(classes) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer(classes=tuple(classes))
    mlb.fit([classes])
    return mlb


def save_artifacts(model: Pipeline, mlb: MultiLabelBinarizer, out_dir: str) -> None:
    joblib.dump(model, f"{out_dir}/best_model.joblib")
    joblib.dump(mlb, f"{out_dir}/label_binarizer.joblib")


def load_artifacts(model_dir: str) -> Tuple[Pipeline, MultiLabelBinarizer]:
    model: Pipeline = joblib.load(f"{model_dir}/best_model.joblib")
    mlb: MultiLabelBinarizer = joblib.load(f"{model_dir}/label_binarizer.joblib")
    return model, mlb

