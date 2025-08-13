from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loading import load_dataset, prepare_text, extract_features_and_labels
from src.evaluation import compute_metrics, plot_label_f1, save_text_report
from src.modeling import fit_with_search, get_label_binarizer, save_artifacts
from src.utils import ensure_dir, save_json, set_global_seed, timestamped_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate multi-label classifier")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_type", type=str, choices=["logistic", "linearsvc"], default="logistic")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=config.RANDOM_STATE_DEFAULT)
    parser.add_argument("--cv", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)

    df = load_dataset(args.data_path)
    df = prepare_text(df)

    # Parse labels and prepare multilabel binarizer
    X_all, labels_df = extract_features_and_labels(df)
    mlb = get_label_binarizer(config.TARGET_LABELS)
    y_all = mlb.transform(labels_df["labels"])  # shape: (n_samples, n_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.random_state
    )

    best_model, search_info = fit_with_search(
        X_train, y_train, model_type=args.model_type, cv=args.cv, n_jobs=-1
    )

    # Evaluate
    y_pred_test = best_model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred_test, label_names=list(mlb.classes_))
    metrics.update(search_info)

    # Persist artifacts
    run_dir = timestamped_dir(args.output_dir, suffix="_model")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "figs"))
    save_artifacts(best_model, mlb, run_dir)
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_text_report(metrics, os.path.join(run_dir, "report.txt"))
    plot_label_f1(y_test, y_pred_test, list(mlb.classes_), os.path.join(run_dir, "figs", "f1_per_label.png"))

    print("Artifacts saved to:", run_dir)
    print("\nTop-level metrics:")
    print({k: metrics[k] for k in ["f1_micro", "f1_macro", "subset_accuracy"]})


if __name__ == "__main__":
    main()

