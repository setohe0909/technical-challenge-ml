from __future__ import annotations

import argparse
import os

import pandas as pd

from src import config
from src.data_loading import prepare_text
from src.evaluation import compute_metrics, save_text_report
from src.modeling import load_artifacts
from src.utils import join_labels, parse_groups, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels for a CSV using a saved model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, mlb = load_artifacts(args.model_dir)

    df = pd.read_csv(args.input_csv)
    if not set(["title", "abstract"]).issubset(df.columns):
        raise ValueError("Input CSV must contain 'title' and 'abstract' columns")

    df = prepare_text(df)
    X = df[config.COMBINED_TEXT_COLUMN]
    y_pred_bin = model.predict(X)

    predicted = mlb.inverse_transform(y_pred_bin)
    df[config.PREDICTION_COLUMN] = [join_labels(labels) for labels in predicted]
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to: {args.output_csv}")

    if args.evaluate and config.GROUP_COLUMN in df.columns:
        y_true = [parse_groups(v) for v in df[config.GROUP_COLUMN].tolist()]
        y_true_bin = mlb.transform(y_true)
        metrics = compute_metrics(y_true_bin, y_pred_bin, label_names=list(mlb.classes_))
        save_json(metrics, os.path.join(args.model_dir, "metrics_from_predict.json"))
        save_text_report(metrics, os.path.join(args.model_dir, "report_from_predict.txt"))
        print("Evaluation completed. Metrics stored next to the model directory.")


if __name__ == "__main__":
    main()

