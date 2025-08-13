from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from . import config


def compute_metrics(y_true_bin, y_pred_bin, label_names: List[str]) -> Dict:
    metrics = {
        "f1_micro": float(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "classification_report": classification_report(
            y_true_bin, y_pred_bin, target_names=label_names, zero_division=0
        ),
    }
    return metrics


def save_text_report(metrics: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Classification Metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"F1 micro: {metrics['f1_micro']:.4f}\n")
        f.write(f"F1 macro: {metrics['f1_macro']:.4f}\n")
        f.write(f"Subset accuracy: {metrics['subset_accuracy']:.4f}\n\n")
        f.write(metrics["classification_report"]) 


def plot_label_f1(y_true_bin, y_pred_bin, label_names: List[str], out_path: str) -> None:
    per_label_f1 = []
    for idx, _ in enumerate(label_names):
        per_label_f1.append(
            f1_score(y_true_bin[:, idx], y_pred_bin[:, idx], average="binary", zero_division=0)
        )

    plt.figure(figsize=(8, 4))
    order = np.argsort(per_label_f1)[::-1]
    plt.bar([label_names[i] for i in order], [per_label_f1[i] for i in order])
    plt.xticks(rotation=30, ha="right")
    plt.title("F1 por etiqueta (one-vs-rest)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

