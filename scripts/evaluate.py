"""
Shared evaluation utilities: metrics computation, results table, save/load.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_NAMES = ["positive", "negative", "neutral", "conflict"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [LABEL_NAMES[i] for i in labels if i < len(LABEL_NAMES)]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = {LABEL_NAMES[i]: float(per_class[j]) for j, i in enumerate(labels) if i < len(LABEL_NAMES)}
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": label_names,
    }


def print_results_table(results_dict):
    header = f"{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'Pos-F1':>8} {'Neg-F1':>8} {'Neu-F1':>8} {'Con-F1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results_dict.items():
        pcf = m.get("per_class_f1", {})
        print(
            f"{name:<30} {m.get('accuracy', 0):>10.4f} {m.get('macro_f1', 0):>10.4f}"
            f" {pcf.get('positive', 0):>8.4f} {pcf.get('negative', 0):>8.4f}"
            f" {pcf.get('neutral', 0):>8.4f} {pcf.get('conflict', 0):>8.4f}"
        )


def save_results(metrics, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path):
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir=None):
    results_dir = results_dir or RESULTS_DIR
    all_results = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        data = load_results(p)
        # Handle files that store multiple model results vs single result
        if "macro_f1" in data:
            all_results[p.stem] = data
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "macro_f1" in v:
                    all_results[f"{p.stem}/{k}"] = v
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Print all results in results/")
    parser.add_argument("--file", type=str, help="Print a specific results JSON file")
    args = parser.parse_args()

    if args.file:
        data = load_results(args.file)
        if "macro_f1" in data:
            print_results_table({os.path.basename(args.file): data})
        else:
            print_results_table(data)
    elif args.all:
        all_results = load_all_results()
        if all_results:
            print_results_table(all_results)
        else:
            print("No results found in results/")
