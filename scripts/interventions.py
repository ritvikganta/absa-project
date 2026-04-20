"""
Targeted interventions for ABSA failure modes.
Each intervention modifies the training data, re-trains BERT, and returns metrics.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import compute_metrics, save_results
from scripts.preprocess import get_splits

CHECKPOINT_BASE = os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints")

NEGATION_CUES = ["not", "no", "never", "neither", "nor", "nothing", "without",
                 "couldn't", "didn't", "doesn't", "don't", "wasn't", "weren't",
                 "isn't", "aren't", "won't", "can't", "cannot", "hardly", "barely"]

CONJUNCTION_RE = re.compile(
    r"\b(but|however|although|though|yet|while|whereas|nevertheless|despite)\b",
    re.IGNORECASE
)


# --- Intervention 1: Negation marking ---

def _mark_negation(text):
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        token_clean = tokens[i].lower().rstrip(".,!?;:")
        if token_clean in NEGATION_CUES:
            scope = []
            j = i + 1
            while j < min(i + 6, len(tokens)):
                scope.append(tokens[j])
                j += 1
            if scope:
                result.append("[NEG]")
                result.append(tokens[i])
                result.extend(scope)
                result.append("[/NEG]")
                i = j
                continue
        result.append(tokens[i])
        i += 1
    return " ".join(result)


def apply_negation_marking(df):
    df = df.copy()
    df["input_text"] = df["input_text"].apply(_mark_negation)
    return df


# --- Intervention 2: Conflict-class oversampling ---

def apply_conflict_oversample(train_df, target_pct=0.15):
    conflict = train_df[train_df["label"] == 3]
    non_conflict = train_df[train_df["label"] != 3]
    if len(conflict) == 0:
        return train_df
    target_count = int(target_pct * len(train_df) / (1 - target_pct))
    n_copies = max(1, target_count // len(conflict))
    oversampled = pd.concat([conflict] * n_copies, ignore_index=True)
    # Trim or pad to exact target
    if len(oversampled) > target_count:
        oversampled = oversampled.sample(target_count, random_state=42)
    result = pd.concat([non_conflict, oversampled], ignore_index=True)
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


# --- Intervention 3: Sentence decomposition ---

def _decompose(text, aspect):
    clauses = CONJUNCTION_RE.split(text)
    aspect_lower = aspect.lower()
    for clause in clauses:
        if aspect_lower in clause.lower():
            return clause.strip()
    return text


def apply_decomposition(df):
    df = df.copy()
    def _rebuild(row):
        decomposed = _decompose(row["text"], row["aspect"])
        return f"[ASPECT: {row['aspect']}] {decomposed}"
    df["input_text"] = df.apply(_rebuild, axis=1)
    return df


# --- Training helper ---

def run_intervention(name, train_df, test_df, epochs=4):
    from models.bert_model import train, predict_with_checkpoint

    checkpoint_dir = os.path.join(CHECKPOINT_BASE, f"intervention_{name}")

    dev_size = max(1, int(0.1 * len(train_df)))
    dev_df = train_df.sample(dev_size, random_state=42)
    train_split = train_df.drop(dev_df.index).reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    print(f"\n[{name}] Training on {len(train_split)} examples...")
    train(train_split, dev_df, checkpoint_dir=checkpoint_dir, epochs=epochs)

    y_pred = predict_with_checkpoint(test_df, checkpoint_dir)
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = f"bert_{name}"
    print(f"[{name}] Test macro-F1: {metrics['macro_f1']:.4f}")
    return metrics


def run_all_interventions(train_df, test_df, epochs=4):
    results = {}

    # negation
    neg_train = apply_negation_marking(train_df)
    neg_test = apply_negation_marking(test_df)
    results["negation"] = run_intervention("negation", neg_train, neg_test, epochs)

    # conflict oversample
    over_train = apply_conflict_oversample(train_df)
    results["conflict_oversample"] = run_intervention("conflict_oversample", over_train, test_df, epochs)

    # sentence decomposition
    decomp_train = apply_decomposition(train_df)
    decomp_test = apply_decomposition(test_df)
    results["decompose"] = run_intervention("decompose", decomp_train, decomp_test, epochs)

    # combined: all three
    combined_train = apply_negation_marking(apply_conflict_oversample(apply_decomposition(train_df)))
    combined_test = apply_negation_marking(apply_decomposition(test_df))
    results["combined"] = run_intervention("combined", combined_train, combined_test, epochs)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention", type=str,
                        choices=["negation", "conflict_oversample", "decompose", "combined", "all"],
                        required=True)
    parser.add_argument("--output", type=str, default="results/interventions.json")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    train_df, test_df = get_splits(sample=args.sample)

    if args.intervention == "all":
        results = run_all_interventions(train_df, test_df, epochs=args.epochs)
    elif args.intervention == "negation":
        results = {"negation": run_intervention("negation",
                    apply_negation_marking(train_df), apply_negation_marking(test_df), args.epochs)}
    elif args.intervention == "conflict_oversample":
        results = {"conflict_oversample": run_intervention("conflict_oversample",
                    apply_conflict_oversample(train_df), test_df, args.epochs)}
    elif args.intervention == "decompose":
        results = {"decompose": run_intervention("decompose",
                    apply_decomposition(train_df), apply_decomposition(test_df), args.epochs)}
    elif args.intervention == "combined":
        combined_train = apply_negation_marking(apply_conflict_oversample(apply_decomposition(train_df)))
        combined_test = apply_negation_marking(apply_decomposition(test_df))
        results = {"combined": run_intervention("combined", combined_train, combined_test, args.epochs)}

    save_results(results, args.output)

    from scripts.evaluate import print_results_table
    print("\n--- Intervention Results ---")
    print_results_table(results)
