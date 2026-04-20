"""
Baseline models: majority class, NBOW, TF-IDF+LR, VADER.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import compute_metrics, save_results
from scripts.preprocess import get_splits

LABEL_NAMES = ["positive", "negative", "neutral", "conflict"]


def run_majority_class(train_df, test_df):
    majority = train_df["label"].mode()[0]
    y_pred = np.full(len(test_df), majority)
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = "majority_class"
    return metrics


def run_nbow(train_df, test_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.feature_extraction.text import CountVectorizer

    pipe = Pipeline([
        ("vec", CountVectorizer(max_features=10000)),
        ("norm", Normalizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(train_df["input_text"], train_df["label"])
    y_pred = pipe.predict(test_df["input_text"])
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = "nbow"
    return metrics


def run_tfidf_lr(train_df, test_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    pipe = Pipeline([
        ("vec", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(train_df["input_text"], train_df["label"])
    y_pred = pipe.predict(test_df["input_text"])
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = "tfidf_lr"
    return metrics


def run_vader(test_df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    y_pred = []
    for _, row in test_df.iterrows():
        scores = analyzer.polarity_scores(row["text"])
        c = scores["compound"]
        if c >= 0.05:
            y_pred.append(0)  # positive
        elif c <= -0.05:
            y_pred.append(1)  # negative
        else:
            y_pred.append(2)  # neutral
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = "vader"
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/baselines.json")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    train_df, test_df = get_splits(sample=args.sample)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    results = {}

    print("\nRunning majority class baseline...")
    results["majority_class"] = run_majority_class(train_df, test_df)
    print(f"  Macro-F1: {results['majority_class']['macro_f1']:.4f}")

    print("Running NBOW baseline...")
    results["nbow"] = run_nbow(train_df, test_df)
    print(f"  Macro-F1: {results['nbow']['macro_f1']:.4f}")

    print("Running TF-IDF + LR baseline...")
    results["tfidf_lr"] = run_tfidf_lr(train_df, test_df)
    print(f"  Macro-F1: {results['tfidf_lr']['macro_f1']:.4f}")

    print("Running VADER baseline...")
    results["vader"] = run_vader(test_df)
    print(f"  Macro-F1: {results['vader']['macro_f1']:.4f}")

    save_results(results, args.output)

    from scripts.evaluate import print_results_table
    print("\n--- Baseline Results ---")
    print_results_table(results)
