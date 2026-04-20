"""
Data loading and preprocessing for SemEval-2014 Task 4 (restaurant ABSA).
Dataset: tomaarsen/setfit-absa-semeval-restaurants on HuggingFace.
"""

import argparse
import os
import re

import pandas as pd

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2, "conflict": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SAMPLE_DIR = os.path.join(DATA_DIR, "samples")


def _hf_to_df(hf_dataset):
    rows = []
    for ex in hf_dataset:
        text = ex.get("text", ex.get("sentence", ""))
        aspect = ex.get("span", ex.get("aspect", ""))
        polarity = ex.get("label", ex.get("polarity", "neutral"))
        if isinstance(polarity, int):
            polarity = ID_TO_LABEL.get(polarity, "neutral")
        polarity = polarity.lower().strip()
        if polarity not in LABEL_MAP:
            polarity = "neutral"
        rows.append({"text": text, "aspect": aspect, "polarity": polarity, "label": LABEL_MAP[polarity]})
    return pd.DataFrame(rows)


def download_dataset():
    from datasets import load_dataset

    os.makedirs(RAW_DIR, exist_ok=True)
    print("Downloading tomaarsen/setfit-absa-semeval-restaurants ...")
    ds = load_dataset("tomaarsen/setfit-absa-semeval-restaurants")

    train_df = _hf_to_df(ds["train"])
    test_df = _hf_to_df(ds["test"])

    train_df.to_csv(os.path.join(RAW_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(RAW_DIR, "test.csv"), index=False)
    print(f"Saved {len(train_df)} train / {len(test_df)} test rows to {RAW_DIR}")
    return train_df, test_df


def load_from_csv():
    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Run --download first. Expected: {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_text(row, mode="aspect_concat"):
    text = clean_text(row["text"])
    aspect = str(row["aspect"]).strip()
    if mode == "aspect_concat":
        return f"[ASPECT: {aspect}] {text}"
    elif mode == "qa_format":
        return f"What is the sentiment toward {aspect}? {text}"
    else:
        return text


def get_splits(sample=False, n=500, download_if_missing=True):
    try:
        train_df, test_df = load_from_csv()
    except FileNotFoundError:
        if download_if_missing:
            train_df, test_df = download_dataset()
        else:
            raise

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)
    train_df["input_text"] = train_df.apply(build_input_text, axis=1)
    test_df["input_text"] = test_df.apply(build_input_text, axis=1)

    if sample:
        train_df = train_df.sample(min(n, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(min(n // 5, len(test_df)), random_state=42).reset_index(drop=True)

    return train_df, test_df


def print_stats(df, name="Dataset"):
    print(f"\n{name}: {len(df)} examples")
    counts = df["polarity"].value_counts()
    for label, count in counts.items():
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download from HuggingFace")
    parser.add_argument("--sample", action="store_true", help="Save sample CSVs to data/samples/")
    parser.add_argument("--n", type=int, default=100, help="Number of sample rows")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    if args.download:
        download_dataset()

    if args.sample or args.stats:
        train_df, test_df = get_splits(download_if_missing=True)
        if args.stats:
            print_stats(train_df, "Train")
            print_stats(test_df, "Test")
        if args.sample:
            os.makedirs(SAMPLE_DIR, exist_ok=True)
            train_df.head(args.n).to_csv(os.path.join(SAMPLE_DIR, "train_sample.csv"), index=False)
            test_df.head(args.n // 5 or 20).to_csv(os.path.join(SAMPLE_DIR, "test_sample.csv"), index=False)
            print(f"Saved samples to {SAMPLE_DIR}")
