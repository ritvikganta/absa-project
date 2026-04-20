"""
Zero-shot and few-shot GPT-4o evaluation on ABSA.
Falls back to mock (keyword heuristic) when OPENAI_API_KEY is not set.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import compute_metrics, save_results
from scripts.preprocess import ID_TO_LABEL, LABEL_MAP, get_splits

SYSTEM_PROMPT = (
    "You are a sentiment analysis expert. Given a restaurant review sentence and a target aspect, "
    "classify the sentiment expressed toward that aspect as one of: positive, negative, neutral, conflict. "
    "'conflict' means the sentence expresses both positive and negative sentiment toward the aspect. "
    "Respond with exactly one word: positive, negative, neutral, or conflict."
)

FEW_SHOT_EXAMPLES = [
    {
        "text": "The food was amazing but the service was terrible.",
        "aspect": "food",
        "label": "positive",
    },
    {
        "text": "I waited 45 minutes and the waiter was rude.",
        "aspect": "service",
        "label": "negative",
    },
    {
        "text": "The prices are pretty average for this area.",
        "aspect": "price",
        "label": "neutral",
    },
    {
        "text": "The pasta was excellent but slightly overcooked.",
        "aspect": "food",
        "label": "conflict",
    },
]


def parse_label(response_text):
    text = response_text.strip().lower()
    for label in ["positive", "negative", "neutral", "conflict"]:
        if label in text:
            return LABEL_MAP[label]
    return LABEL_MAP["neutral"]


def _build_messages(text, aspect, mode):
    user_content = f"Sentence: {text}\nAspect: {aspect}\nSentiment:"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "few_shot":
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user",
                              "content": f"Sentence: {ex['text']}\nAspect: {ex['aspect']}\nSentiment:"})
            messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": user_content})
    return messages


def predict_openai(test_df, mode="zero_shot", limit=None):
    from openai import OpenAI
    client = OpenAI()

    rows = test_df.head(limit) if limit else test_df
    predictions = []
    for i, (_, row) in enumerate(rows.iterrows()):
        if i % 50 == 0:
            print(f"  LLM eval: {i}/{len(rows)}")
        messages = _build_messages(row["text"], row["aspect"], mode)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=5,
                temperature=0,
            )
            pred = parse_label(resp.choices[0].message.content)
        except Exception as e:
            print(f"  API error: {e}, defaulting to neutral")
            pred = LABEL_MAP["neutral"]
        predictions.append(pred)
    return np.array(predictions)


def mock_predict(test_df, limit=None):
    """Keyword heuristic — used when no API key is available."""
    POSITIVE_WORDS = {"great", "good", "excellent", "amazing", "delicious", "love", "best",
                      "wonderful", "fantastic", "fresh", "tasty", "nice", "awesome", "perfect"}
    NEGATIVE_WORDS = {"bad", "terrible", "awful", "horrible", "poor", "worst", "disgusting",
                      "slow", "rude", "dirty", "expensive", "mediocre", "disappointing", "cold"}

    rows = test_df.head(limit) if limit else test_df
    predictions = []
    for _, row in rows.iterrows():
        words = set(row["text"].lower().split())
        has_pos = bool(words & POSITIVE_WORDS)
        has_neg = bool(words & NEGATIVE_WORDS)
        if has_pos and has_neg:
            predictions.append(LABEL_MAP["conflict"])
        elif has_pos:
            predictions.append(LABEL_MAP["positive"])
        elif has_neg:
            predictions.append(LABEL_MAP["negative"])
        else:
            predictions.append(LABEL_MAP["neutral"])
    return np.array(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "mock"], default="mock")
    parser.add_argument("--output", type=str, default="results/llm_mock.json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N examples")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    _, test_df = get_splits(sample=args.sample)
    print(f"Test set: {len(test_df)} examples")

    if args.mode == "mock":
        print("Running mock (keyword heuristic) evaluation...")
        y_pred = mock_predict(test_df, limit=args.limit)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("OPENAI_API_KEY not set. Use --mode mock instead.")
            sys.exit(1)
        print(f"Running {args.mode} GPT-4o evaluation (limit={args.limit})...")
        y_pred = predict_openai(test_df, mode=args.mode, limit=args.limit)

    subset = test_df.head(args.limit) if args.limit else test_df
    y_true = subset["label"].values

    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = f"llm_{args.mode}"
    metrics["predictions"] = y_pred.tolist()
    metrics["true_labels"] = y_true.tolist()

    save_results(metrics, args.output)
    print(f"\n{args.mode} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  Per-class: {metrics['per_class_f1']}")
