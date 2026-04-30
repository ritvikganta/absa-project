"""
LLM evaluation for ABSA: supports HuggingFace Inference API (default), Gemini, OpenAI, and mock.
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
    {"text": "The food was amazing but the service was terrible.", "aspect": "food", "label": "positive"},
    {"text": "I waited 45 minutes and the waiter was rude.", "aspect": "service", "label": "negative"},
    {"text": "The prices are pretty average for this area.", "aspect": "price", "label": "neutral"},
    {"text": "The pasta was excellent but slightly overcooked.", "aspect": "food", "label": "conflict"},
]


def parse_label(response_text):
    text = response_text.strip().lower()
    for label in ["positive", "negative", "neutral", "conflict"]:
        if label in text:
            return LABEL_MAP[label]
    return LABEL_MAP["neutral"]


def _build_messages(text, aspect, mode):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "few_shot":
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user",
                              "content": f"Sentence: {ex['text']}\nAspect: {ex['aspect']}\nSentiment:"})
            messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": f"Sentence: {text}\nAspect: {aspect}\nSentiment:"})
    return messages


def _build_prompt(text, aspect, mode):
    few_shot_block = ""
    if mode == "few_shot":
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_block += f"Sentence: {ex['text']}\nAspect: {ex['aspect']}\nSentiment: {ex['label']}\n\n"
    return f"{SYSTEM_PROMPT}\n\n{few_shot_block}Sentence: {text}\nAspect: {aspect}\nSentiment:"


def predict_local_zeroshot(test_df, limit=None, model_name="facebook/bart-large-mnli"):
    """Zero-shot classification using a local NLI model. No API key needed."""
    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device = 0  # transformers pipeline uses device index for MPS via 'mps'
        device_arg = "mps"
    elif torch.cuda.is_available():
        device_arg = 0
    else:
        device_arg = -1  # CPU

    print(f"Loading {model_name} locally...")
    classifier = pipeline("zero-shot-classification", model=model_name, device=device_arg)

    candidate_labels = ["positive", "negative", "neutral", "conflict"]
    rows = test_df.head(limit) if limit else test_df
    predictions = []
    for i, (_, row) in enumerate(rows.iterrows()):
        if i % 50 == 0:
            print(f"  Local eval: {i}/{len(rows)}")
        text = f"Aspect: {row['aspect']}. {row['text']}"
        result = classifier(text, candidate_labels=candidate_labels, multi_label=False)
        top_label = result["labels"][0]
        predictions.append(LABEL_MAP[top_label])
    return np.array(predictions)


def predict_gemini(test_df, mode="zero_shot", limit=None, model_name="gemini-2.5-pro-preview-05-06"):
    import time
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=api_key)

    rows = test_df.head(limit) if limit else test_df
    predictions = []
    for i, (_, row) in enumerate(rows.iterrows()):
        if i % 50 == 0:
            print(f"  Gemini eval: {i}/{len(rows)}")
        prompt = _build_prompt(row["text"], row["aspect"], mode)
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=10, temperature=0.0),
            )
            pred = parse_label(resp.text)
        except Exception as e:
            print(f"  API error at {i}: {e}, defaulting to neutral")
            pred = LABEL_MAP["neutral"]
            time.sleep(5)
        predictions.append(pred)
        time.sleep(12.1)
    return np.array(predictions)


def predict_openai(test_df, mode="zero_shot", limit=None):
    from openai import OpenAI
    client = OpenAI()

    rows = test_df.head(limit) if limit else test_df
    predictions = []
    for i, (_, row) in enumerate(rows.iterrows()):
        if i % 50 == 0:
            print(f"  OpenAI eval: {i}/{len(rows)}")
        messages = _build_messages(row["text"], row["aspect"], mode)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=5, temperature=0,
            )
            pred = parse_label(resp.choices[0].message.content)
        except Exception as e:
            print(f"  API error: {e}, defaulting to neutral")
            pred = LABEL_MAP["neutral"]
        predictions.append(pred)
    return np.array(predictions)


def mock_predict(test_df, limit=None):
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
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "mock"], default="zero_shot")
    parser.add_argument("--provider", choices=["local", "gemini", "openai"], default="local")
    parser.add_argument("--model", type=str, default="facebook/bart-large-mnli",
                        help="Model name for HF or Gemini provider")
    parser.add_argument("--output", type=str, default="results/llm_local.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    _, test_df = get_splits(sample=args.sample)
    print(f"Test set: {len(test_df)} examples")

    if args.mode == "mock":
        print("Running mock (keyword heuristic) evaluation...")
        y_pred = mock_predict(test_df, limit=args.limit)
    elif args.provider == "local":
        print(f"Running zero-shot local ({args.model}) evaluation (limit={args.limit})...")
        y_pred = predict_local_zeroshot(test_df, limit=args.limit, model_name=args.model)
    elif args.provider == "gemini":
        print(f"Running {args.mode} Gemini ({args.model}) evaluation (limit={args.limit})...")
        y_pred = predict_gemini(test_df, mode=args.mode, limit=args.limit, model_name=args.model)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("OPENAI_API_KEY not set.")
            sys.exit(1)
        print(f"Running {args.mode} GPT-4o evaluation (limit={args.limit})...")
        y_pred = predict_openai(test_df, mode=args.mode, limit=args.limit)

    subset = test_df.head(args.limit) if args.limit else test_df
    y_true = subset["label"].values

    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = f"llm_{args.provider}_{args.mode}"
    metrics["predictions"] = y_pred.tolist()
    metrics["true_labels"] = y_true.tolist()

    save_results(metrics, args.output)
    print(f"\n{args.mode} ({args.provider}) Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  Per-class: {metrics['per_class_f1']}")
