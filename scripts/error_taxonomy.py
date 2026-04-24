"""
Categorizes BERT test-set errors into a 6-class taxonomy:
  conflict_class, boundary, negation, implicit, multi_aspect, other

Priority order: conflict_class > boundary > negation > implicit > multi_aspect > other
"""

import argparse
import json
import os
import re

NEGATION_CUES = {
    "not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere",
    "without", "lack", "lacks", "lacking", "failed", "fails", "fail",
    "couldn't", "didn't", "doesn't", "don't", "wasn't", "weren't", "isn't",
    "aren't", "won't", "can't", "cannot", "hardly", "barely", "scarcely",
}

SENTIMENT_WORDS = {
    "positive": {"great", "good", "excellent", "amazing", "wonderful", "fantastic",
                 "delicious", "tasty", "fresh", "nice", "love", "best", "perfect",
                 "awesome", "superb", "outstanding", "brilliant", "friendly", "fast"},
    "negative": {"bad", "terrible", "awful", "horrible", "poor", "worst", "disgusting",
                 "slow", "rude", "dirty", "expensive", "overpriced", "mediocre",
                 "disappointing", "cold", "bland", "stale", "unfriendly", "hate"},
    "neutral": {"average", "okay", "ok", "fine", "decent", "normal", "standard",
                "typical", "usual", "regular"},
}

COMMON_ASPECTS = {
    "food", "service", "price", "ambiance", "atmosphere", "location",
    "staff", "waiter", "waitress", "menu", "portion", "quality",
    "value", "taste", "drink", "dessert", "place",
}

CONJUNCTION_PATTERNS = re.compile(
    r"\b(but|however|although|though|yet|while|whereas|nevertheless|nonetheless|despite|except)\b",
    re.IGNORECASE
)


def _has_negation_near_aspect(text, aspect, window=40):
    text_lower = text.lower()
    aspect_lower = aspect.lower()
    idx = text_lower.find(aspect_lower)
    if idx == -1:
        return False
    start = max(0, idx - window)
    end = min(len(text_lower), idx + len(aspect_lower) + window)
    region = text_lower[start:end]
    tokens = set(re.findall(r"\b\w+\b", region))
    return bool(tokens & NEGATION_CUES)


def _has_implicit_aspect(text, true_label_name):
    text_lower = text.lower()
    tokens = set(re.findall(r"\b\w+\b", text_lower))
    sentiment_words = SENTIMENT_WORDS.get(true_label_name, set())
    return len(tokens & sentiment_words) == 0


def _has_multi_aspect(text, aspect):
    text_lower = text.lower()
    found = 0
    for a in COMMON_ASPECTS:
        if a != aspect.lower() and re.search(r"\b" + re.escape(a) + r"\b", text_lower):
            found += 1
    return found >= 2


def categorize_error(text, aspect, true_label, pred_label):
    ID_TO_LABEL = {0: "positive", 1: "negative", 2: "neutral", 3: "conflict"}
    true_name = ID_TO_LABEL.get(true_label, "neutral")

    if true_label == 3:  # conflict
        return "conflict_class"

    text_lower = text.lower()
    if aspect.lower() not in text_lower:
        return "boundary"

    if _has_negation_near_aspect(text, aspect):
        return "negation"

    if _has_implicit_aspect(text, true_name):
        return "implicit"

    if _has_multi_aspect(text, aspect) or bool(CONJUNCTION_PATTERNS.search(text)):
        return "multi_aspect"

    return "other"


def run_taxonomy(predictions_path, output_path):
    with open(predictions_path) as f:
        data = json.load(f)

    texts = data.get("texts", [])
    aspects = data.get("aspects", [])
    true_labels = data.get("true_labels", [])
    pred_labels = data.get("predictions", [])

    if not texts:
        raise ValueError("predictions JSON must contain 'texts', 'aspects', 'true_labels', 'predictions'")

    errors = []
    category_counts = {
        "conflict_class": 0, "boundary": 0, "negation": 0,
        "implicit": 0, "multi_aspect": 0, "other": 0
    }

    for text, aspect, true_l, pred_l in zip(texts, aspects, true_labels, pred_labels):
        if true_l == pred_l:
            continue
        cat = categorize_error(text, aspect, true_l, pred_l)
        category_counts[cat] += 1
        errors.append({
            "text": text, "aspect": aspect,
            "true_label": true_l, "pred_label": pred_l,
            "category": cat,
        })

    total_errors = len(errors)
    total_examples = len(texts)
    error_rate = total_errors / total_examples if total_examples > 0 else 0

    result = {
        "total_examples": total_examples,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "category_counts": category_counts,
        "category_pct": {k: v / total_errors if total_errors > 0 else 0
                         for k, v in category_counts.items()},
        "errors": errors,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nError Taxonomy ({total_errors} errors / {total_examples} examples, error rate: {error_rate:.2%})")
    for cat, count in category_counts.items():
        pct = count / total_errors * 100 if total_errors > 0 else 0
        print(f"  {cat:<20}: {count:>4} ({pct:.1f}%)")
    print(f"\nSaved to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to results/bert.json (must have texts, aspects, true_labels, predictions)")
    parser.add_argument("--output", type=str, default="results/errors.json")
    args = parser.parse_args()

    run_taxonomy(args.predictions, args.output)
