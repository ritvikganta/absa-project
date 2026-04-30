"""
Fine-tune bert-base-uncased for ABSA (4-class classification).
Input format: "[ASPECT: <aspect>] <text>"
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import compute_metrics, save_results
from scripts.preprocess import get_splits

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 4
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model")


class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = df["input_text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(train_df, dev_df, checkpoint_dir=CHECKPOINT_DIR, epochs=EPOCHS):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    train_loader = DataLoader(ABSADataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(ABSADataset(dev_df, tokenizer), batch_size=BATCH_SIZE)

    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on dev
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch["labels"].cpu().numpy())

        dev_metrics = compute_metrics(all_true, all_preds)
        dev_f1 = dev_metrics["macro_f1"]
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, dev macro-F1={dev_f1:.4f}")
        history.append({"epoch": epoch + 1, "loss": avg_loss, "dev_macro_f1": dev_f1})

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  -> Saved new best checkpoint (dev F1={best_f1:.4f})")

    return history


def predict_with_checkpoint(test_df, checkpoint_path=CHECKPOINT_DIR):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    loader = DataLoader(ABSADataset(test_df, tokenizer), batch_size=BATCH_SIZE)
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--output", type=str, default="results/bert.json")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train_df, test_df = get_splits(sample=args.sample)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    if args.train:
        # 90/10 train/dev split
        dev_size = max(1, int(0.1 * len(train_df)))
        dev_df = train_df.sample(dev_size, random_state=42)
        train_split = train_df.drop(dev_df.index).reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
        print(f"Training on {len(train_split)} examples, dev on {len(dev_df)}")

        history = train(train_split, dev_df, checkpoint_dir=args.checkpoint, epochs=args.epochs)

    if args.eval or args.train:
        checkpoint = args.checkpoint
        if not os.path.exists(os.path.join(checkpoint, "config.json")):
            print(f"No checkpoint found at {checkpoint}. Run --train first.")
            sys.exit(1)

        print(f"\nEvaluating checkpoint: {checkpoint}")
        y_pred = predict_with_checkpoint(test_df, checkpoint)
        y_true = test_df["label"].values

        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = "bert"

        # Save predictions + texts + aspects for error_taxonomy.py
        metrics["predictions"] = y_pred.tolist()
        metrics["true_labels"] = y_true.tolist()
        metrics["texts"] = test_df["text"].tolist()
        metrics["aspects"] = test_df["aspect"].tolist()

        save_results(metrics, args.output)
        print(f"\nTest Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
        print(f"  Per-class: {metrics['per_class_f1']}")
