"""
Fine-tune DistilBERT to detect clickbait headlines.

Dataset: Chakraborty et al. (2016) - 32,000 balanced headlines
         (~16k clickbait, ~16k genuine news), downloaded from GitHub.

This is an explicit PyTorch training loop (not the HuggingFace Trainer) so that
every step - forward pass, loss, backward pass, optimiser step, scheduler - is visible.

Outputs (all written into ml/):
    clickbait_model/      fine-tuned model + tokenizer (loaded by the bot)
    metrics.json          accuracy / precision / recall / F1 + confusion matrix
    training_curve.png    train loss and validation accuracy per epoch
    confusion_matrix.png  confusion matrix on the held-out test set

Run:  python ml/train_clickbait.py
"""

import gzip
import io
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # write plots to file, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ----------------------------- configuration -----------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 48       # headlines are short
BATCH_SIZE = 64
EPOCHS     = 3
LR         = 2e-5
SEED       = 42

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "clickbait_model")
BASE_URL  = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset"

LABELS = {0: "genuine", 1: "clickbait"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------- data -----------------------------
def download(name: str) -> list[str]:
    """Fetch one gzipped headline file (cached on disk) and return its lines."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  downloading {name} ...")
        r = requests.get(f"{BASE_URL}/{name}", timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def load_dataset():
    print("Loading dataset...")
    clickbait = download("clickbait_data.gz")
    genuine   = download("non_clickbait_data.gz")
    texts  = clickbait + genuine
    labels = [1] * len(clickbait) + [0] * len(genuine)
    print(f"  clickbait: {len(clickbait)} | genuine: {len(genuine)} | total: {len(texts)}")

    # 80% train / 10% validation / 10% test, stratified so both classes stay balanced.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    print(f"  train: {len(X_tr)} | val: {len(X_val)} | test: {len(X_te)}")
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, padding="max_length",
                             max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


# ----------------------------- evaluation -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over a loader and return (predictions, gold labels)."""
    model.eval()
    preds, golds = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        golds.extend(labels.tolist())
    return preds, golds


def score(preds, golds) -> dict:
    return {
        "accuracy":  accuracy_score(golds, preds),
        "precision": precision_score(golds, preds),
        "recall":    recall_score(golds, preds),
        "f1":        f1_score(golds, preds),
    }


# ----------------------------- training -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}\n")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)

    train_loader = DataLoader(HeadlineDataset(X_tr, y_tr, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(HeadlineDataset(X_val, y_val, tokenizer), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(HeadlineDataset(X_te, y_te, tokenizer), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    history = {"train_loss": [], "val_accuracy": []}

    print(f"\nTraining for {EPOCHS} epochs ({total_steps} steps)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss      # forward pass
            loss.backward()                 # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                # update weights
            scheduler.step()                # decay learning rate
            optimizer.zero_grad()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running/step:.4f}")

        train_loss = running / len(train_loader)
        val_preds, val_golds = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_golds, val_preds)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_accuracy={val_acc:.4f}")

    # ---------------- final evaluation on the held-out test set ----------------
    print("\nEvaluating on the held-out test set...")
    preds, golds = evaluate(model, test_loader, device)
    metrics = score(preds, golds)
    cm = confusion_matrix(golds, preds)

    print("\n================ TEST RESULTS ================")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")
    print(f"\n  Confusion matrix (rows = actual, cols = predicted):")
    print(f"              genuine  clickbait")
    print(f"  genuine   {cm[0][0]:>8} {cm[0][1]:>10}")
    print(f"  clickbait {cm[1][0]:>8} {cm[1][1]:>10}")
    print("=============================================\n")

    # ---------------- save model, metrics and plots ----------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump({"test_metrics": metrics,
                   "confusion_matrix": cm.tolist(),
                   "history": history,
                   "config": {"model": MODEL_NAME, "epochs": EPOCHS,
                              "batch_size": BATCH_SIZE, "lr": LR, "max_len": MAX_LEN},
                   "dataset": {"train": len(X_tr), "val": len(X_val), "test": len(X_te)}},
                  f, indent=2)

    # training curve
    fig, ax1 = plt.subplots(figsize=(6, 4))
    epochs_x = range(1, EPOCHS + 1)
    ax1.plot(epochs_x, history["train_loss"], "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs_x, history["val_accuracy"], "s-", color="tab:blue", label="val accuracy")
    ax2.set_ylabel("Validation accuracy", color="tab:blue")
    plt.title("DistilBERT fine-tuning: clickbait detection")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "training_curve.png"), dpi=150)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["genuine", "clickbait"])
    ax.set_yticks([0, 1], ["genuine", "clickbait"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "confusion_matrix.png"), dpi=150)
    print("Saved metrics.json, training_curve.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
