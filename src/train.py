"""Train the baseline CNN.

Usage (from the repo root):
    python -m src.train --data-root /kaggle/input/gtsrb-german-traffic-sign --epochs 15
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import GTSRBDataset, prepare_data
from .evaluate import clean_metrics, robustness_bench, top_confused_pairs
from .models import BaselineCNN
from .utils import CLASS_NAMES, get_device, set_seed


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="baseline")
    a = ap.parse_args()

    set_seed(a.seed)
    device = get_device()
    print("Device:", device, "| torch", torch.__version__)
    d = prepare_data(a.data_root, a.size, seed=a.seed)
    print({k: v.shape for k, v in d.items() if k.startswith(("X_", "y_"))})

    tr = DataLoader(GTSRBDataset(d["X_train"], d["y_train"]), a.batch_size, shuffle=True)
    va = DataLoader(GTSRBDataset(d["X_val"], d["y_val"]), a.batch_size)

    model = BaselineCNN(dropout=a.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = 0.0
    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, tr, device, criterion, opt)
        vl, vacc = run_epoch(model, va, device, criterion)
        for k, v in zip(hist, (tl, vl, ta, vacc)):
            hist[k].append(v)
        print(f"Epoch {ep:02d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {vacc:.3f}")
        if vacc > best:  # keep the best checkpoint by validation accuracy
            best = vacc
            torch.save(model.state_dict(), f"checkpoints/{a.name}.pth")

    # Learning curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"results/figures/{a.name}_learning_curves.png", dpi=150)

    # Final evaluation on the official test set with the best checkpoint
    model.load_state_dict(torch.load(f"checkpoints/{a.name}.pth", map_location=device))
    m, preds = clean_metrics(model, d["X_test"], d["y_test"], device)
    print(f"TEST  accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f}")
    pairs = top_confused_pairs(d["y_test"], preds, CLASS_NAMES)
    print(pairs.to_string(index=False))
    pairs.to_csv(f"results/{a.name}_confused_pairs.csv", index=False)
    with open(f"results/{a.name}_metrics.json", "w") as f:
        json.dump({**m, "best_val_acc": best, "args": vars(a)}, f, indent=2)

    # Quick robustness bench (full plots/analysis come on Day 12)
    robustness_bench(model, d["X_test"], d["y_test"], device).to_csv(
        f"results/{a.name}_robustness.csv", index=False)


if __name__ == "__main__":
    main()
