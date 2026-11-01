"""Metrics, confusion matrix, most-confused pairs, and the robustness bench."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from .dataset import to_tensor


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    """X: uint8 (N,H,W,3) -> predicted class ids."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = to_tensor(X[i:i + batch_size]).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def clean_metrics(model, X, y, device):
    p = predict(model, X, device)
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}, p


def top_confused_pairs(y_true, y_pred, class_names, k=10):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm, axis=None), cm.shape))[0][:k]
    return pd.DataFrame([(class_names[t], class_names[p], int(cm[t, p])) for t, p in idx],
                        columns=["true", "predicted_as", "count"])


def robustness_bench(model, X_test, y_test, device, seed=0):
    """Accuracy for every corruption x severity (same seed => same images for every model)."""
    rows = [("clean", 0, accuracy_score(y_test, predict(model, X_test, device)))]
    for name in CORRUPTIONS:
        for sev in SEVERITIES:
            Xc = apply_corruption(X_test, name, sev, seed)
            rows.append((name, sev, accuracy_score(y_test, predict(model, Xc, device))))
    df = pd.DataFrame(rows, columns=["corruption", "severity", "accuracy"])
    mca = df[df.corruption != "clean"].accuracy.mean()
    print(f"Clean acc: {rows[0][2]:.4f} | mean corruption acc: {mca:.4f}")
    return df
