"""Loading GTSRB, leakage-free (track-based) split, and a PyTorch Dataset.

Expected folder (Kaggle mirror: meowmeowmeowmeowmeow/gtsrb-german-traffic-sign):
    <root>/Train.csv, <root>/Test.csv, <root>/Train/<class>/*.png, <root>/Test/*.png
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


def add_track_group(df):
    """Add a 'group' column = '<class>_<track>'.

    Filenames look like 00020_00003_00017.png -> class 20, track 3, frame 17
    (older 2-part names '00003_00017.png' = track 3, frame 17 also work).
    All frames of one physical sign share a group, so they must stay together.
    """
    stems = df["Path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    track = stems.apply(lambda s: s.split("_")[-2])
    df = df.copy()
    df["track"] = track
    df["group"] = df["ClassId"].astype(str) + "_" + track
    return df


def track_split(df, val_size=0.2, seed=42):
    """Split by group (track), never by individual image."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, df["ClassId"], groups=df["group"]))
    # Safety check: no physical sign appears on both sides
    assert not (set(df.iloc[tr_idx]["group"]) & set(df.iloc[va_idx]["group"])), "LEAKAGE!"
    return tr_idx, va_idx


def load_images(root, df, size=32):
    """Read images, crop to the sign using the ROI columns, resize, return uint8 RGB."""
    has_roi = all(c in df.columns for c in ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"])
    out = np.zeros((len(df), size, size, 3), np.uint8)
    for i, row in enumerate(df.to_dict("records")):
        img = cv2.imread(os.path.join(root, row["Path"].replace("\\", "/")))
        if img is None:
            raise FileNotFoundError(row["Path"])
        if has_roi:
            x1, y1, x2, y2 = (int(row[c]) for c in ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"])
            img = img[y1:y2 + 1, x1:x2 + 1]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (i + 1) % 10000 == 0:
            print(f"  loaded {i + 1}/{len(df)}")
    return out


def prepare_data(root, size=32, val_size=0.2, seed=42, cache_dir="cache"):
    """Return dict with X_train/y_train/X_val/y_val/X_test/y_test (uint8 images).

    The first call reads all PNGs (a few minutes); later calls load a cached .npz.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"gtsrb_{size}_seed{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    train_df = add_track_group(pd.read_csv(os.path.join(root, "Train.csv")))
    test_df = pd.read_csv(os.path.join(root, "Test.csv"))
    tr_idx, va_idx = track_split(train_df, val_size, seed)

    print("Loading train images...")
    X_all = load_images(root, train_df, size)
    y_all = train_df["ClassId"].values.astype(np.int64)
    print("Loading test images...")
    X_test = load_images(root, test_df, size)

    data = {
        "X_train": X_all[tr_idx], "y_train": y_all[tr_idx],
        "X_val": X_all[va_idx], "y_val": y_all[va_idx],
        "X_test": X_test, "y_test": test_df["ClassId"].values.astype(np.int64),
        "groups_train": train_df["group"].values[tr_idx].astype(str),
        "groups_val": train_df["group"].values[va_idx].astype(str),
    }
    np.savez_compressed(cache, **data)
    return data


MEAN, STD = 0.5, 0.5


def to_tensor(X):
    """uint8 (N,H,W,3) -> normalised float tensor (N,3,H,W)."""
    t = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


class GTSRBDataset(Dataset):
    """Holds uint8 images in memory. `transform` (optional) works on one uint8 HxWx3 image,
    which is where label-safe augmentation will go on Day 12."""

    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X, y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        return to_tensor(img[None])[0], int(self.y[i])
