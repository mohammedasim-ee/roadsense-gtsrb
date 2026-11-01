"""Robustness test bench: 7 corruptions x 3 severities.

Everyone on the team must use THESE functions so results are comparable.
Input/output: uint8 array of shape (N, H, W, 3), RGB, values 0-255.
Same `seed` -> same corrupted images, for every model.
"""
import cv2
import numpy as np

SEVERITIES = (1, 2, 3)


def gaussian_noise(imgs, severity, seed=0):
    sigma = (8, 16, 32)[severity - 1]  # std-dev on the 0-255 scale
    rng = np.random.default_rng(seed)
    out = imgs.astype(np.float32) + rng.normal(0, sigma, imgs.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(imgs, severity, seed=0):
    sigma = (0.6, 1.0, 1.5)[severity - 1]
    return np.stack([cv2.GaussianBlur(im, (0, 0), sigma) for im in imgs])


def motion_blur(imgs, severity, seed=0):
    k = (3, 5, 7)[severity - 1]  # blur length in pixels
    rng = np.random.default_rng(seed)
    out = []
    for im in imgs:
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0
        angle = rng.uniform(0, 180)
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        out.append(cv2.filter2D(im, -1, kernel))
    return np.stack(out)


def brightness(imgs, severity, seed=0):
    """Make each image darker or brighter (random sign per image)."""
    delta = (40, 80, 120)[severity - 1]
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], size=(len(imgs), 1, 1, 1))
    out = imgs.astype(np.float32) + sign * delta
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast(imgs, severity, seed=0):
    factor = (0.6, 0.4, 0.2)[severity - 1]  # lower = flatter, washed-out image
    f = imgs.astype(np.float32)
    mean = f.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip(mean + (f - mean) * factor, 0, 255).astype(np.uint8)


def occlusion(imgs, severity, seed=0):
    """Paste a grey square at a random place (sticker, dirt, branch...)."""
    frac = (0.2, 0.35, 0.5)[severity - 1]  # patch side as a fraction of image side
    rng = np.random.default_rng(seed)
    out = imgs.copy()
    n, h, w, _ = imgs.shape
    ph, pw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    for i in range(n):
        y = rng.integers(0, h - ph + 1)
        x = rng.integers(0, w - pw + 1)
        out[i, y:y + ph, x:x + pw, :] = rng.integers(0, 256)
    return out


def jpeg(imgs, severity, seed=0):
    quality = (30, 15, 5)[severity - 1]  # lower = more compression artefacts
    out = []
    for im in imgs:
        bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "contrast": contrast,
    "occlusion": occlusion,
    "jpeg": jpeg,
}


def apply_corruption(imgs, name, severity, seed=0):
    assert imgs.dtype == np.uint8 and imgs.ndim == 4, "expected uint8 (N,H,W,3)"
    return CORRUPTIONS[name](imgs, severity, seed)
