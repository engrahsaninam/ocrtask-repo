from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _suppress_colored_ruling(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    blue_rule = (blue > red + 18) & (green > red + 4) & (blue > 90)
    red_margin = (red > green + 25) & (red > blue + 25) & (red > 100)
    mask = blue_rule | red_margin
    cleaned = rgb.copy()
    cleaned[mask] = 255
    return Image.fromarray(np.clip(cleaned, 0, 255).astype(np.uint8), mode="RGB")


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_back = 0.0
    weight_back = 0.0
    best_var = -1.0
    threshold = 127
    for value in range(256):
        weight_back += hist[value]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += value * hist[value]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        var_between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if var_between > best_var:
            best_var = var_between
            threshold = value
    return threshold


def _estimate_skew_angle(gray: Image.Image) -> float:
    best_angle = 0.0
    best_score = -1.0
    for angle in range(-10, 11, 2):
        rotated = gray.rotate(angle, expand=True, fillcolor=255)
        arr = np.asarray(rotated, dtype=np.uint8)
        threshold = _otsu_threshold(arr)
        ink = arr <= threshold
        projection = ink.sum(axis=1)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def _adaptive_threshold(gray: np.ndarray, block_size: int = 35, offset: int = 12) -> np.ndarray:
    pad = block_size // 2
    padded = np.pad(gray.astype(np.float32), pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = gray.shape
    y0 = np.arange(h)
    y1 = y0 + block_size
    x0 = np.arange(w)
    x1 = x0 + block_size
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    local_mean = sums / (block_size * block_size)
    return np.where(gray < local_mean - offset, 0, 255).astype(np.uint8)


def _remove_straight_lines(binary: np.ndarray) -> np.ndarray:
    cleaned = binary.copy()
    ink = cleaned == 0
    h, w = ink.shape

    for y in np.where(ink.mean(axis=1) > 0.35)[0]:
        cleaned[max(0, y - 1):min(h, y + 2), :] = 255
    for x in np.where(ink.mean(axis=0) > 0.35)[0]:
        cleaned[:, max(0, x - 1):min(w, x + 2)] = 255

    for y in range(h):
        xs = np.flatnonzero(cleaned[y] == 0)
        for run in np.split(xs, np.where(np.diff(xs) > 1)[0] + 1) if xs.size else []:
            if run.size > w * 0.22:
                cleaned[y, run] = 255

    for x in range(w):
        ys = np.flatnonzero(cleaned[:, x] == 0)
        for run in np.split(ys, np.where(np.diff(ys) > 1)[0] + 1) if ys.size else []:
            if run.size > h * 0.18:
                cleaned[run, x] = 255
    return cleaned


def _normalize_strokes(binary: Image.Image) -> Image.Image:
    return binary.filter(ImageFilter.MedianFilter(size=3))


def preprocess_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    image = _suppress_colored_ruling(Image.open(path))
    gray_image = image.convert("L")
    gray_image = ImageEnhance.Contrast(gray_image).enhance(2.2)
    gray_image = gray_image.filter(ImageFilter.MedianFilter(size=3))

    angle = _estimate_skew_angle(gray_image)
    if abs(angle) >= 1:
        gray_image = gray_image.rotate(angle, expand=True, fillcolor=255)

    gray = np.asarray(gray_image, dtype=np.uint8)
    binary = _remove_straight_lines(_adaptive_threshold(gray))
    normalized = _normalize_strokes(Image.fromarray(binary, mode="L"))
    return np.asarray(normalized, dtype=np.uint8)
