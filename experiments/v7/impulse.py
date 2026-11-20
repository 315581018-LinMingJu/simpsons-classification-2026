"""固定、保守的孤立顆粒處理與有標籤配對驗證。"""

import numpy as np
from PIL import Image, ImageFilter

from v4.geometry import seeded

MIN_TRIGGER_FRACTION = 0.003
CONDITIONS = ("clean", "texture_control", "textured_impulse")


def impulse_mask(image):
    rgb = np.asarray(image, dtype=np.int16)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    if min(rgb.shape[:2]) < 3:
        return mask
    center = rgb[1:-1, 1:-1]
    median = np.asarray(image.filter(ImageFilter.MedianFilter(3)), dtype=np.int16)[1:-1, 1:-1]
    deviations = np.abs(center - median)
    neighbor_count = np.zeros(center.shape[:2], dtype=np.uint8)
    away_from_center = np.ones_like(neighbor_count, dtype=bool)
    h, w = rgb.shape[:2]
    for dy, dx in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
        adjacent = rgb[dy:h - 2 + dy, dx:w - 2 + dx]
        neighbor_count += (np.abs(adjacent - median).max(axis=2) <= 32)
        away_from_center &= np.abs(adjacent - center).max(axis=2) >= 27
    mask[1:-1, 1:-1] = ((deviations.max(axis=2) >= 58) & ((deviations >= 30).sum(axis=2) >= 2)
                         & away_from_center & (neighbor_count >= 5))
    return mask


def clean_impulses(image):
    mask = impulse_mask(image)
    fraction = float(mask.mean())
    if fraction < MIN_TRIGGER_FRACTION:
        return image, fraction
    data = np.array(image)
    median = np.asarray(image.filter(ImageFilter.MedianFilter(3)))
    data[mask] = median[mask]
    return Image.fromarray(data), fraction


def add_impulses(image, seed, item):
    rng = np.random.default_rng(seeded(seed, f"v7:impulses:{item}").getrandbits(64))
    pixels = np.array(image)
    height, width = pixels.shape[:2]
    # 局部顆粒分布不依賴已知角色位置，保留原圖標籤與幾何。
    size_x, size_y = round(width * rng.uniform(0.54, 0.83)), round(height * rng.uniform(0.54, 0.83))
    x = rng.integers(0, width - size_x + 1)
    y = rng.integers(0, height - size_y + 1)
    region = pixels[y:y + size_y, x:x + size_x]
    selected = rng.random(region.shape[:2]) < rng.uniform(0.12, 0.25)
    region[selected] = rng.integers(0, 256, (int(selected.sum()), 3), dtype=np.uint8)
    return Image.fromarray(pixels)
