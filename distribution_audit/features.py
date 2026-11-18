"""從影像內容量測可重現的低階特徵。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from v5.background import DARK_MAX

FEATURES = (
    "black_edge_fraction", "black_outer_band_fraction", "weak_color_fraction",
    "center_weak_color_fraction", "brightness", "contrast", "fine_detail", "edge_luma_std",
)
THUMBNAIL_SIZE = 192
IMAGE_ERRORS = (OSError, SyntaxError, ValueError, Image.DecompressionBombError)


@dataclass(frozen=True)
class ImageReading:
    path: Path
    actual_format: str | None
    width: int | None
    height: int | None
    extension_match: bool | None
    values: tuple | None
    error: str | None = None


def quantiles(values, points):
    return [float(x) for x in np.quantile(values, points)]


def visual_features(rgb):
    native = np.asarray(rgb)
    height, width = native.shape[:2]
    outer = np.zeros((height, width), dtype=bool)
    outer[0, :] = outer[-1, :] = outer[:, 0] = outer[:, -1] = True
    ring = native[outer]
    black_edge = (ring.max(axis=1) <= DARK_MAX).mean()
    band_x, band_y = max(1, round(width * 0.05)), max(1, round(height * 0.05))
    outer[:band_y, :] = outer[-band_y:, :] = outer[:, :band_x] = outer[:, -band_x:] = True
    black_band = (native[outer].max(axis=1) <= DARK_MAX).mean()

    thumb = rgb.copy()
    thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    colors = np.asarray(thumb, dtype=np.int16)
    low_color = np.ptp(colors, axis=2) <= 5
    h, w = low_color.shape
    central = low_color[h // 4:h - h // 4, w // 4:w - w // 4]
    gray = np.asarray(thumb.convert("L"), dtype=np.float32) / 255
    edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    detail = 0.0
    if min(gray.shape) >= 3:
        laplacian = 4 * gray[1:-1, 1:-1] - (
            gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        )
        detail = np.median(np.abs(laplacian))
    return tuple(float(value) for value in (
        black_edge, black_band, low_color.mean(), central.mean(), gray.mean(), gray.std(),
        detail, edge.std(),
    ))


def inspect_image(path):
    fmt, width, height, match = None, None, None, None
    try:
        with Image.open(path) as image:
            fmt = image.format
            width, height = image.size
            match = Image.registered_extensions().get(path.suffix.lower()) == fmt
            values = visual_features(image.convert("RGB"))
        return ImageReading(path, fmt, width, height, match, values)
    except IMAGE_ERRORS as exc:
        return ImageReading(path, fmt, width, height, match, None, f"{type(exc).__name__}: {exc}")
