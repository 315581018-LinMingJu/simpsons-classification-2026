"""依完整分布稽核的亮度、對比與細節統計合成場景。"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from v5.background import texture
from v5.challenge import SIZE, inverse_perspective, panel, primary_geometry, warp


@dataclass(frozen=True)
class Targets:
    brightness: float
    contrast: float
    fine_detail: float

    @classmethod
    def from_summary(cls, summary):
        medians = summary["test"]["feature_medians"]
        values = [float(medians[name]) for name in ("brightness", "contrast", "fine_detail")]
        if any(not np.isfinite(value) or not 0 < value < 1 for value in values):
            raise ValueError("Audit medians must be finite fractions between 0 and 1")
        return cls(*values)


def training_project(size, rng):
    width, height = size
    corners = np.array(((0, 0), (width, 0), (width, height), (0, height)), dtype=float)
    corners -= np.array((width / 2, height / 2))
    corners += rng.uniform(-0.045, 0.045, (4, 2)) * min(size)
    angle = np.deg2rad(rng.uniform(-18, 18))
    sine, cosine = np.sin(angle), np.cos(angle)
    corners = corners @ np.array(((cosine, sine), (-sine, cosine)))
    corners -= corners.min(axis=0)
    span = corners.max(axis=0)
    corners *= SIZE * rng.uniform(0.48, 0.82) / span.max()
    corners += (SIZE - corners.max(axis=0)) * rng.uniform(0.08, 0.92, 2)
    return inverse_perspective(corners, size)


def composite(image, targets, rng, *, held_out=False):
    source = panel(image)
    project = primary_geometry(source.size, rng)[0] if held_out else training_project(source.size, rng)
    foreground = warp(source, project)
    alpha = foreground.getchannel("A")

    backdrop = Image.fromarray(texture((SIZE, SIZE), rng, held_out)).filter(ImageFilter.GaussianBlur(1.6))
    background = np.array(backdrop, dtype=np.float32)
    background -= background.mean(axis=(0, 1))
    level = max(float(background[:, :, 1].std()), 1)
    background *= np.clip(0.85 * targets.contrast * 255 / level, 0.7, 2.8)
    brightness = targets.brightness + rng.uniform(-0.065, 0.065)
    background += brightness * 255 + rng.uniform(-22, 22, 3)
    sigma = np.clip(targets.fine_detail * 255 * 0.42, 1.0, 7.0) * rng.uniform(0.75, 1.35)
    # 顆粒只加入合成背景，避免把輪廓內的角色辨識線索破壞。
    background += rng.normal(0, sigma, (SIZE, SIZE, 1))
    canvas = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    canvas.paste(foreground, mask=alpha)

    luma = np.asarray(canvas.convert("L"), dtype=np.float32)
    mean = float(luma.mean())
    gain = float(np.clip(targets.contrast * 255 / max(float(luma.std()), 1), 0.70, 1.20))
    shift = float(np.clip(brightness * 255 - mean, -36, 36))
    frame = (np.asarray(canvas, dtype=np.float32) - mean) * gain + mean + shift
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)), float(np.asarray(alpha).mean() / 255)
