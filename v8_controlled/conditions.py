"""保護完整原圖塊，分別變動外框、背景及固定視角的聚合。"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from v3.inference import SCALES
from v4.geometry import seeded
from v7.impulse import clean_impulses

CANVAS = 280
SPAN = 176
RIM = 12
GUARD = 3
NEUTRAL = 112
BRIGHT = 174
BLACK = 0
DETAIL = 22
RETAIN = .98
ARMS = ("neutral", "black_rim", "bright_background", "outer_detail", "offcenter")
CORNER_NAMES = ("tl", "tr", "bl", "br", "center")
VIEW_NAMES = tuple(f"{scale}_{name}" for scale in SCALES for name in CORNER_NAMES)


@dataclass(frozen=True)
class Example:
    frames: dict
    bounds: dict
    retention: np.ndarray
    cue_retention: np.ndarray | None
    valid: bool
    invalid_reason: str | None


def geometry(image, origin=None):
    ratio = SPAN / max(image.size)
    shape = tuple(max(1, round(side * ratio)) for side in image.size)
    if origin is None:
        x, y = ((CANVAS - side) // 2 for side in shape)
    else:
        x, y = (8 if flag else CANVAS - side - 8 for side, flag in zip(shape, origin, strict=True))
    resized = image.resize(shape, Image.Resampling.BILINEAR)
    box = (x, y, x + shape[0], y + shape[1])
    return resized, box


def controlled_frames(image, seed, item):
    patch, box = geometry(image)
    rng = seeded(seed, f"v8controlled:position:{item}")
    off_patch, off_box = geometry(image, (rng.choice((True, False)), rng.choice((True, False))))
    canvas = Image.new("RGB", (CANVAS, CANVAS), (NEUTRAL,) * 3)
    canvas.paste(patch, box[:2])
    neutral = np.asarray(canvas)
    rim = np.ones((CANVAS, CANVAS), dtype=bool)
    rim[RIM:-RIM, RIM:-RIM] = False
    background = ~rim
    x0, y0, x1, y1 = box
    background[y0-GUARD:y1+GUARD, x0-GUARD:x1+GUARD] = False

    black = neutral.copy()
    black[rim] = BLACK
    bright = neutral.copy()
    bright[background] = BRIGHT
    detail = neutral.copy()
    # 小尺度連續棋格。正負像素精確配平，保留相同平均亮度。
    gx, gy = np.meshgrid(np.arange(CANVAS), np.arange(CANVAS))
    stripes = ((gx // 3 + gy // 3 + rng.randrange(2)) % 2).astype(bool)
    plus = np.flatnonzero(stripes & background)
    minus = np.flatnonzero(~stripes & background)
    n = min(len(plus), len(minus))
    detail.reshape(-1, 3)[plus[:n]] = NEUTRAL + DETAIL
    detail.reshape(-1, 3)[minus[:n]] = NEUTRAL - DETAIL
    offcenter = Image.new("RGB", canvas.size, (NEUTRAL,) * 3)
    offcenter.paste(off_patch, off_box[:2])
    raw = {"neutral": canvas, "black_rim": Image.fromarray(black), "bright_background": Image.fromarray(bright),
           "outer_detail": Image.fromarray(detail), "offcenter": offcenter}
    filtered = {name: clean_impulses(frame)[0] for name, frame in raw.items()}
    base = np.asarray(filtered["neutral"])
    edited = {"black_rim": rim, "bright_background": background, "outer_detail": background}
    for name, region in edited.items():
        if not np.array_equal(np.asarray(filtered[name])[~region], base[~region]):
            return Example(filtered, {}, np.empty(0), None, False, f"denoise_changes_outside_{name}")
        if np.array_equal(np.asarray(filtered[name])[region], base[region]):
            return Example(filtered, {}, np.empty(0), None, False, f"empty_treatment_{name}")
    if np.asarray(filtered["outer_detail"])[background].sum() != base[background].sum():
        return Example(filtered, {}, np.empty(0), None, False, "detail_luminance_changed")
    # 固定前景原塊的長寬、可見位置以及其中非近黑線索的位置。
    patch_mask = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    x0, y0, x1, y1 = off_box
    patch_mask[y0:y1, x0:x1] = 255
    cue_mask = np.zeros_like(patch_mask)
    cue_mask[y0:y1, x0:x1] = (np.asarray(off_patch).max(axis=2) > 32).astype(np.uint8) * 255
    return Example(filtered, {"neutral": box, "offcenter": off_box}, crop_retention(patch_mask),
                   crop_retention(cue_mask) if cue_mask.any() else None, True, None)


def crop_boxes(width, height):
    dx, dy = width - 224, height - 224
    for x, y in ((0, 0), (dx, 0), (0, dy), (dx, dy), (round(dx / 2), round(dy / 2))):
        yield x, y, x + 224, y + 224


def crop_retention(mask):
    # V8 在正方形畫布的兩個尺度與 FiveCrop 次序。
    result = []
    for scale in SCALES:
        full = np.asarray(Image.fromarray(mask).resize((scale, scale), Image.Resampling.BILINEAR), dtype=np.float64)
        mass = full.sum()
        if mass <= 0:
            raise ValueError("Cannot measure the area of an empty patch")
        result.extend(float(full[y0:y1, x0:x1].sum() / mass) for x0, y0, x1, y1 in crop_boxes(scale, scale))
    return np.asarray(result)


def kept_views(retention):
    values = np.asarray(retention)
    selected = values >= RETAIN
    # 預先固定閾值。全都低於閾值時，使用保留量最大的視角。
    if not selected.any():
        selected[values.argmax()] = True
    return selected


def blind_views(selected, seed, item):
    # 每個尺度保留一樣多的視角，只移除使用原圖塊位置的資訊。
    control = np.zeros_like(selected)
    for i, scale in enumerate(SCALES):
        options = range(i * len(CORNER_NAMES), (i + 1) * len(CORNER_NAMES))
        count = int(selected[list(options)].sum())
        rng = seeded(seed, f"v8controlled:blind:{item}:{scale}")
        control[rng.sample(options, count)] = True
    return control
