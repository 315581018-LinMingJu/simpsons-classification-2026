"""偏心小角色與明亮紋理背景，保留角色像素及完整取景。"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from v4.geometry import seeded
from v5.background import exterior_mask, texture
from v8.data import open_rgb

SIZE = 224
SCENE_PROBABILITY = 0.40
SPAN_RANGE = (0.44, 0.66)
MIN_VISIBLE_FRACTION = 0.065
MIN_VISIBLE_EDGE = 36
MIN_OFFSET_FRACTION = 0.14
MARGIN_RANGE = (0.04, 0.06)
CUE_RGB_THRESHOLD = 32


@dataclass(frozen=True)
class Render:
    image: Image.Image
    reason: str
    span_fraction: float = 0.0
    visible_fraction: float = 0.0
    offset_fraction: float = 0.0
    cue_box: tuple | None = None


def bright_background(targets, rng):
    # 程序生成的斜向光影和多尺度紋理。只改背景，保留前景色彩。
    base = Image.fromarray(texture((SIZE, SIZE), rng, held_out=False)).filter(ImageFilter.GaussianBlur(0.75))
    canvas = np.asarray(base, dtype=np.float32) / 255
    y, x = np.mgrid[-1:1:complex(0, SIZE), -1:1:complex(0, SIZE)]
    angle = rng.uniform(0, 2 * np.pi)
    axis = x * np.cos(angle) + y * np.sin(angle)
    intervals = rng.uniform(1.4, 3.8)
    patterned = (rng.uniform(0.05, 0.13) * np.tanh(4 * np.sin(axis * intervals + rng.uniform(-np.pi, np.pi)))
                 + rng.uniform(0.05, 0.14) * axis)
    canvas += patterned[:, :, None]
    luminance = canvas.mean(axis=2)
    contrast = np.clip(targets.contrast * rng.uniform(0.80, 1.10), 0.10, 0.22)
    canvas = ((canvas - luminance.mean()) * (contrast / max(float(luminance.std()), 0.01))
              + np.clip(targets.brightness + rng.uniform(0.015, 0.105), 0.42, 0.65))
    canvas += rng.uniform(-0.08, 0.08, size=3)
    grains = np.clip(targets.fine_detail * rng.uniform(0.18, 0.40), 0.004, 0.012)
    canvas += rng.normal(0, grains, (SIZE, SIZE, 1))
    return Image.fromarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8))


def render_scene(image, targets, seed, item):
    rng = np.random.default_rng(seeded(seed, f"v10:placement:{item}").getrandbits(64))
    rgb = np.asarray(image)
    rem = exterior_mask(rgb)
    if not rem.any():
        return Render(image, "uncertain_matte")
    # 黑底遮罩縮一像素，保留靠近前景的深色輪廓。
    matte = Image.fromarray(rem.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
    alpha = np.asarray(matte) == 0
    cues = alpha & (rgb.max(axis=2) > CUE_RGB_THRESHOLD)
    alpha_img = Image.fromarray(alpha.astype(np.uint8) * 255)
    cue_img = Image.fromarray(cues.astype(np.uint8) * 255)
    box = alpha_img.getbbox()
    if box is None or cues.sum() == 0:
        return Render(image, "insufficient_cues")
    width, height = box[2] - box[0], box[3] - box[1]
    longest = max(width, height)
    min_span = max(longest * np.sqrt(MIN_VISIBLE_FRACTION * SIZE**2 / cues.sum()) * 1.02,
                   MIN_VISIBLE_EDGE * longest / min(width, height))
    span = max(round(SIZE * rng.uniform(*SPAN_RANGE)), int(np.ceil(min_span)))
    if span > round(SIZE * SPAN_RANGE[1]):
        return Render(image, "insufficient_cues")

    shape = (round(width * span / longest), round(height * span / longest))
    cue_mask = cue_img.crop(box).resize(shape, Image.Resampling.NEAREST)
    cue_box = cue_mask.getbbox()
    visible = float((np.asarray(cue_mask) > 0).sum() / SIZE**2)
    if (cue_box is None or visible < MIN_VISIBLE_FRACTION
            or min(cue_box[2] - cue_box[0], cue_box[3] - cue_box[1]) < MIN_VISIBLE_EDGE):
        return Render(image, "insufficient_cues")
    gap_x, gap_y = SIZE - shape[0], SIZE - shape[1]
    mx, my = (round(SIZE * value) for value in rng.uniform(*MARGIN_RANGE, 2))
    x, y = (mx if rng.integers(0, 2) == 0 else gap_x - mx,
            my if rng.integers(0, 2) == 0 else gap_y - my)
    cue_box = (x + cue_box[0], y + cue_box[1], x + cue_box[2], y + cue_box[3])
    cx, cy = (cue_box[0] + cue_box[2]) / 2, (cue_box[1] + cue_box[3]) / 2
    offset = float(np.hypot(cx - SIZE / 2, cy - SIZE / 2) / SIZE)
    if offset < MIN_OFFSET_FRACTION:
        return Render(image, "too_centered")

    foreground = image.crop(box)
    foreground.putalpha(alpha_img.crop(box))
    foreground = foreground.resize(shape, Image.Resampling.BILINEAR)
    background = bright_background(targets, rng)
    background.paste(foreground, (x, y), foreground.getchannel("A"))
    return Render(background, "accepted", span / SIZE, visible, offset, cue_box)


class TrainingImages:
    def __init__(self, root, cases, base, normal, targets, seed):
        self.root, self.cases, self.base, self.normal = root, cases, base, normal
        self.targets, self.seed = targets, seed

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        if seeded(self.seed, f"v10:apply:{case.item}").random() < SCENE_PROBABILITY:
            rendered = render_scene(open_rgb(self.root / case.item), self.targets, self.seed, case.item)
            if rendered.reason == "accepted":
                # 保留整張場景，避免二次裁切或擦除小角色。
                return self.normal(rendered.image), case.label
        return self.base[index]
