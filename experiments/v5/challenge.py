"""以現有 Train／Val 合成同圖配對的壓力測試。"""

import numpy as np
from PIL import Image, ImageOps

from v4.geometry import seeded
from v5.background import exterior_mask, texture

SIZE = 264
PARENTS = {
    "clean": None, "geometry": "clean", "context": "geometry",
    "clipped": "context", "distractor": "context", "negative": "context",
}


def panel(image):
    alpha = Image.fromarray((~exterior_mask(np.asarray(image))).astype(np.uint8) * 255)
    cutout = image.copy()
    cutout.putalpha(alpha)
    return cutout.crop(alpha.getbbox())


def inverse_perspective(corners, source_size):
    width, height = source_size
    src = ((0, 0), (width, 0), (width, height), (0, height))
    rows, values = [], []
    for (x, y), (u, v) in zip(corners, src, strict=True):
        rows.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y),
                     (0, 0, 0, x, y, 1, -v * x, -v * y)))
        values.extend((u, v))
    return tuple(np.linalg.solve(rows, values))


def primary_geometry(size, rng):
    width, height = size
    corners = np.array(((0, 0), (width, 0), (width, height), (0, height)), dtype=float)
    corners -= np.array((width / 2, height / 2))
    corners += rng.uniform(-0.08, 0.08, (4, 2)) * min(size)
    angle = rng.uniform(12, 30) * rng.choice((-1, 1))
    sine, cosine = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    corners = corners @ np.array(((cosine, sine), (-sine, cosine)))
    corners -= corners.min(axis=0)
    span = corners.max(axis=0)
    span_fraction = rng.uniform(0.40, 0.70)
    corners *= SIZE * span_fraction / span.max()
    width, height = corners.max(axis=0)
    left, top = (bool(rng.integers(0, 2)), bool(rng.integers(0, 2)))
    margin_x, margin_y = rng.uniform(0.04, 0.09, 2) * SIZE
    corners += (margin_x if left else SIZE - margin_x - width,
                margin_y if top else SIZE - margin_y - height)
    return inverse_perspective(corners, size), left, top, angle, span_fraction


def clip_panel(source, rng):
    alpha = np.array(source.getchannel("A"))
    fraction = rng.uniform(0.10, 0.17)
    edge = int(rng.integers(0, 4))
    if edge < 2:
        strip = min(alpha.shape[1] - 1, max(1, round(alpha.shape[1] * fraction)))
        if not strip:
            return source
        if edge == 0:
            alpha[:, :strip] = 0
        else:
            alpha[:, -strip:] = 0
    else:
        strip = min(alpha.shape[0] - 1, max(1, round(alpha.shape[0] * fraction)))
        if not strip:
            return source
        if edge == 2:
            alpha[:strip, :] = 0
        else:
            alpha[-strip:, :] = 0
    clipped = source.copy()
    clipped.putalpha(Image.fromarray(alpha))
    return clipped


def warp(source, project):
    return source.transform((SIZE, SIZE), Image.Transform.PERSPECTIVE, project,
                            Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))


def scene(image, condition, seed, item, distractor=None):
    if condition == "clean":
        return image, {}
    if condition not in PARENTS:
        raise ValueError(f"Unknown challenge condition: {condition}")
    source = panel(image)
    key = str(item)
    rng = np.random.default_rng(seeded(seed, key).getrandbits(64))
    project, left, top, angle, fraction = primary_geometry(source.size, rng)
    if condition == "clipped":
        source = clip_panel(source, rng)
    elif condition == "negative":
        alpha = source.getchannel("A")
        source = ImageOps.invert(source.convert("RGB"))
        source.putalpha(alpha)
    foreground = warp(source, project)
    matte = foreground.getchannel("A")
    canvas = Image.new("RGB", (SIZE, SIZE))
    if condition != "geometry":
        # 交替使用接縫與多尺度紋理，不從測試圖擷取任何像素。
        background_rng = np.random.default_rng(seeded(seed, key + ":background").getrandbits(64))
        held_out = bool(background_rng.integers(0, 2))
        canvas = Image.fromarray(texture(canvas.size, background_rng, held_out))
    secondary_area = 0.0
    if condition == "distractor":
        if distractor is None:
            raise ValueError("A training image of a different class is required for distractor")
        margin = round(SIZE * 0.06)
        x0, y0, x1, y1 = matte.getbbox()
        free_width = SIZE - margin - x1 if left else x0 - margin
        free_height = SIZE - margin - y1 if top else y0 - margin
        side = min(round(SIZE * 0.21), max(free_width, free_height))
        secondary = ImageOps.contain(panel(distractor), (side, side))
        x = SIZE - margin - secondary.width if left else margin
        y = SIZE - margin - secondary.height if top else margin
        if np.asarray(matte)[y:y + secondary.height, x:x + secondary.width].any():
            raise ValueError("Distractor overlaps the primary image")
        secondary_mask = secondary.getchannel("A")
        canvas.paste(secondary, (x, y), secondary_mask)
        secondary_area = float(np.asarray(secondary_mask).sum() / (255 * SIZE**2))
    canvas.paste(foreground, mask=matte)
    return canvas, {
        "primary_alpha_fraction": float(np.asarray(matte).mean() / 255),
        "secondary_alpha_fraction": secondary_area, "rotation_deg": float(angle),
        "max_span_fraction": float(fraction),
    }


class ChallengeImages:
    def __init__(self, samples, condition, transform, seed, distractors):
        self.samples = samples
        self.condition = condition
        self.transform = transform
        self.seed = seed
        self.distractors = distractors

    def __len__(self):
        return len(self.samples)

    def case(self, index):
        path, item = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        secondary = None
        if self.condition == "distractor":
            with Image.open(self.distractors[item].path) as source:
                secondary = source.convert("RGB")
        return scene(image, self.condition, self.seed, item, secondary)

    def __getitem__(self, index):
        image, _ = self.case(index)
        return self.transform(image), self.samples[index][1]
