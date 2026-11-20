"""同圖配對的幾何干擾與原始中心裁切保留率。"""

import hashlib
import random
from collections import defaultdict

from PIL import Image

CONDITIONS = ("clean", "center", "offcenter")
METHODS = ("baseline", "full_resize", "multiscale")


def seeded(seed, key):
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big"))


def stratified_samples(samples, targets, per_class, seed, class_count):
    by_class = defaultdict(list)
    for path, item in samples:
        by_class[targets[item]].append((path, item))
    if set(by_class) != set(range(class_count)):
        raise ValueError("At least one deduplicated validation image is required per class")
    return [
        sample
        for label, group in sorted(by_class.items())
        for sample in seeded(seed, f"class:{label}").sample(sorted(group), min(per_class, len(group)))
    ]


def layout(image_size, aspect_ratios, seed, item):
    rng = seeded(seed, item)
    ratio = min(1.5, max(2 / 3, rng.choice(aspect_ratios)))
    size = (round(448 * ratio), 448) if ratio >= 1 else (448, round(448 / ratio))
    fraction = rng.uniform(0.54, 0.72)
    scale = min(canvas * fraction / axis for canvas, axis in zip(size, image_size, strict=True))
    patch = tuple(max(1, round(axis * scale)) for axis in image_size)
    gaps = tuple(canvas - label for canvas, label in zip(size, patch, strict=True))
    center = tuple(gap // 2 for gap in gaps)
    offcenter = tuple(round(gap * rng.choice((0.04, 0.96))) for gap in gaps)
    shade = rng.randrange(90, 171)
    return size, patch, center, offcenter, (shade, shade, shade)


def render(image, condition, aspect_ratios, seed, item):
    if condition == "clean":
        return image
    size, patch, center, offcenter, fill = layout(image.size, aspect_ratios, seed, item)
    result = Image.new("RGB", size, fill)
    position = {"center": center, "offcenter": offcenter}[condition]
    result.paste(image.resize(patch, Image.Resampling.BILINEAR), position)
    return result


def retained_fraction(size, patch, position):
    # 換算 Resize(256)／CenterCrop(224) 在原畫布中的裁切區域。
    resized = tuple(int(256 * axis / min(size)) for axis in size)
    fraction = 1.0
    for canvas, axis, start, length in zip(size, resized, position, patch, strict=True):
        crop_start = round((axis - 224) / 2) * canvas / axis
        crop_end = crop_start + 224 * canvas / axis
        overlap = max(0.0, min(start + length, crop_end) - max(start, crop_start))
        fraction *= overlap / length
    return fraction


class ChallengeImages:
    def __init__(self, samples, condition, aspects, seed, transform):
        self.samples = samples
        self.condition = condition
        self.aspects = aspects
        self.seed = seed
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item = self.samples[index]
        with Image.open(path) as image:
            canvas = render(image.convert("RGB"), self.condition, self.aspects, self.seed, item)
        return self.transform(canvas), item
