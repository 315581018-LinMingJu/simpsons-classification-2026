"""用程序紋理替換保守偵測到的外圍黑底。"""

import numpy as np
from PIL import Image, ImageDraw, ImageOps

DARK_MAX = 24
MIN_COVERAGE, MAX_COVERAGE = 0.07, 0.70


def exterior_mask(rgb):
    dark = rgb.max(axis=2) <= DARK_MAX
    if dark.mean() < MIN_COVERAGE:
        return np.zeros_like(dark)
    height, width = dark.shape
    # 在小圖上連通外圍，只把原圖仍接近黑色的像素列為候選。
    matte = Image.fromarray(dark.astype(np.uint8) * 255)
    small = matte.resize((min(width, 64), min(height, 64)), Image.Resampling.BOX)
    small = small.point(lambda value: 255 if value >= 236 else 0)
    padded = ImageOps.expand(small, border=1, fill=255)
    ImageDraw.floodfill(padded, (0, 0), 128)
    connected = padded.crop((1, 1, padded.width - 1, padded.height - 1))
    mask = (np.asarray(connected.resize((width, height), Image.Resampling.NEAREST)) == 128) & dark
    if not MIN_COVERAGE <= mask.mean() <= MAX_COVERAGE:
        return np.zeros_like(dark)
    return mask


def texture(size, rng, held_out):
    width, height = size
    if held_out:
        # 驗證背景採用規則接縫和斑駁梯度，訓練背景沒有規則接縫。
        y, x = np.ogrid[:height, :width]
        step = int(rng.integers(17, 36))
        grout = ((x + (y // step % 2) * (step // 2)) % (step * 2) < 2) | (y % step < 2)
        grain = 25 * np.sin(x / 14 + y / 10) + 12 * np.cos(x / 4 - y / 19) - 35 * grout
    else:
        grain = np.zeros((height, width), dtype=np.float32)
        for cells, amplitude in ((4, 68), (14, 44), (45, 25)):
            small = Image.fromarray(rng.integers(0, 256, (cells, cells), dtype=np.uint8))
            noise = np.asarray(small.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
            grain += (noise / 255 - 0.5) * amplitude
    grain = grain + rng.normal(0, 5, (height, width))
    color = rng.uniform(45, 210) + rng.uniform(-35, 35, size=3)
    return np.clip(grain[:, :, None] + color, 0, 255).astype(np.uint8)


def replace_background(image, rng, *, held_out=False):
    rgb = np.array(image)
    mask = exterior_mask(rgb)
    if not mask.any():
        return image
    background = texture(image.size, rng, held_out)
    rgb[mask] = background[mask]
    return Image.fromarray(rgb)
