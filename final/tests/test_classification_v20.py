import random

import numpy as np
from PIL import Image

import classification_v20 as v20


def make_tree(root, n=10):
    for c in range(50):
        d = root / f"class_{c:02d}"
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8)).save(d / f"{i}.jpg")


def test_baseline_split_matches_reference_algorithm(tmp_path):
    make_tree(tmp_path)
    classes, train, val = v20.baseline_split(tmp_path)
    assert len(classes) == 50 and len(val) == 100 and len(train) == 400
    rng = random.Random(2026)
    files = sorted((tmp_path / classes[0]).glob("*.jpg"))
    rng.shuffle(files)
    assert [p for p, y in val if y == 0] == files[:2]


def test_composite_masks_black_and_white_pixels():
    arr = np.full((20, 30, 3), 128, np.uint8)
    arr[:5] = 0
    arr[-5:] = 255
    out = np.asarray(v20.composite(Image.fromarray(arr), [], random.Random(1)))
    h, w = out.shape[:2]
    assert 20 <= h - 20 <= 60 and 20 <= w - 30 <= 60
    py, px = (h - 20) // 2, (w - 30) // 2
    assert (out[py + 5:py + 15, px:px + 30] == 128).all()


def test_procedural_background_size():
    bg = v20.procedural_background((50, 40), random.Random(3))
    assert bg.size == (50, 40)
