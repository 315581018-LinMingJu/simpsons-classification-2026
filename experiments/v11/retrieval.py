"""凍結 ConvNeXt 的角色圖塊檢索與決定性的局部幾何驗證。"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from PIL import Image

from v5.challenge import panel

SIZE = 224
LOCAL_SIDE = 28
DESCRIPTORS = 96
SHORTLIST = 3
INLIER_RADIUS = 0.055
MIN_TOKEN_COSINE = 0.58
MAX_TOKEN_RATIO = 0.90


@dataclass
class Feature:
    vector: np.ndarray
    coordinates: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class Match:
    item: str
    label: int
    cosine: float
    inliers: int
    inlier_fraction: float
    spread: float
    error: float


def reference_view(image):
    cutout = panel(image)
    canvas = Image.new("RGB", cutout.size, (118, 118, 118))
    mask = cutout.getchannel("A")
    canvas.paste(cutout.convert("RGB"), mask=mask)
    return canvas, mask


def query_views(image):
    # 四個重疊局部區域只用於找近鄰。V8 分類仍沿用原本固定推論。
    width, height = image.size
    w, h = round(width * 0.68), round(height * 0.68)
    return [(image, None)] + [(image.crop((x, y, x + w, y + h)), None)
                             for y in (0, height - h) for x in (0, width - w)]


def point_cells(gray, mask):
    rgb_delta = np.asarray(gray, dtype=np.float32) / 255
    dy = np.abs(np.diff(rgb_delta, axis=0, append=rgb_delta[-1:, :]))
    dx = np.abs(np.diff(rgb_delta, axis=1, append=rgb_delta[:, -1:]))
    activity = (dx + dy).reshape(LOCAL_SIDE, 8, LOCAL_SIDE, 8).mean(axis=(1, 3))
    covered = mask.reshape(LOCAL_SIDE, 8, LOCAL_SIDE, 8).mean(axis=(1, 3))
    activity[covered < 0.80] = 0
    # 每 2×2 只保留一點，避免同一條邊累積大量重複票數。
    tiles = activity.reshape(14, 2, 14, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    indices = tiles.argmax(axis=1)
    values = tiles[np.arange(len(indices)), indices]
    chosen = np.argsort(-values, kind="stable")[:DESCRIPTORS]
    chosen = chosen[values[chosen] >= 0.008]
    return (chosen // 14 * 2 + indices[chosen] // 2, chosen % 14 * 2 + indices[chosen] % 2)


def extract_batch(model, pairs, normalize, device="cuda"):
    import torch
    import torch.nn.functional as F

    frames, masks, cells = [], [], []
    for image, mask in pairs:
        frame = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        alpha = (np.ones((SIZE, SIZE), np.float32) if mask is None else
                 np.asarray(mask.resize((SIZE, SIZE), Image.Resampling.BILINEAR), np.float32) / 255)
        frames.append(normalize(frame))
        masks.append(alpha.reshape(14, 16, 14, 16).mean(axis=(1, 3)))
        cells.append(point_cells(frame.convert("L"), alpha))
    with torch.inference_mode():
        x = torch.stack(frames).to(device)
        local = model.features[:4](x).float()
        coarse = model.features[4:6](local).float()
        if local.shape[-2:] != (28, 28) or coarse.shape[-2:] != (14, 14):
            raise ValueError("ConvNeXt feature grid differs from the V11 protocol")
        mask_tensor = torch.from_numpy(np.stack(masks)).to(device).unsqueeze(1)
        vectors = F.normalize((coarse * mask_tensor).sum((2, 3)), dim=1).cpu().numpy()
        local = F.normalize(local, dim=1).permute(0, 2, 3, 1).cpu().numpy()
    result = []
    for i, (y, x) in enumerate(cells):
        coordinates = ((np.column_stack((x, y)) + 0.5) / LOCAL_SIDE).astype(np.float32)
        result.append(Feature(vectors[i], coordinates, local[i, y, x].astype(np.float16)))
    return result


def descriptor_pairs(left, right):
    if min(len(left), len(right)) < 2:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    sim = left.astype(np.float32) @ right.astype(np.float32).T
    np.clip(sim, -1, 1, out=sim)
    target, source = sim.argmax(axis=1), sim.argmax(axis=0)
    all_left = np.arange(len(left))
    top = sim[all_left, target]
    second = np.partition(sim, -2, axis=1)[:, -2]
    unique = (source[target] == all_left) & (top >= MIN_TOKEN_COSINE)
    unique &= (1 - top) < MAX_TOKEN_RATIO * np.maximum(1 - second, 1e-6)
    rows = np.flatnonzero(unique)
    rows = rows[np.argsort(-top[rows], kind="stable")[:32]]
    return rows, target[rows]


def geometric_fit(left, right):
    if len(left) < 2:
        return 0, 0.0, 0.0, 1.0
    # 用兩個點估計旋轉、平移與縮放，枚舉固定順序的幾何假設。
    a, b = left[:, 0] + 1j * left[:, 1], right[:, 0] + 1j * right[:, 1]
    i, j = np.asarray(list(combinations(range(len(a)), 2))).T
    base = a[j] - a[i]
    usable = np.abs(base) >= 0.11
    i, j, base = i[usable], j[usable], base[usable]
    factor = (b[j] - b[i]) / base
    good = (np.abs(factor) >= 0.30) & (np.abs(factor) <= 3.0)
    if not good.any():
        return 0, 0.0, 0.0, 1.0
    factor, i = factor[good], i[good]
    intercept = b[i] - factor * a[i]
    errors = np.abs(factor[:, None] * a + intercept[:, None] - b)
    mask = errors <= INLIER_RADIUS
    totals = mask.sum(axis=1)
    mean_error = (errors * mask).sum(axis=1) / np.maximum(totals, 1)
    best = np.lexsort((mean_error, -totals))[0]
    matched = mask[best]
    count = int(totals[best])
    if not count:
        return 0, 0.0, 0.0, 1.0
    spread = min(np.ptp(left[matched], axis=0).min(), np.ptp(right[matched], axis=0).min())
    return count, count / len(left), float(spread), float(mean_error[best])


def match_rank(match):
    return match.inliers, match.inlier_fraction, match.spread, -match.error, match.cosine


class Index:
    def __init__(self, manifest, features):
        if len(manifest) != len(features) or not len(features):
            raise ValueError("Index features and manifest differ")
        self.manifest, self.features = manifest, features
        self.vectors = np.stack([entry.vector for entry in features])
        if not np.isfinite(self.vectors).all():
            raise ValueError("Non-finite index vectors")
        self.labels = np.array([row["label"] for row in manifest])

    def search(self, queries, absent_label=None):
        cosine = np.stack([query.vector for query in queries]) @ self.vectors.T
        if absent_label is not None:
            cosine[:, self.labels == absent_label] = -np.inf
        matches = {}
        for query, row in zip(queries, cosine, strict=True):
            order = np.argsort(-row, kind="stable")
            shortlist = list(order[:SHORTLIST])
            if len({self.labels[i] for i in shortlist if np.isfinite(row[i])}) == 1:
                rivals = order[(self.labels[order] != self.labels[order[0]]) & np.isfinite(row[order])]
                shortlist.extend(rivals[:1])
            for i in shortlist:
                if not np.isfinite(row[i]):
                    continue
                ref, entry = self.features[i], self.manifest[i]
                left, right = descriptor_pairs(ref.descriptors, query.descriptors)
                geometry = geometric_fit(ref.coordinates[left], query.coordinates[right])
                match = Match(entry["item"], entry["label"], float(row[i]), *geometry)
                old = matches.get(match.item)
                if old is None or match_rank(match) > match_rank(old):
                    matches[match.item] = match
        return sorted(matches.values(), key=match_rank, reverse=True)
