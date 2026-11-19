"""一般影像量測、風險分組與有標籤錯誤率。"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import hypot, sqrt
from pathlib import Path

import numpy as np
from PIL import Image

from distribution_audit.features import FEATURES, IMAGE_ERRORS, visual_features
from v4.geometry import seeded
from v6.composite import composite
from v8.data import open_rgb, stress
from v9.reports import frame_fraction
from v12.protocol import SHIFT

SPATIAL = ("outer_activity", "activity_offcenter", "activity_compactness", "center_crop_activity_loss", "activity_peaks")
VISUAL = ("width", "height", "aspect_ratio", *FEATURES, "center_crop_frame_fraction_288", *SPATIAL)
SCORES = ("confidence", "entropy_nats", "margin", "view_disagreement")
GROUPS = {
    "bright": ("brightness", ">=", .45),
    "little_black_edge": ("black_edge_fraction", "<", .50),
    "high_detail_proxy": ("fine_detail", ">=", .025),
    "high_outer_activity_proxy": ("outer_activity", ">=", .045),
    "low_center_frame_retention": ("center_crop_frame_fraction_288", "<", .55),
    "crop_activity_loss_proxy": ("center_crop_activity_loss", ">=", .35),
    "offcenter_activity_proxy": ("activity_offcenter", ">=", .16),
    "compact_activity_proxy": ("activity_compactness", "<=", .25),
    "separated_activity_peaks_proxy": ("activity_peaks", ">=", 2),
    "low_confidence": ("confidence", "<", .50),
    "small_margin": ("margin", "<", .10),
    "high_entropy": ("entropy_nats", ">=", 1.50),
    "view_disagreement": ("view_disagreement", ">=", .30),
}
COMBOS = {"bright_textured_little_black": ("bright", "high_detail_proxy", "little_black_edge")}
GROUP_NAMES = (*GROUPS, *COMBOS)


@dataclass(frozen=True)
class Scan:
    item: str
    path: Path
    condition: str = "clean"
    seed: int = 0
    distractor: Path | None = None
    targets: object = None


def activity_proxies(image):
    # 同一張縮圖上量測亮度邊緣；它可同時受角色、背景、外框及雜訊影響。
    size = 96
    gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
                      dtype=np.float64) / 255
    activity = abs(np.diff(gray, axis=0, append=gray[-1:])) + abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    total = activity.sum()
    axis = (np.arange(size) + .5) / size - .5
    outer = (abs(axis) > .30)
    outer_activity = float(activity[outer[:, None] | outer[None, :]].mean())
    if total <= 1e-8:
        return {"outer_activity": outer_activity, **{key: None for key in SPATIAL[1:]}}
    offcenter = hypot(float((activity.sum(axis=0) * axis).sum() / total),
                      float((activity.sum(axis=1) * axis).sum() / total))
    width, height = image.size
    resized_width = 288 * width // min(width, height)
    resized_height = 288 * height // min(width, height)
    x = abs(axis) <= 112 / resized_width
    y = abs(axis) <= 112 / resized_height
    outside = 1 - float(activity[np.ix_(y, x)].sum() / total)
    cells = activity.reshape(6, 16, 6, 16).sum((1, 3))
    compact = float(total**2 / (cells.size * (cells**2).sum()))
    peaks = []
    order = np.argsort(-cells.ravel(), kind="stable")
    for i in order:
        p = divmod(int(i), cells.shape[1])
        if cells[p] < .65 * cells.max():
            break
        if all(hypot(p[0] - q[0], p[1] - q[1]) >= 2.5 for q in peaks):
            peaks.append(p)
    return {"outer_activity": outer_activity, "activity_offcenter": offcenter, "activity_compactness": compact,
            "center_crop_activity_loss": max(0.0, outside), "activity_peaks": len(peaks)}


def visual_record(image):
    width, height = image.size
    return {"width": width, "height": height, "aspect_ratio": width / height,
            **dict(zip(FEATURES, visual_features(image), strict=True)),
            "center_crop_frame_fraction_288": frame_fraction(width, height, 288),
            **activity_proxies(image)}


def scan_one(task):
    try:
        image = open_rgb(task.path)
        if task.condition == SHIFT:
            rng = np.random.default_rng(seeded(task.seed, f"v12:heldout:{task.item}").getrandbits(64))
            image, _ = composite(image, task.targets, rng, held_out=True)
        elif task.condition != "clean":
            other = open_rgb(task.distractor) if task.condition == "distractor" else None
            image = stress(image, task.condition, task.seed, task.item, other)
        return task.item, visual_record(image), None
    except IMAGE_ERRORS as exc:
        return task.item, {}, f"{type(exc).__name__}: {exc}"


def scan(tasks, workers):
    if workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(scan_one, tasks, chunksize=16))
    else:
        results = list(map(scan_one, tasks))
    return ({item: record for item, record, error in results if error is None},
            {item: error for item, _, error in results if error is not None})


def novelty_record(row):
    width, height = int(row["width"]), int(row["height"])
    measurements = {key: float(row[key]) for key in FEATURES}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in measurements.values()):
        raise ValueError(f"Non-finite or invalid distribution audit feature for {row['id']}")
    return {"width": width, "height": height, "aspect_ratio": width / height, **measurements,
            "center_crop_frame_fraction_288": frame_fraction(width, height, 288)}


def percentile_rows(rows, fields):
    result = {}
    for field in fields:
        values = [row[field] for row in rows if row.get(field) is not None]
        if values:
            result[field] = {"n": len(values), "p10_p50_p90": np.percentile(values, [10, 50, 90]).tolist()}
    return result


def flag(row, name):
    if name in COMBOS:
        values = [flag(row, sub) for sub in COMBOS[name]]
        return None if None in values else all(values)
    metric, op, threshold = GROUPS[name]
    value = row.get(metric)
    if value is None:
        return None
    if op == ">=":
        return value >= threshold
    return value <= threshold if op == "<=" else value < threshold


def wilson(errors, n):
    if not n:
        return None
    p, z = errors / n, 1.959963984540054
    mid = (p + z*z / (2*n)) / (1 + z*z / n)
    half = z * sqrt(p * (1-p) / n + z*z / (4*n*n)) / (1 + z*z / n)
    return [max(0.0, mid - half), min(1.0, mid + half)]


def group_rows(rows, truth=None):
    result = []
    for name in GROUP_NAMES:
        selected, rest = [], []
        for item, row in rows.items():
            status = flag(row, name)
            if status is None or (truth is not None and (item not in truth or "prediction" not in row)):
                continue
            (selected if status else rest).append(item)
        n, other = len(selected), len(rest)
        row = {"group": name, "eligible": n + other, "flagged": n,
               "flagged_rate": n / (n + other) if n + other else None}
        if truth is not None:
            errors = sum(rows[item]["prediction"] != truth[item] for item in selected)
            other_errors = sum(rows[item]["prediction"] != truth[item] for item in rest)
            rate, other_rate = (errors / n if n else None, other_errors / other if other else None)
            band, other_band = wilson(errors, n), wilson(other_errors, other)
            row.update(errors=errors, error_rate=rate, ci95=band, rest_count=other, rest_errors=other_errors,
                       rest_error_rate=other_rate, rest_ci95=other_band,
                       error_gap_pp=100 * (rate - other_rate) if n and other else None,
                       exploratory_signal=(min(n, other) >= 20 and errors + other_errors >= 5
                                           and band[0] > other_band[1]))
        result.append(row)
    return result
