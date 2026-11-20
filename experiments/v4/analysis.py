"""無標籤影像摘要，以及按相同驗證圖配對的效果估計。"""

import csv
import io
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

from v3.storage import publish
from v4.geometry import CONDITIONS, METHODS

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def image_sizes(samples):
    sizes = {}
    for path, item in samples:
        with Image.open(path) as image:
            sizes[item] = image.size
    return sizes


def quantiles(numbers):
    return [float(value) for value in np.quantile(numbers, QUANTILES)]


def size_summary(sizes):
    dimensions = list(sizes.values())
    widths, heights = zip(*dimensions, strict=True)
    return {
        "samples": len(dimensions),
        "square_count": sum(width == height for width, height in dimensions),
        "width_quantiles": quantiles(widths), "height_quantiles": quantiles(heights),
        "aspect_width_over_height_quantiles": quantiles([w / h for w, h in dimensions]),
    }


def thumbnail_signals(path):
    with Image.open(path) as image:
        exif_orientation = image.getexif().get(274, 1)
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((192, 192))
    colors = np.array(thumbnail, dtype=np.int16)
    # 色彩弱的圖片不必然是灰階原檔，邊緣亮度也不代表場景複雜度。
    weak_color = float((np.ptp(colors, axis=2) <= 5).mean())
    gray = np.asarray(thumbnail.convert("L"))
    edge = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    return weak_color, float(edge.std()), exif_orientation


def aspect_bin(width, height):
    ratio = width / height
    if ratio < 0.8:
        return "portrait"
    if ratio <= 1.25:
        return "near_square"
    return "landscape"


def read_v3_changes(path, ids):
    """只引用 V3 已存的改判與信心欄位，維持測試標籤未知。"""
    expected = set(ids)
    by_method = {method: {} for method in ("full_resize", "five_crop", "multiscale")}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            method = row["method"]
            if method in by_method:
                item = row["id"]
                if item not in expected or item in by_method[method]:
                    raise ValueError("Unexpected or repeated ID in V3 test_diagnostics.csv")
                changed, v1_changed = row["changed_vs_v2"], row["v1_v2_changed"]
                confidence = float(row["confidence"])
                if changed not in ("0", "1") or v1_changed not in ("0", "1") or not 0 <= confidence <= 1:
                    raise ValueError("Invalid disagreement or confidence in V3 diagnostics")
                by_method[method][item] = int(changed), int(v1_changed), confidence
    if any(rows.keys() != expected for rows in by_method.values()):
        raise ValueError("V4 needs the full V3 test_diagnostics.csv, not its sample report")
    return by_method


def changes_by_aspect(sizes, changes):
    rows = []
    for group in ("portrait", "near_square", "landscape"):
        ids = [item for item, size in sizes.items() if aspect_bin(*size) == group]
        if ids:
            for method, predictions in changes.items():
                rows.append((group, method, len(ids),
                             100 * sum(predictions[item][0] for item in ids) / len(ids),
                             100 * sum(predictions[item][1] for item in ids) / len(ids),
                             sum(predictions[item][2] for item in ids) / len(ids)))
    return rows


def save_sheet(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = max(len(row) for row in rows)
    figure, axes = plt.subplots(len(rows), width, squeeze=False, layout="constrained",
                                figsize=(width * 2.4, len(rows) * 2.7))
    for row, cells in enumerate(rows):
        for axis in axes[row]:
            axis.axis("off")
        for column, (title, source) in enumerate(cells):
            if isinstance(source, Path):
                with Image.open(source) as image:
                    thumbnail = image.convert("RGB")
                    thumbnail.thumbnail((360, 280))
            else:
                thumbnail = source
            axis = axes[row, column]
            axis.imshow(thumbnail)
            axis.set_title(textwrap.fill(title, 30), fontsize=8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=125)
    plt.close(figure)
    publish(path, buffer.getvalue())


def paired_effect(values, labels, seed, draws=2000):
    # 各類內重抽配對圖片。估計量對 50 類等權重。
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    draw_sums = np.zeros(draws)
    means = []
    for label in np.unique(labels):
        group = values[labels == label]
        means.append(group.mean())
        draw_sums += group[rng.integers(len(group), size=(draws, len(group)))].mean(axis=1)
    lower, upper = np.percentile(100 * draw_sums / len(means), [2.5, 97.5])
    return {"pp": 100 * float(np.mean(means)), "ci95_pp": [float(lower), float(upper)]}


def compare(targets, predictions, seed):
    ids = list(targets)
    labels = np.array([targets[item] for item in ids])
    hits = {
        (condition, method): np.array(
            [predictions[condition, method][item].label == targets[item] for item in ids], dtype=int
        )
        for condition in CONDITIONS for method in METHODS
    }
    scores = []
    for condition in CONDITIONS:
        original = hits[condition, "baseline"]
        for method in METHODS:
            result = hits[condition, method]
            difference = result - original
            effect = paired_effect(difference, labels, seed)
            scores.append({
                "condition": condition, "method": method, "samples": len(ids),
                "macro_accuracy_pct": paired_effect(result, labels, seed)["pp"],
                "micro_accuracy_pct": 100 * float(result.mean()),
                "wins_vs_baseline": int((difference == 1).sum()),
                "losses_vs_baseline": int((difference == -1).sum()),
                "gain_pp": effect["pp"], "gain_ci95_low_pp": effect["ci95_pp"][0],
                "gain_ci95_high_pp": effect["ci95_pp"][1],
            })
    multiscale_gain = {case: hits[case, "multiscale"] - hits[case, "baseline"] for case in CONDITIONS}
    off = paired_effect(multiscale_gain["offcenter"], labels, seed)
    interaction = paired_effect(multiscale_gain["offcenter"] - multiscale_gain["center"], labels, seed)
    clean = paired_effect(multiscale_gain["clean"], labels, seed)
    if off["ci95_pp"][0] > 0 and interaction["ci95_pp"][0] > 0 and clean["pp"] >= -0.5:
        decision = "supports_offcenter_hypothesis"
    elif off["ci95_pp"][1] <= 0 or interaction["ci95_pp"][1] <= 0:
        decision = "does_not_support_offcenter_hypothesis"
    else:
        decision = "inconclusive"
    return scores, {
        "multiscale_offcenter_gain": off, "multiscale_clean_gain": clean,
        "offcenter_minus_center_gain": interaction, "decision": decision,
    }
