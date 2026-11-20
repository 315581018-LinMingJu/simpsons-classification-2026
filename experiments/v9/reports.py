"""有標籤配對判定與不含真實標籤的 Test 分組。"""

from collections import defaultdict

import numpy as np
from PIL import Image

from v4.analysis import paired_effect
from v4.geometry import seeded
from v5.challenge import panel, primary_geometry, warp
from v7.impulse import MIN_TRIGGER_FRACTION
from v8.data import CONDITIONS, HARD_WEIGHTS, open_rgb
from v9.views import RECIPES, REFERENCE

SCREEN_MIN_PP = 0.75
CONFIRM_MIN_PP = 1.0
MAX_CLEAN_DROP_PP = 0.5
MAX_MAJOR_DROP_PP = 5.0


def passes(result, confirm=False):
    hard = result["hard_effect"]
    return (hard["pp"] >= (CONFIRM_MIN_PP if confirm else SCREEN_MIN_PP)
            and (not confirm or hard["ci95_pp"][0] > 0)
            and result["gain_pp"]["clean"] >= -MAX_CLEAN_DROP_PP
            and min(result["gain_pp"][condition] for condition in HARD_WEIGHTS if condition != "negative")
            >= -MAX_MAJOR_DROP_PP)


def validation_report(by_condition, labels, names, seed, confirm=False):
    report = {}
    baseline = {key: by_condition[key][REFERENCE].labels for key in CONDITIONS}
    base_correct = {key: (pred == labels).astype(np.int8) for key, pred in baseline.items()}
    base_accuracy = {key: float(value.mean()) for key, value in base_correct.items()}
    for name in names:
        scores = {key: by_condition[key][name] for key in CONDITIONS}
        correct = {key: (score.labels == labels).astype(np.int8) for key, score in scores.items()}
        effects = {key: value - base_correct[key] for key, value in correct.items()}
        result = {
            "accuracy": {key: float(value.mean()) for key, value in correct.items()},
            "gain_pp": {key: 100 * float(value.mean()) for key, value in effects.items()},
            "hard_accuracy": sum(weight * float(correct[key].mean()) for key, weight in HARD_WEIGHTS.items()),
            "hard_effect": paired_effect(sum(weight * effects[key] for key, weight in HARD_WEIGHTS.items()),
                                         labels, seed),
            "change_rate": {key: float((scores[key].labels != baseline[key]).mean()) for key in CONDITIONS},
            "wins": {key: int((effects[key] > 0).sum()) for key in CONDITIONS},
            "losses": {key: int((effects[key] < 0).sum()) for key in CONDITIONS},
            "views_per_image": len(RECIPES[name].views),
        }
        result["passes"] = name != REFERENCE and passes(result, confirm)
        report[name] = result
    eligible = [name for name, result in report.items() if result["passes"]]
    chosen = max(eligible, key=lambda name: (report[name]["hard_effect"]["pp"],
                                             -report[name]["views_per_image"])) if eligible else None
    return {"samples_per_condition": len(labels), "reference_accuracy": base_accuracy, "methods": report,
            "decision": "passes_confirmation" if chosen and confirm else "frozen_for_confirm" if chosen
            else "no_candidate_passed", "selected": chosen}


def frame_fraction(width, height, scale):
    """224 裁切佔縮放後畫面面積的比例。"""
    return 224**2 / (scale * (scale * max(width, height) // min(width, height)))


def primary_retention(examples, seed):
    """用合成場景已知的主圖塊 alpha 估計裁切保留量。"""
    by_scale = {scale: [] for scale in (240, 256, 288)}
    for example in examples:
        source = panel(open_rgb(example.path))
        rng = np.random.default_rng(seeded(seed, example.item).getrandbits(64))
        project = primary_geometry(source.size, rng)[0]
        alpha = warp(source, project).getchannel("A")
        for scale in by_scale:
            mask = np.asarray(alpha.resize((scale, scale), Image.Resampling.BILINEAR), dtype=float)
            offset = scale - 224
            center = round(offset / 2)
            top_lefts = ((0, 0), (offset, 0), (0, offset), (offset, offset), (center, center))
            by_scale[scale].append([
                float(mask[y:y + 224, x:x + 224].sum() / mask.sum()) for x, y in top_lefts
            ])
    return {"synthetic_labeled_images": len(examples),
            "median_center_fraction": {str(scale): float(np.median(np.asarray(rows)[:, -1]))
                                       for scale, rows in by_scale.items()},
            "median_least_retained_crop_fraction": {
                str(scale): float(np.median(np.asarray(rows).min(axis=1))) for scale, rows in by_scale.items()
            }}


def test_cohort(items, fractions, features):
    for i, item in enumerate(items):
        row = features[item]
        yield {"visual_band": row["visual_band"], "combo_support": row["combo_support"],
               "outer_black": "minority" if float(row["black_edge_fraction"]) < 0.5 else "majority",
               "impulse_trigger": "on" if fractions[i] >= MIN_TRIGGER_FRACTION else "off"}


def test_diagnostics(items, methods, fractions, features, archived):
    groups = defaultdict(list)
    saved = np.array([archived[item] for item in items])
    dims = list(test_cohort(items, fractions, features))
    for i, row in enumerate(dims):
        groups["all", "all"].append(i)
        for key, value in row.items():
            groups[key, value].append(i)
    reference = methods[REFERENCE]
    records = []
    for name, scores in methods.items():
        for (dimension, group), indices in sorted(groups.items()):
            ix = np.array(indices)
            records.append((name, dimension, group, len(ix), float(scores.confidence[ix].mean()),
                            float(scores.margin[ix].mean()), float(scores.disagreement[ix].mean()),
                            float((scores.labels[ix] != reference.labels[ix]).mean()),
                            float((scores.labels[ix] != saved[ix]).mean())))
    mismatched = np.flatnonzero(reference.labels != saved)
    return dims, records, mismatched
