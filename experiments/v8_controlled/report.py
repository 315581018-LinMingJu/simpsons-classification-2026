"""配對準確率、改判率、固定多重檢定與停損。"""

from math import comb

import numpy as np

from v4.analysis import paired_effect

CONTRASTS = {
    "black_rim": ("neutral", "black_rim"),
    "brightness": ("neutral", "bright_background"),
    "outer_detail": ("neutral", "outer_detail"),
    "content_retention": ("offcenter", "retained"),
    "retention_vs_count_control": ("matched_count", "retained"),
}
MIN_ACCURACY_EFFECT_PP = 2.0
MIN_FLIP_PP = 5.0
MIN_FLIP_LOWER_PP = 2.0
MIN_NEUTRAL_ACCURACY = .75
MIN_RETENTION_GAIN_PP = 5.0


def exact_mcnemar(wins, losses):
    changes = wins + losses
    if not changes:
        return 1.0
    return min(1.0, 2 * sum(comb(changes, k) for k in range(min(wins, losses) + 1)) / 2**changes)


def holm(pvalues):
    names = sorted(pvalues, key=pvalues.get)
    result, prior = {}, 0.0
    for i, name in enumerate(names):
        prior = max(prior, min(1.0, pvalues[name] * (len(names) - i)))
        result[name] = prior
    return result


def effects(labels, predictions, seed, geometry):
    rows = {}
    for name, (baseline, changed) in CONTRASTS.items():
        first, second = predictions[baseline], predictions[changed]
        before, after = first == labels, second == labels
        deltas = after.astype(np.int8) - before.astype(np.int8)
        wins, losses = int((deltas > 0).sum()), int((deltas < 0).sum())
        rows[name] = {
            "reference": baseline, "treatment": changed, "images": len(labels),
            "reference_accuracy_pct": 100 * float(before.mean()),
            "treatment_accuracy_pct": 100 * float(after.mean()),
            "accuracy_effect": paired_effect(deltas, labels, seed),
            "flip_rate": paired_effect(first != second, labels, seed),
            "wins": wins, "losses": losses, "mcnemar_p": exact_mcnemar(wins, losses),
        }

    adjusted = holm({key: value["mcnemar_p"] for key, value in rows.items()})
    neutral_ok = float((predictions["neutral"] == labels).mean()) >= MIN_NEUTRAL_ACCURACY
    for name, row in rows.items():
        row["mcnemar_holm_p"] = adjusted[name]
        effect, flip = row["accuracy_effect"], row["flip_rate"]
        lo, hi = effect["ci95_pp"]
        positive = effect["pp"] >= MIN_ACCURACY_EFFECT_PP and lo > 0
        negative = effect["pp"] <= -MIN_ACCURACY_EFFECT_PP and hi < 0
        if not neutral_ok:
            status = "stop_weak_control"
        elif ((name == "content_retention" and geometry["mean_patch_gain_pp"] < MIN_RETENTION_GAIN_PP)
              or (name == "retention_vs_count_control"
                  and geometry["mean_patch_gain_vs_matched_pp"] < MIN_RETENTION_GAIN_PP)):
            status = "stop_insufficient_retention_contrast"
        elif (positive or negative) and adjusted[name] <= .05:
            status = "accuracy_improved" if positive else "accuracy_decreased"
        elif flip["pp"] >= MIN_FLIP_PP and flip["ci95_pp"][0] > MIN_FLIP_LOWER_PP:
            status = "prediction_sensitive_accuracy_unresolved"
        else:
            status = "stop_no_clear_effect"
        row["decision"] = status
    if (rows["content_retention"]["decision"] == "accuracy_improved"
            and rows["retention_vs_count_control"]["decision"] != "accuracy_improved"):
        rows["content_retention"]["decision"] = "improved_vs_v8_but_selection_effect_unresolved"
    return rows


def geometry_summary(retention, selected, matched, cues):
    before = retention.mean(axis=1)
    after = (retention * selected).sum(axis=1) / selected.sum(axis=1)
    control = (retention * matched).sum(axis=1) / matched.sum(axis=1)
    result = {
        "mean_patch_retained_v8_pct": 100 * float(before.mean()),
        "mean_patch_retained_selected_pct": 100 * float(after.mean()),
        "mean_patch_gain_pp": 100 * float((after - before).mean()),
        "mean_patch_retained_matched_count_pct": 100 * float(control.mean()),
        "mean_patch_gain_vs_matched_pp": 100 * float((after - control).mean()),
        "images_with_pruned_views": int((~selected.all(axis=1)).sum()),
        "mean_selected_views": float(selected.sum(axis=1).mean()),
    }
    present = [i for i, row in enumerate(cues) if row is not None]
    if present:
        values = np.stack([cues[i] for i in present])
        choice = selected[present]
        result.update(cue_proxy_images=len(present),
                      mean_cue_proxy_retained_v8_pct=100 * float(values.mean()),
                      mean_cue_proxy_retained_selected_pct=100 * float(
                          ((values * choice).sum(axis=1) / choice.sum(axis=1)).mean()))
    return result
