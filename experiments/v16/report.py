"""固定比例與有標籤 Screen 的配對選擇。"""

import numpy as np

from v4.analysis import paired_effect
from v8.data import CONDITIONS, HARD_WEIGHTS
from v16.probabilities import fuse

WEIGHTS = (.90, .80, .70, .50)
DEFAULT = .80
MIN_HARD_GAIN_PP = .75
MAX_CLEAN_DROP_PP = .50
MAX_HARD_CONDITION_DROP_PP = 4.0


def key(weight):
    return f"a{round(100 * weight):02d}"


def compare(before, after, labels, seed):
    items, deltas = {}, {}
    for condition in CONDITIONS:
        left, right = before[condition], after[condition]
        delta = (right == labels).astype(np.int8) - (left == labels).astype(np.int8)
        deltas[condition] = delta
        items[condition] = {
            "accuracy_pct": 100 * float((right == labels).mean()),
            "accuracy_effect": paired_effect(delta, labels, seed),
            "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
            "disagreement": paired_effect(right != left, labels, seed),
        }
    hard_delta = sum(weight * deltas[c] for c, weight in HARD_WEIGHTS.items())
    hard_disagreement = sum(weight * (after[c] != before[c]) for c, weight in HARD_WEIGHTS.items())
    return {"conditions": items, "hard_accuracy_pct": sum(w * items[c]["accuracy_pct"] for c, w in HARD_WEIGHTS.items()),
            "hard_effect": paired_effect(hard_delta, labels, seed),
            "hard_disagreement": paired_effect(hard_disagreement, labels, seed),
            "hard_image_condition_wins": sum(items[c]["wins"] for c in HARD_WEIGHTS),
            "hard_image_condition_losses": sum(items[c]["losses"] for c in HARD_WEIGHTS)}


def screen(a, b, labels, seed):
    predictions = {"a": {c: a[c].argmax(axis=1) for c in CONDITIONS},
                   "b": {c: b[c].argmax(axis=1) for c in CONDITIONS}}
    predictions.update({key(w): {c: fuse(a[c], b[c], w).argmax(axis=1) for c in CONDITIONS} for w in WEIGHTS})
    default = predictions[key(DEFAULT)]
    trials, eligible = {}, []
    for name, values in predictions.items():
        trials[name] = compare(predictions["a"], values, labels, seed)
        if name in ("a", "b", key(DEFAULT)):
            continue
        relative = compare(default, values, labels, seed)
        trials[name]["versus_default"] = relative
        lift = relative["hard_effect"]
        by_condition = relative["conditions"]
        if (lift["pp"] + 1e-9 >= MIN_HARD_GAIN_PP and lift["ci95_pp"][0] > 0
                and by_condition["clean"]["accuracy_effect"]["pp"] >= -MAX_CLEAN_DROP_PP - 1e-9
                and min(by_condition[c]["accuracy_effect"]["pp"] for c in HARD_WEIGHTS)
                >= -MAX_HARD_CONDITION_DROP_PP - 1e-9):
            eligible.append((lift["pp"], name))
    selected = max(eligible)[1] if eligible else key(DEFAULT)
    weight = next(w for w in WEIGHTS if key(w) == selected)
    return {"reused_screen": True, "independent_validation": False,
            "images_per_condition": len(labels), "selected": selected, "weight_a": weight, "weight_b": round(1 - weight, 2),
            "selection": "predeclared_advantage_vs_default" if eligible else "default_a80_b20",
            "selection_thresholds": {"min_hard_gain_pp": MIN_HARD_GAIN_PP, "hard_ci95_lower_gt_pp": 0,
                                     "max_clean_drop_pp": MAX_CLEAN_DROP_PP,
                                     "max_hard_condition_drop_pp": MAX_HARD_CONDITION_DROP_PP},
            "hard_condition_weights": HARD_WEIGHTS,
            "trials_vs_a": trials}
