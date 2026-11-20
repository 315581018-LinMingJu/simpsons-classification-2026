"""固定門檻格與一次 Confirm 的檢索停損。"""

import math

import numpy as np

from v4.analysis import paired_effect

PROFILES = {
    "balanced": {"inliers": 5, "fraction": 0.48, "spread": 0.16, "cosine": 0.58},
    "conservative": {"inliers": 7, "fraction": 0.55, "spread": 0.20, "cosine": 0.64},
    "strict": {"inliers": 9, "fraction": 0.60, "spread": 0.23, "cosine": 0.70},
}
CROSS_CLASS_MARGIN = 2
MAX_NEGATIVE_RATE = 0.02
MAX_NEGATIVE_WILSON95_UPPER = 0.06
MIN_SOURCE_RECALL = 0.15
MIN_VAL_COVERAGE = 0.05
MIN_GAIN_PP = 1.0
MIN_WINS = 3


def accepted(matches, profile):
    qualified = [match for match in matches if
                 match.inliers >= profile["inliers"] and match.inlier_fraction >= profile["fraction"]
                 and match.spread >= profile["spread"] and match.cosine >= profile["cosine"]]
    if not qualified:
        return None
    best = qualified[0]
    if any(match.label != best.label and best.inliers - match.inliers < CROSS_CLASS_MARGIN
           for match in qualified[1:]):
        return None
    return best


def wilson_upper(hits, size):
    z = 1.959963984540054
    p = hits / size
    return (p + z**2 / (2 * size) + z * math.sqrt(p * (1 - p) / size + z**2 / (4 * size**2))) / (
        1 + z**2 / size)


def metrics(evidence, name, seed, confirm=False):
    profile = PROFILES[name]
    positive = [row for row in evidence if row["role"] == "positive"]
    val = [row for row in evidence if row["role"] == "validation"]
    pos_picks = [accepted(row["matches"], profile) for row in positive]
    negative_count = sum(accepted(row["negative_matches"], profile) is not None for row in positive)
    recovered = sum(pick is not None and pick.item == row["item"]
                    for row, pick in zip(positive, pos_picks, strict=True))
    val_picks = [accepted(row["matches"], profile) for row in val]
    old = np.asarray([row["baseline"] for row in val])
    truth = np.asarray([row["label"] for row in val])
    new = np.asarray([row["baseline"] if pick is None else pick.label
                      for row, pick in zip(val, val_picks, strict=True)])
    effects = (new == truth).astype(np.int8) - (old == truth).astype(np.int8)
    wins, losses = int((effects > 0).sum()), int((effects < 0).sum())
    hard_gain = paired_effect(effects, truth, seed)
    recall = recovered / len(positive)
    false_rate, false_upper = negative_count / len(positive), wilson_upper(negative_count, len(positive))
    coverage = sum(pick is not None for pick in val_picks) / len(val)
    failures = []
    if recall < MIN_SOURCE_RECALL:
        failures.append("low_source_recall")
    if false_rate > MAX_NEGATIVE_RATE or false_upper > MAX_NEGATIVE_WILSON95_UPPER:
        failures.append("false_acceptance_risk")
    if coverage < MIN_VAL_COVERAGE:
        failures.append("low_validation_coverage")
    if hard_gain["pp"] + 1e-9 < MIN_GAIN_PP or wins < MIN_WINS or wins <= losses:
        failures.append("no_reliable_validation_gain")
    if confirm and hard_gain["ci95_pp"][0] <= 0:
        failures.append("paired_ci_includes_zero")
    report = {
        "threshold": name, "samples": {"known_source": len(positive), "absent_label": len(positive),
                                       "isolated_validation": len(val)},
        "exact_source_recall": recall, "exact_sources_recovered": recovered,
        "known_source_coverage": sum(pick is not None for pick in pos_picks) / len(positive),
        "absent_label_false_acceptance": false_rate, "absent_label_false_accepts": negative_count,
        "absent_label_wilson95_upper": false_upper, "isolated_validation_coverage": coverage,
        "isolated_validation_accept_wrong_class": sum(pick is not None and pick.label != row["label"]
                                                     for row, pick in zip(val, val_picks, strict=True)),
        "v8_accuracy": float((old == truth).mean()), "retrieval_accuracy": float((new == truth).mean()),
        "gain": hard_gain, "wins": wins, "losses": losses,
        "decision": "pass" if not failures else "stop", "failures": failures,
    }
    return report


def choose_screen(evidence, seed):
    rows = [metrics(evidence, name, seed) for name in PROFILES]
    feasible = [row for row in rows if row["decision"] == "pass"]
    winner = max(feasible, key=lambda row: (row["gain"]["pp"], -row["absent_label_false_accepts"],
                                            PROFILES[row["threshold"]]["inliers"])) if feasible else None
    return {"thresholds": rows, "locked_threshold": winner["threshold"] if winner else None,
            "decision": "advance_to_confirm" if winner else "stop_screen"}
