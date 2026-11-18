"""舊 V5 結果逐圖稽核與困難驗證的配對統計。"""

import csv
from collections import Counter

import numpy as np

from v4.analysis import paired_effect
from v5.challenge import PARENTS


def legacy_audit(run):
    conditions = {"clean": {}, "heldout_background": {}}
    with (run / "paired_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            condition, item = row["condition"], row["relative_path"]
            if condition not in conditions or item in conditions[condition]:
                raise ValueError("Unexpected condition or duplicate image in V5 paired predictions")
            conditions[condition][item] = (
                row["true_character"], row["v2_prediction"], row["v5_prediction"]
            )
    clean, stress = conditions.values()
    if not clean or clean.keys() != stress.keys() or any(clean[i][0] != stress[i][0] for i in clean):
        raise ValueError("Legacy Clean and Heldout must have the same paired labels")

    result, transitions = {}, {}
    for condition, rows in conditions.items():
        wins, losses, swaps = [], [], []
        for item, (true, a, b) in rows.items():
            detail = {"relative_path": item, "true_character": true, "v2": a, "v5": b}
            if a != true == b:
                wins.append(detail)
            elif b != true == a:
                losses.append(detail)
            elif a != b:
                swaps.append(detail)
        result[condition] = {
            "wins": wins, "losses": losses, "wrong_to_wrong_changes": swaps,
            "v2_errors": sum(a != true for true, a, _ in rows.values()),
            "v5_errors": sum(b != true for true, _, b in rows.values()),
        }

    for model, index in (("v2", 1), ("v5", 2)):
        changed = [item for item in clean if clean[item][index] != stress[item][index]]
        clean_wrong = {item for item in clean if clean[item][index] != clean[item][0]}
        stress_wrong = {item for item in stress if stress[item][index] != stress[item][0]}
        transitions[model] = {
            "prediction_changes": len(changed), "same_error_ids": clean_wrong == stress_wrong,
            "clean_only_error_ids": sorted(clean_wrong - stress_wrong),
            "heldout_only_error_ids": sorted(stress_wrong - clean_wrong),
        }

    matte = run / "validation_samples.csv"
    matte_stats = None
    if matte.is_file():
        with matte.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if {row["relative_path"]: row["label"] for row in rows} != {key: value[0] for key, value in clean.items()}:
            raise ValueError("Validation matte manifest differs from paired predictions")
        fractions = [float(row["matte_fraction"]) for row in rows]
        if len(rows) != len(clean) or any(not 0 <= fraction <= 1 for fraction in fractions):
            raise ValueError("Invalid matte coverage values")
        matte_stats = {"unchanged": sum(fraction == 0 for fraction in fractions),
                       "median_replaced_fraction": float(np.median(fractions))}
        by_item = dict(zip((row["relative_path"] for row in rows), fractions, strict=True))
        for summary in result.values():
            for kind in ("wins", "losses", "wrong_to_wrong_changes"):
                for case in summary[kind]:
                    case["matte_fraction"] = by_item[case["relative_path"]]
    return {"paired_images": len(clean), "matte": matte_stats,
            "conditions": result, "background_changes_by_model": transitions}


def evaluate_challenge(targets, predictions, seed, alpha_fractions):
    labels = np.array(list(targets.values()))
    hits = {
        (condition, name): np.array([scores[item].label == true for item, true in targets.items()], dtype=int)
        for (condition, name), scores in predictions.items()
    }
    area = np.array([alpha_fractions[item] for item in targets])
    sizes = {"alpha_at_most_12_pct": area <= 0.12, "alpha_above_12_pct": area > 0.12}
    results = {}
    for condition, parent in PARENTS.items():
        a, b = (hits[condition, name] for name in ("v2", "v5"))
        delta = b - a
        errors = int((a == 0).sum())
        group_a = Counter(int(label) for label, ok in zip(labels, a, strict=True) if not ok)
        row = {
            "parent": parent, "images": len(labels), "v2_errors": errors,
            "v2_accuracy_pct": 100 * float(a.mean()), "v5_accuracy_pct": 100 * float(b.mean()),
            "v5_gain_macro_pp": paired_effect(delta, labels, seed),
            "v5_wins": int((delta == 1).sum()), "v5_losses": int((delta == -1).sum()),
            "both_wrong": int(((a == 0) & (b == 0)).sum()),
            "v2_error_counts_by_class_index": dict(sorted(group_a.items())),
            "near_ceiling": errors < 0.05 * len(labels),
            "foreground_size_slices": {
                key: {
                    "images": int(mask.sum()), "v2_accuracy_pct": 100 * float(a[mask].mean()),
                    "v5_accuracy_pct": 100 * float(b[mask].mean()),
                } for key, mask in sizes.items() if condition != "clean" and mask.any()
            },
        }
        if parent is not None:
            a_parent, b_parent = (hits[parent, name] for name in ("v2", "v5"))
            row.update({
                "v2_drop_vs_parent_macro_pp": paired_effect(a_parent - a, labels, seed),
                "v5_drop_vs_parent_macro_pp": paired_effect(b_parent - b, labels, seed),
                "v5_gain_change_vs_parent_macro_pp": paired_effect(
                    delta - (b_parent - a_parent), labels, seed
                ),
            })
        results[condition] = row
    return results
