"""後 TTA 機率校準與預先固定的配對確認規則。"""

import numpy as np

from v4.analysis import paired_effect
from v5.challenge import PARENTS

FAMILIES = {
    "clean": ("clean",),
    "v5_challenge": tuple(condition for condition in PARENTS if condition != "clean"),
    "v6_composite": ("heldout_composite",),
}
CONDITIONS = tuple(condition for family in FAMILIES.values() for condition in family)
WEIGHTS = (0.25, 0.5, 0.75)
CLEAN_LOSS_LIMIT_PP = 0.5


def rescale(probs, temperature):
    if temperature == 1:
        return probs
    values = np.log(np.clip(probs, 1e-12, 1)) / temperature
    values -= values.max(axis=-1, keepdims=True)
    adjusted = np.exp(values)
    return adjusted / adjusted.sum(axis=-1, keepdims=True)


def fuse(v5, v6, v5_weight, temperatures):
    return v5_weight * rescale(v5, temperatures["v5"]) + (1 - v5_weight) * rescale(v6, temperatures["v6"])


def family_reduce(values):
    return np.stack([values[[CONDITIONS.index(case) for case in family]].mean(axis=0)
                     for family in FAMILIES.values()])


def class_weights(labels):
    weights = 1 / np.bincount(labels)[labels]
    return weights / weights.sum()


def nll_rows(probs, labels):
    chosen = probs[:, np.arange(len(labels)), labels]
    return -np.log(np.clip(chosen, 1e-12, 1))


def fit_temperature(probs, labels):
    weights = class_weights(labels)
    grid = sorted(set([1.0, *np.geomspace(0.5, 3, 41).tolist()]))

    def score(value):
        family_loss = family_reduce(nll_rows(rescale(probs, value), labels))
        return float((family_loss.mean(axis=0) * weights).sum()), abs(np.log(value))

    return float(min(grid, key=score))


def metrics(probs, labels):
    prediction = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)
    correct = prediction == labels
    weights = class_weights(labels)
    ece = np.zeros(len(CONDITIONS))
    bins = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        ece += np.abs(((confidence - correct) * weights * (bins == index)).sum(axis=1))
    one_hot = np.eye(probs.shape[-1])[labels]
    measures = {
        "macro_accuracy_pct": 100 * (correct * weights).sum(axis=1),
        "nll": (nll_rows(probs, labels) * weights).sum(axis=1),
        "brier": (((probs - one_hot) ** 2).sum(axis=-1) * weights).sum(axis=1),
        "ece10": ece,
        "mean_confidence": (confidence * weights).sum(axis=1),
    }

    def columns(keys, arrays):
        return {key: {name: float(arrays[name][i]) for name in measures}
                for i, key in enumerate(keys)}

    families = {name: family_reduce(values) for name, values in measures.items()}
    return {
        "conditions": columns(CONDITIONS, measures),
        "families": columns(FAMILIES, families),
        "balanced": {name: float(values.mean()) for name, values in families.items()},
    }


def gains(probs, v6, labels, seed):
    correct = probs.argmax(axis=-1) == labels
    base_correct = v6.argmax(axis=-1) == labels
    difference = correct.astype(int) - base_correct
    families = family_reduce(difference)
    return {
        "balanced": paired_effect(families.mean(axis=0), labels, seed),
        "families": {key: paired_effect(values, labels, seed)
                     for key, values in zip(FAMILIES, families, strict=True)},
        "conditions": {
            key: {"wins": int((values == 1).sum()), "losses": int((values == -1).sum()),
                  "both_wrong": int((~(correct[i] | base_correct[i])).sum())}
            for i, (key, values) in enumerate(zip(CONDITIONS, difference, strict=True))
        },
    }


def guarded(gain, *, confirm):
    balanced = gain["balanced"]
    return ((balanced["ci95_pp"][0] if confirm else balanced["pp"]) > 0
            and gain["families"]["v5_challenge"]["pp"] > 0
            and gain["families"]["clean"]["pp"] >= -CLEAN_LOSS_LIMIT_PP)


def evaluate(v5, v6, labels, folds, seed):
    calibration, selection, confirmation = (np.flatnonzero(np.asarray(folds) == fold)
                                           for fold in ("calibration", "selection", "confirmation"))
    fitted = {}
    selection_calibration = {}
    for name, probs in (("v5", v5), ("v6", v6)):
        temperature = fit_temperature(probs[:, calibration], labels[calibration])
        raw = metrics(probs[:, selection], labels[selection])
        scaled = metrics(rescale(probs[:, selection], temperature), labels[selection])
        applied = (scaled["balanced"]["nll"] < raw["balanced"]["nll"]
                   and scaled["families"]["v5_challenge"]["nll"] <= raw["families"]["v5_challenge"]["nll"])
        fitted[name] = temperature if applied else 1.0
        selection_calibration[name] = {"fitted_temperature": temperature, "applied_temperature": fitted[name],
                                       "raw": raw, "scaled": scaled}

    select_v5, select_v6, select_labels = v5[:, selection], v6[:, selection], labels[selection]
    reports = []
    for weight in WEIGHTS:
        fused = fuse(select_v5, select_v6, weight, fitted)
        gain = gains(fused, select_v6, select_labels, seed)
        reports.append({"v5_weight": weight, "v6_weight": 1 - weight,
                        "metrics": metrics(fused, select_labels), "gain_vs_v6": gain,
                        "eligible": guarded(gain, confirm=False)})

    final_v5, final_v6, final_labels = v5[:, confirmation], v6[:, confirmation], labels[confirmation]
    final_report = {
        "v5": metrics(final_v5, final_labels), "v6": metrics(final_v6, final_labels),
        "v5_vs_v6": gains(final_v5, final_v6, final_labels, seed),
        "calibrated_models": {
            name: metrics(rescale(probs, fitted[name]), final_labels)
            for name, probs in (("v5", final_v5), ("v6", final_v6))
        },
    }
    eligible = [row for row in reports if row["eligible"]]
    chosen = None
    decision = "no_candidate_passed_selection"
    if eligible:
        best = min(eligible, key=lambda row: (
            -row["metrics"]["balanced"]["macro_accuracy_pct"],
            row["metrics"]["balanced"]["nll"], abs(row["v5_weight"] - 0.5),
        ))
        chosen = best["v5_weight"]
        final = fuse(final_v5, final_v6, chosen, fitted)
        gain = gains(final, final_v6, final_labels, seed)
        final_report.update({"ensemble": metrics(final, final_labels), "ensemble_gain_vs_v6": gain})
        decision = "supports_ensemble" if guarded(gain, confirm=True) else "not_confirmed"
    return {
        "decision": decision, "selected_v5_weight": chosen, "applied_temperatures": fitted,
        "fold_counts": {"calibration": len(calibration), "selection": len(selection),
                        "confirmation": len(confirmation)},
        "selection_calibration": selection_calibration, "selection_weights": reports,
        "confirmation": final_report,
    }
