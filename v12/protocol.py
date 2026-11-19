"""固定原 V8 配對條件，額外量測亮背景複合場景。"""

import time

import numpy as np

from v3.inference import infer_loader
from v4.analysis import paired_effect
from v4.geometry import seeded
from v6.composite import composite
from v7.impulse import clean_impulses
from v8.data import CONDITIONS, HARD_WEIGHTS, EvaluationImages, open_rgb
from v8.experiment import data_loader, load_evaluation, save_evaluation

SHIFT = "heldout_composite"
ALL_CONDITIONS = (*CONDITIONS, SHIFT)
HEAD_MIN_CLEAN_PCT = 85.0
HEAD_MAX_CLEAN_DROP = 8.0
HEAD_MAX_SHIFT_DROP = 15.0
MAX_CLEAN_DROP = 0.5
MAX_OTHER_DROP = 4.0
HARD_MIN = {"screen": .75, "confirm": 1.0}
SHIFT_MIN = 1.0


class ShiftImages:
    def __init__(self, root, cases, views, targets, seed):
        self.root, self.cases, self.views, self.targets, self.seed = root, cases, views, targets, seed

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image = open_rgb(self.root / case.item)
        rng = np.random.default_rng(seeded(self.seed, f"v12:heldout:{case.item}").getrandbits(64))
        shifted, _ = composite(image, self.targets, rng, held_out=True)
        shifted, _ = clean_impulses(shifted)
        return self.views(shifted), case.item


def evaluate_condition(directory, model, info, cases, condition, views, args, settings):
    if (directory / "summary.json").is_file():
        scores, _ = load_evaluation(directory, cases, settings, info.inputs["classes"])
        return scores[condition]
    dataset = (ShiftImages(info.root, cases, views, info.targets, info.inputs["seed"]) if condition == SHIFT
               else EvaluationImages(info.root, cases, condition, views, info.inputs["seed"], True))
    loader = data_loader(dataset, args.eval_batch_size, args.workers)
    started = time.perf_counter()
    predictions = [pair for batch in infer_loader(model, loader) for pair in batch]
    if [item for item, _ in predictions] != [case.item for case in cases]:
        raise ValueError("Validation inference changed image order")
    scores = np.asarray([pred.label for _, pred in predictions])
    timings = {condition: {"seconds": time.perf_counter() - started}}
    save_evaluation(directory, cases, info.inputs["classes"], {condition: scores}, timings, settings)
    return scores


def comparison(phase, baseline, adapted, cohorts, seed):
    effects, reports, labels = {}, {}, {}
    for name in baseline:
        labels[name] = np.asarray([case.label for case in cohorts[name]])
        before, after = baseline[name] == labels[name], adapted[name] == labels[name]
        delta = after.astype(np.int8) - before.astype(np.int8)
        effects[name] = delta
        reports[name] = {"images": len(delta), "v8_accuracy_pct": 100 * float(before.mean()),
                         "v12_accuracy_pct": 100 * float(after.mean()), "effect": paired_effect(delta, labels[name], seed),
                         "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum())}
    clean = reports["clean"]
    shift = reports[SHIFT]["effect"]
    if phase == "head":
        failures = []
        if clean["v12_accuracy_pct"] < HEAD_MIN_CLEAN_PCT or clean["effect"]["pp"] < -HEAD_MAX_CLEAN_DROP - 1e-9:
            failures.append("weak_head_clean")
        if shift["pp"] < -HEAD_MAX_SHIFT_DROP - 1e-9:
            failures.append("weak_head_shift")
        return {"decision": "stop_head" if failures else "advance_to_tune", "reasons": failures,
                "conditions": reports, "heldout_effect": shift}
    hard = paired_effect(sum(effects[name] * weight for name, weight in HARD_WEIGHTS.items()),
                         labels["clean"], seed)
    failures = []
    if hard["pp"] + 1e-9 < HARD_MIN[phase]:
        failures.append("hard_gain_below_minimum")
    if shift["pp"] + 1e-9 < SHIFT_MIN:
        failures.append("heldout_gain_below_minimum")
    if clean["effect"]["pp"] < -MAX_CLEAN_DROP - 1e-9:
        failures.append("clean_regression")
    if min(reports[name]["effect"]["pp"] for name in HARD_WEIGHTS) < -MAX_OTHER_DROP - 1e-9:
        failures.append("hard_condition_regression")
    if phase == "confirm" and min(hard["ci95_pp"][0], shift["ci95_pp"][0]) <= 0:
        failures.append("paired_ci_includes_zero")
    return {"decision": f"stop_{phase}" if failures else
            "advance_to_confirm" if phase == "screen" else "eligible_for_candidate",
            "reasons": failures, "conditions": reports, "hard_effect": hard, "heldout_effect": shift}
