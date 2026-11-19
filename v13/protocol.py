"""預先鎖定同預算比較、加跑與候選 Confirm 門檻。"""

from v12.protocol import comparison

ENDPOINTS = ("hard_effect", "heldout_effect")


def wins_deployment(report):
    return (not report["reasons"] and
            all(report[key]["ci95_pp"][0] > 0 for key in ENDPOINTS))


def base_advantage(report):
    gains = [report[key] for key in ENDPOINTS]
    return (min(item["pp"] for item in gains) >= -.5 - 1e-9
            and any(item["pp"] >= 1 - 1e-9 and item["ci95_pp"][0] > 0 for item in gains)
            and report["conditions"]["clean"]["effect"]["pp"] >= -.5 - 1e-9)


def extend_reasons(tiny, base, relative):
    failures = []
    if min(report["accuracy_pct"][arm]["clean"] for arm, report in (("tiny", tiny), ("base", base))) < 70:
        failures.append("weak_clean")
    if base["hard_effect"]["pp"] < -15 or base["heldout_effect"]["pp"] < -18:
        failures.append("base_too_far_from_v8")
    gains = [relative[key]["pp"] for key in ENDPOINTS]
    if max(gains) < 1 - 1e-9 or min(gains) < -3 or relative["conditions"]["clean"]["effect"]["pp"] < -2:
        failures.append("no_base_signal")
    return failures


def paired_report(phase, before, after, cohorts, seed, baseline_arm, adapted_arm):
    result = comparison(phase, before, after, cohorts, seed)
    conditions = result.pop("conditions")
    accuracy = {baseline_arm: {}, adapted_arm: {}}
    for name, values in conditions.items():
        accuracy[baseline_arm][name] = values.pop("v8_accuracy_pct")
        accuracy[adapted_arm][name] = values.pop("v12_accuracy_pct")
    return {**result, "baseline_arm": baseline_arm, "adapted_arm": adapted_arm,
            "conditions": conditions, "accuracy_pct": accuracy}


def screen_decision(phase, tiny, base, relative):
    if wins_deployment(base) and base_advantage(relative):
        return {"decision": "advance_to_confirm", "candidate": "base", "reasons": []}
    if wins_deployment(tiny):
        return {"decision": "advance_to_confirm", "candidate": "tiny", "reasons": []}
    blocked = extend_reasons(tiny, base, relative)
    if phase == "pilot" and not blocked:
        return {"decision": "advance_to_extend", "candidate": None, "reasons": []}
    return {"decision": "stop", "candidate": None, "reasons": blocked or ["no_clear_pairwise_win"]}
