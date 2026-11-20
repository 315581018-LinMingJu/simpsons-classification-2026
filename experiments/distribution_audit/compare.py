"""Train-only 視覺分箱、偏移排序與模型改判交叉統計。"""

from collections import Counter, defaultdict

import numpy as np

from distribution_audit.features import FEATURES, quantiles

COMBO_FIELDS = ("black_edge", "weak_color", "brightness", "contrast", "fine_detail")
SCORE_FEATURES = (
    "black_edge_fraction", "weak_color_fraction", "brightness", "contrast", "fine_detail",
)
QUANTILE_POINTS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def edge_bin(value):
    if value >= 0.5:
        return "mostly_black"
    return "mixed" if value >= 0.1 else "little_black"


def color_bin(value):
    if value >= 0.95:
        return "very_weak_color"
    return "mixed_color" if value >= 0.45 else "colored"


def learned_bin(value, bounds):
    if value < bounds[0]:
        return "low"
    return "high" if value > bounds[1] else "mid"


def fit_visual_reference(train):
    matrix = np.asarray([reading.values for reading in train])
    bins = {name: quantiles(matrix[:, FEATURES.index(name)], (0.20, 0.80))
            for name in COMBO_FIELDS[2:]}
    tails = {}
    for name in SCORE_FEATURES:
        lo, q1, q3, hi = quantiles(matrix[:, FEATURES.index(name)], (0.01, 0.25, 0.75, 0.99))
        tails[name] = (lo, hi, max(q3 - q1, 0.01))
    return bins, tails


def visual_signature(reading, bins):
    values = dict(zip(FEATURES, reading.values, strict=True))
    return (
        edge_bin(values["black_edge_fraction"]),
        color_bin(values["weak_color_fraction"]),
        *(learned_bin(values[name], bins[name]) for name in COMBO_FIELDS[2:]),
    )


def tail_distance(reading, tails):
    values = dict(zip(FEATURES, reading.values, strict=True))
    components = {feature: min(8.0, max(0.0, lo - values[feature], values[feature] - hi) / scale)
                  for feature, (lo, hi, scale) in tails.items()}
    return sum(components.values()), sorted(components, key=components.get, reverse=True)


def support_type(count, train_count):
    if count == 0:
        return "unseen"
    if count / train_count <= 0.001:
        return "rare"
    return "observed"


def mark_novelty(train, test):
    bins, tails = fit_visual_reference(train)
    signatures = Counter(visual_signature(reading, bins) for reading in train)
    formats = {reading.actual_format for reading in train}
    dimensions = {(reading.width, reading.height) for reading in train}
    technical = {(reading.actual_format, reading.width, reading.height) for reading in train}
    result = []
    for item, reading in test:
        signature = visual_signature(reading, bins)
        score, reasons = tail_distance(reading, tails)
        support = signatures[signature]
        result.append({
            "id": item, "signature": signature, "train_combo_count": support,
            "combo_support": support_type(support, len(train)),
            "unseen_format": reading.actual_format not in formats,
            "unseen_size": (reading.width, reading.height) not in dimensions,
            "unseen_format_size": (reading.actual_format, reading.width, reading.height) not in technical,
            "visual_tail_distance": score, "leading_tail_feature": reasons[0] if score else "",
        })
    cutoff = float(np.quantile([row["visual_tail_distance"] for row in result], 0.90))
    for row in result:
        score = row["visual_tail_distance"]
        row["visual_band"] = ("inside_reference" if not score else
                              "upper_tail" if score >= cutoff else "other_outside_reference")
    settings = {"bins_from_train_p20_p80": bins, "tail_train_p01_p99_iqr": tails,
                "upper_tail_test_p90_threshold": cutoff, "rare_train_frequency_at_most": 0.001}
    return result, signatures, settings


def combination_rows(train_signatures, novel, changed, train_count):
    tests = Counter(row["signature"] for row in novel)
    flips = Counter(row["signature"] for row in novel if changed[row["id"]])
    for key in sorted(set(train_signatures) | set(tests), key=lambda key: (-tests[key], key)):
        count, test_count = train_signatures[key], tests[key]
        yield (*key, count, test_count, 100 * count / train_count, 100 * test_count / len(novel),
               support_type(count, train_count), flips[key],
               100 * flips[key] / test_count if test_count else None)


def grouped_changes(groups, changed):
    counts = defaultdict(Counter)
    total_changes = sum(changed.values())
    overall = total_changes / len(changed)
    for item, labels in groups:
        for dimension, group in labels.items():
            counts[dimension, str(group)][bool(changed[item])] += 1
    result = []
    for (dimension, group), counter in sorted(counts.items()):
        flips = counter[True]
        count = sum(counter.values())
        result.append((dimension, group, count, flips, 100 * flips / count,
                       100 * flips / total_changes if total_changes else 0,
                       (flips / count) / overall if overall else None))
    return result


def diagnostics_priorities(train, test):
    train_matrix = np.asarray([reading.values for reading in train])
    test_matrix = np.asarray([reading.values for reading in test])
    specs = (
        ("outer_black_support", "black_edge_fraction", lambda x: x < 0.5,
         "人工確認照片背景／圖塊，優先對照 V5 的背景適應與區塊取景。"),
        ("weak_color_support", "center_weak_color_fraction", lambda x: x >= 0.45,
         "人工區分局部灰階、反色及低飽和，再測原圖與校色視角的配對改判。"),
        ("high_frequency_support", "fine_detail",
         lambda x: x > np.quantile(train_matrix[:, FEATURES.index("fine_detail")], 0.95),
         "用拼圖區分紋理、真實雜訊和壓縮痕跡，再決定是否驗證去噪。"),
    )
    results = []
    for name, feature, rule, action in specs:
        index = FEATURES.index(feature)
        train_pct = 100 * float(rule(train_matrix[:, index]).mean())
        test_pct = 100 * float(rule(test_matrix[:, index]).mean())
        results.append({"signal": name, "train_pct": train_pct, "test_pct": test_pct,
                        "test_minus_train_pp": test_pct - train_pct, "next_check": action})
    return sorted(results, key=lambda item: -abs(item["test_minus_train_pp"]))
