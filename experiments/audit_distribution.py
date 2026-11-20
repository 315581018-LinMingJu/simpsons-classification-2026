"""以 CPU 稽核 Simpsons Train／Test 低階影像分布與 V3／V5 改判。"""

import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from compare_v1_v2 import SPLIT_SHA256
from distribution_audit.compare import (COMBO_FIELDS, QUANTILE_POINTS, combination_rows,
                                       diagnostics_priorities, edge_bin, grouped_changes,
                                       mark_novelty)
from distribution_audit.features import FEATURES, THUMBNAIL_SIZE, inspect_image, quantiles
from distribution_audit.novelty import COMBO_COLUMNS
from v2.data_checks import load_split, read_json
from v3.reports import read_reference, read_test_images
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v4.analysis import aspect_bin, save_sheet, size_summary
from v4.geometry import seeded
from v5.background import DARK_MAX


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v3-submission", type=Path,
                        default=Path("runs/v3_tta_155/submission_v3_multiscale.csv"))
    parser.add_argument("--v5-run", type=Path, default=Path("runs/v5_background"))
    parser.add_argument("--v5-submission", type=Path)
    parser.add_argument("--workers", type=int, default=4, help="0 使用單進程")
    parser.add_argument("--output", type=Path, default=Path("runs/distribution_audit"))
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    audit_root = Path("runs/distribution_audit").absolute()
    if (not args.output.resolve().is_relative_to(audit_root.resolve()) or audit_root.is_symlink()
            or any(args.output.resolve().is_relative_to(path.resolve())
                   for path in (args.data, args.v2_run, args.v5_run))):
        parser.error("--output must be an independent path under runs/distribution_audit")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    args.v5_submission = args.v5_submission or args.v5_run / "submission_v5.csv"
    return args


def collect(paths, workers, split):
    def consume(iterator):
        result = []
        for reading in iterator:
            result.append(reading)
            if len(result) % 10000 == 0:
                print(f"{split}: {len(result)}/{len(paths)} files", flush=True)
        return result

    if not workers:
        return consume(map(inspect_image, paths))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return consume(pool.map(inspect_image, paths, chunksize=64))


def class_summary(rows, classes, decoded):
    counts = Counter(row.label for row in rows)
    valid = Counter(row.label for row, reading in zip(rows, decoded, strict=True) if reading.values is not None)
    results = [(name, counts[index], valid[index]) for index, name in enumerate(classes)]
    count_values = [count for _, count, _ in results]
    lo, med, hi = quantiles(count_values, (0, 0.5, 1))
    return results, {"min": int(lo), "median": med, "max": int(hi), "max_over_min": hi / lo,
                     "classes_below_one_fifth_median": sum(value < med / 5 for value in count_values),
                     "lowest_frequency": sorted(results, key=lambda row: row[1])[:5],
                     "highest_frequency": sorted(results, key=lambda row: -row[1])[:5]}


def split_summary(readings):
    headers = [item for item in readings if item.actual_format]
    decoded = [item for item in readings if item.values is not None]
    if not decoded:
        raise ValueError("Audit needs decodable images in both splits")
    nonblack = sum(reading.values[0] < 0.5 for reading in decoded)
    mismatch = sum(item.extension_match is False for item in headers)
    sizes = size_summary({str(i): (row.width, row.height) for i, row in enumerate(headers)})
    sizes.update({
        "width_range": [min(row.width for row in headers), max(row.width for row in headers)],
        "height_range": [min(row.height for row in headers), max(row.height for row in headers)],
        "aspect_groups": dict(Counter(aspect_bin(row.width, row.height) for row in headers)),
    })
    return {
        "files": len(readings), "readable_headers": len(headers), "decoded_images": len(decoded),
        "decode_or_header_errors": len(readings) - len(decoded),
        "formats_by_readable_header": dict(Counter(item.actual_format for item in headers)),
        "extension_mismatches": mismatch, "extension_mismatch_pct": 100 * mismatch / len(headers),
        "sizes": sizes,
        "non_mostly_black_edge_count": nonblack,
        "non_mostly_black_edge_pct": 100 * nonblack / len(decoded),
        "feature_medians": {name: quantiles([reading.values[i] for reading in decoded], (0.5,))[0]
                            for i, name in enumerate(FEATURES)},
    }


def technical_rows(train, test):
    counts = {
        name: Counter((row.actual_format, row.width, row.height) for row in readings if row.actual_format)
        for name, readings in (("train", train), ("test", test))
    }
    tr, te = counts["train"], counts["test"]
    return [(fmt, width, height, tr[key], te[key])
            for key in sorted(tr.keys() | te.keys(), key=lambda item: (-te[item], item))
            for fmt, width, height in (key,)]


def montage(output, paths, train, ranked, strongest_combo):
    rows = []
    review = []
    manual_columns = ("selection", "test_id", "photo_background", "small_character", "off_center",
                      "rotation", "perspective", "crop", "multi_character", "negative_colors",
                      "noise", "border", "notes")
    for group, picked in (
        ("largest visual tail", ranked[:6]),
        ("sparse Train combination", [row for row in ranked if row["signature"] == strongest_combo][:6]),
        ("random Test", seeded(2026, "audit review").sample(ranked, min(6, len(ranked)))),
    ):
        if picked:
            rows.append([(f"{group} | ID {row['id']} | {row['visual_tail_distance']:.2f}", paths[row["id"]])
                         for row in picked])
            review.extend((group, row["id"], *([""] * (len(manual_columns) - 2))) for row in picked)
    tr = seeded(2026, "audit train review").sample(train, min(6, len(train)))
    rows.append([("Train reference", row.path) for row in tr])
    save_sheet(output / "test_visual_outliers.png", rows)
    publish_csv(output / "manual_review.csv", manual_columns, review)


def audit(args):
    start = time.perf_counter()
    train_root = args.data / "train/train/train"
    classes = read_json(args.v2_run / "classes.json")
    settings_v5 = read_json(args.v5_run / "settings.json")
    split_path = args.v2_run / "split.csv"
    split_hash = file_sha256(split_path)
    if (settings_v5["classes"] != classes or settings_v5["split_sha256"] != split_hash
            or split_hash != SPLIT_SHA256):
        raise ValueError("Expected the original V2/V5 split and ordered 50-class mapping")
    rows = load_split(train_root, classes, split_path.read_bytes())
    tests = read_test_images(args.data / "test_images/test_images", args.data / "sample_submission.csv")
    test_ids = [item for _, item in tests]
    v3, v5 = (read_reference(path, test_ids, classes)
              for path in (args.v3_submission, args.v5_submission))
    changed = {item: v3[item] != v5[item] for item in test_ids}
    print("Scanning full Train and Test images on CPU……", flush=True)
    train_all = collect([row.path for row in rows], args.workers, "train")
    test_all = collect([path for path, _ in tests], args.workers, "test")
    train_stats, test_stats = split_summary(train_all), split_summary(test_all)
    train = [reading for reading in train_all if reading.values is not None]
    test = [(item, reading) for item, reading in zip(test_ids, test_all, strict=True)
            if reading.values is not None]
    by_id = dict(zip(test_ids, test_all, strict=True))
    novel, support, thresholds = mark_novelty(train, test)
    by_novelty = {row["id"]: row for row in novel}
    combo = list(combination_rows(support, novel, changed, len(train)))
    combo_columns = (*COMBO_FIELDS, "train_count", "test_count", "train_pct", "test_pct",
                     "train_support", "v3_v5_changed", "v3_v5_changed_pct")
    combo = [dict(zip(combo_columns, row, strict=True)) for row in combo]
    sparse = [row for row in combo if row["train_support"] in ("unseen", "rare") and row["test_count"]]
    ranked = sorted(novel, key=lambda row: (-row["visual_tail_distance"], row["train_combo_count"], row["id"]))
    groups = []
    for item, reading in by_id.items():
        novelty = by_novelty.get(item)
        labels = {
            "actual_format": reading.actual_format or "unknown",
            "aspect": aspect_bin(reading.width, reading.height) if reading.width else "unknown",
            "outer_edge": edge_bin(reading.values[0]) if reading.values is not None else "unknown",
            "combo_support": novelty["combo_support"] if novelty else "unknown",
            "visual_tail": novelty["visual_band"] if novelty else "unknown",
            "train_size_support": ("unknown" if not novelty else
                                   "unseen" if novelty["unseen_size"] else "observed"),
        }
        if novelty:
            labels.update(zip(COMBO_FIELDS[1:], novelty["signature"][1:], strict=True))
        groups.append((item, labels))
    grouped = grouped_changes(groups, changed)
    classes_table, class_counts = class_summary(rows, classes, train_all)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    publish_csv(output / "train_class_counts.csv", ("character", "files", "decoded"), classes_table)
    publish_csv(output / "format_size_counts.csv", ("format", "width", "height", "train", "test"),
                technical_rows(train_all, test_all))
    publish_csv(output / "feature_quantiles.csv",
                ("split", "feature", "p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                ((group, feature, *quantiles([reading.values[i] for reading in readings], QUANTILE_POINTS))
                 for group, readings in (("train", train), ("test", [row for _, row in test]))
                 for i, feature in enumerate(FEATURES)))
    publish_csv(output / "feature_combinations.csv", combo_columns, (row.values() for row in combo))
    group_columns = ("dimension", "group", "images", "changed", "changed_pct",
                     "share_of_all_changes_pct", "rate_ratio_vs_all_test")
    publish_csv(output / "v3_v5_by_group.csv", group_columns, grouped)
    publish_csv(output / "test_novelty.csv",
                ("id", "actual_format", "width", "height", "extension_match", *FEATURES, *COMBO_COLUMNS,
                 "train_combo_count", "combo_support", "unseen_format", "unseen_size", "unseen_format_size",
                 "visual_tail_distance", "leading_tail_feature", "visual_band",
                 "v3_prediction", "v5_prediction", "changed"),
                ((item, by_id[item].actual_format, by_id[item].width, by_id[item].height,
                  by_id[item].extension_match, *by_id[item].values, *row["signature"],
                  row["train_combo_count"], row["combo_support"], row["unseen_format"], row["unseen_size"],
                  row["unseen_format_size"], row["visual_tail_distance"], row["leading_tail_feature"],
                  row["visual_band"], classes[v3[item]], classes[v5[item]], changed[item])
                 for row in ranked for item in (row["id"],)))
    publish_csv(output / "unreadable_images.csv", ("source", "relative_path", "format", "error"),
                ((source, row.path.relative_to(args.data).as_posix(), row.actual_format, row.error)
                 for source, images in (("train", train_all), ("test", test_all))
                 for row in images if row.error))
    montage(output, {item: reading.path for item, reading in test}, train, ranked,
            tuple(sparse[0][key] for key in COMBO_FIELDS) if sparse else None)

    summary = {
        "provenance": {"split_sha256": split_hash,
                       "v3_submission_sha256": file_sha256(args.v3_submission),
                       "v5_submission_sha256": file_sha256(args.v5_submission),
                       "workers": args.workers, "thumbnail_max_edge": THUMBNAIL_SIZE,
                       "black_rgb_channel_max": DARK_MAX},
        "labels": {"allowed_characters": classes, "test_ground_truth_available": False,
                   "train_class_imbalance": class_counts},
        "train": train_stats, "test": test_stats,
        "reference_fit": thresholds,
        "test_novelty": {
            key: sum(row[key] for row in novel)
            for key in ("unseen_format", "unseen_size", "unseen_format_size")
        } | {"visual_support": dict(Counter(row["combo_support"] for row in novel)),
             "top_rare_or_unseen_combinations": sparse[:8],
             "highest_visual_tail_ids": [row["id"] for row in ranked[:12]]},
        "v3_v5_changes": {"test_images": len(test_ids), "changed": sum(changed.values()),
                          "changed_pct": 100 * sum(changed.values()) / len(test_ids),
                          "visual_band_groups": [dict(zip(group_columns, row, strict=True))
                                                 for row in grouped if row[0] == "visual_tail"]},
        "priority_signals": diagnostics_priorities(train, [row for _, row in test]),
        "seconds": time.perf_counter() - start,
    }
    publish(output / "summary.json", json_bytes(summary))
    print(f"Distribution audit: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    audit(arguments())
