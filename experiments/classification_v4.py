"""V4 資料分布診斷與同圖配對的困難驗證。"""

import argparse
import time
from collections import Counter
from pathlib import Path

from PIL import Image, __version__ as pillow_version

from classification_v2 import create_model, load_checkpoint, transforms_v2
from classification_v3 import validation_subset
from compare_v1_v2 import load_comparison_split
from v2.data_checks import read_json
from v3.inference import VIEWS, Views, infer_loader
from v3.reports import read_test_images
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v4.analysis import (QUANTILES, aspect_bin, changes_by_aspect, compare, image_sizes,
                         quantiles, read_v3_changes, save_sheet, size_summary, thumbnail_signals)
from v4.geometry import (CONDITIONS, METHODS, ChallengeImages, layout, render, retained_fraction,
                         seeded, stratified_samples)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("diagnostic", "benchmark"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v3-run", type=Path, default=Path("runs/v3_tta_155"))
    parser.add_argument("--output", type=Path, default=Path("runs/v4_geometry"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-size", type=int, default=384, help="每組影像抽樣數")
    parser.add_argument("--per-class", type=int, default=24, help="困難驗證每類抽樣數")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.sample_size, args.per_class, args.batch_size) < 1 or args.workers < 0:
        parser.error("sample sizes and batch size must be positive; workers must be nonnegative")
    output = args.output.resolve()
    protected = (args.data, args.v1_run, args.v2_run, args.v3_run)
    if (output.parent != Path("runs").resolve() or not output.name.startswith("v4_")
            or any(output.is_relative_to(path.resolve()) for path in protected)):
        parser.error("--output must be an independent runs/v4_... directory")
    return args


def diagnostic(args, classes, rows, train_root, test, provenance):
    output = args.output / "diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    sizes = image_sizes(test)
    all_test = {**size_summary(sizes),
                "aspect_bin_counts": dict(Counter(aspect_bin(*size) for size in sizes.values()))}
    groups = {"train": [], "val": [], "test": [(path, item, "") for path, item in test]}
    for row in rows:
        groups[row.part].append((row.path, row.path.relative_to(train_root).as_posix(), classes[row.label]))
    summaries, sheets, signal_rows = {}, [], []
    for group, pool in groups.items():
        subset = seeded(args.seed, group).sample(pool, min(args.sample_size, len(pool)))
        dimensions = image_sizes((path, item) for path, item, _ in subset)
        measurements = []
        sheet = []
        for path, item, label in subset:
            width, height = dimensions[item]
            weak_color, edge_sd, orientation = thumbnail_signals(path)
            measurements.append((weak_color, edge_sd, orientation))
            signal_rows.append((group, item, label, width, height, width / height,
                                weak_color, edge_sd, orientation))
            if len(sheet) < 6:
                caption = label.replace("_", " ") if label else f"ID {item} (unlabeled)"
                sheet.append((f"{group} {width}x{height}: {caption}", path))
        weak_color, edge_sd, orientation = zip(*measurements, strict=True)
        summaries[group] = {**size_summary(dimensions), "pool_size": len(pool),
                            "weak_color_fraction_quantiles": quantiles(weak_color),
                            "border_luma_std_quantiles": quantiles(edge_sd),
                            "nonstandard_exif_orientation_count": sum(value != 1 for value in orientation)}
        sheets.append(sheet)

    files = {item: path for path, item in test}
    ranked = sorted(sizes, key=lambda item: sizes[item][0] / sizes[item][1])
    random_ids = seeded(args.seed, "test_review").sample(list(sizes), min(24, len(sizes)))
    review = [("random", random_ids), ("aspect_extreme", list(dict.fromkeys(ranked[:3] + ranked[-3:])))]
    source = args.v3_run / "test_diagnostics.csv"
    prior = args.v3_run / "test_summary.json"
    v3_status = "no_full_report"
    if source.is_file() and prior.is_file():
        settings = read_json(prior).get("settings", {})
        if (settings.get("image_signature") == provenance["test_image_signature"]
                and settings.get("v2_checkpoint_sha256") == provenance["v2_checkpoint_sha256"]):
            changes = read_v3_changes(source, files)
            publish_csv(output / "v3_changes_by_aspect.csv",
                        ("aspect", "method", "samples", "changed_pct", "v1_v2_changed_pct", "mean_confidence"),
                        changes_by_aspect(sizes, changes))
            v3_status = "matched"
            ms = changes["multiscale"]
            low = sorted(ms, key=lambda item: ms[item][2])[:6]
            changed = [item for item in ms if ms[item][0]]
            picked = seeded(args.seed, "v3_changed").sample(changed, min(6, len(changed)))
            review.append(("multiscale_low_confidence", low))
            if picked:
                review.append(("multiscale_changed", picked))
        else:
            v3_status = "provenance_mismatch"

    publish_csv(output / "sample_stats.csv",
                ("split", "id_or_relative_path", "known_label", "width", "height", "aspect_ratio",
                 "weak_color_fraction", "border_luma_std", "exif_orientation"), signal_rows)
    evidence = []
    for group, ids in review:
        cells = [(f"{group.replace('_', ' ')} ID {item}: {sizes[item][0]}x{sizes[item][1]}", files[item])
                 for item in ids]
        evidence.extend(cells[offset:offset + 6] for offset in range(0, len(cells), 6))
    publish_csv(output / "test_review.csv",
                ("selection", "id", "offcenter", "small_subject", "multiple_characters", "frame",
                 "perspective", "complex_background", "notes"),
                ((group, item, "", "", "", "", "", "", "") for group, ids in review for item in ids))
    save_sheet(output / "split_montage.png", sheets)
    save_sheet(output / "test_cases.png", evidence)
    report = {
        "provenance": provenance, "seed": args.seed, "sample_size": args.sample_size,
        "quantiles": QUANTILES, "sampled_splits": summaries, "full_test_header_scan": all_test,
        "v3_diagnostics": {"status": v3_status, "source": str(source),
                           "sha256": file_sha256(source) if v3_status == "matched" else None},
    }
    publish(output / "diagnostic.json", json_bytes(report))
    print(f"Diagnostic: {output} (V3 overlap report: {v3_status})", flush=True)


def paired_data(args, rows, train_root, class_count):
    signature = image_signature(row.path for row in rows)
    local = args.output / "validation_subset.json"
    previous = args.v3_run / "validation_subset.json"
    source = "v4"
    if not local.exists() and previous.is_file():
        report = read_json(previous)
        if report.get("image_signature") == signature:
            publish(local, json_bytes(report))
            source = str(previous)
    samples, targets, counts = validation_subset(args.output, rows, signature, train_root)
    selected = stratified_samples(samples, targets, args.per_class, args.seed, class_count)
    return selected, {item: targets[item] for _, item in selected}, counts, signature, source


def benchmark_examples(output, samples, targets, classes, predictions, aspects, seed):
    paths = {item: path for path, item in samples}
    baseline = predictions["offcenter", "baseline"]
    multiscale = predictions["offcenter", "multiscale"]
    wins = [item for item, y in targets.items() if baseline[item].label != y == multiscale[item].label]
    losses = [item for item, y in targets.items() if multiscale[item].label != y == baseline[item].label]
    joint = [item for item, y in targets.items()
             if baseline[item].label != y and multiscale[item].label != y]
    examples = []
    for group, items in (("Win", wins), ("Loss", losses), ("Both wrong", joint)):
        for item in seeded(seed, group).sample(items, min(2, len(items))):
            with Image.open(paths[item]) as file:
                image = file.convert("RGB")
            outcomes = [f"{group} ({condition})"
                        f"\nV2: {classes[predictions[condition, 'baseline'][item].label]}"
                        f"\nMS: {classes[predictions[condition, 'multiscale'][item].label]}"
                        for condition in CONDITIONS]
            outcomes[0] = f"{item}\nLabel: {classes[targets[item]]}\n" + outcomes[0]
            examples.append([(title.replace("_", " "), render(image, condition, aspects, seed, item))
                             for condition, title in zip(CONDITIONS, outcomes, strict=True)])
    if not examples:
        item = next(iter(targets))
        with Image.open(paths[item]) as file:
            image = file.convert("RGB")
        examples = [[(condition, render(image, condition, aspects, seed, item)) for condition in CONDITIONS]]
    save_sheet(output / "paired_examples.png", examples)


def benchmark(args, classes, rows, train_root, provenance):
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a confirmed available CUDA GPU")
    diagnostic_path = args.output / "diagnostic/diagnostic.json"
    if not diagnostic_path.is_file():
        raise ValueError("Run diagnostic first to measure the current test aspect distribution")
    report = read_json(diagnostic_path)
    if report["provenance"] != provenance:
        raise ValueError("Diagnostic provenance changed; start a new runs/v4_... directory")
    aspects = report["full_test_header_scan"]["aspect_width_over_height_quantiles"]
    checkpoint = args.v2_run / "best.pt"
    _, evaluation = transforms_v2()
    if read_json(args.v2_run / "run_config.json")["eval_transform"] != repr(evaluation):
        raise ValueError("V2 evaluation transform differs from the current runtime")

    output = args.output / "benchmark"
    output.mkdir(exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    batch_size, workers = args.batch_size, args.workers
    samples, targets, counts, signature, subset_source = paired_data(args, rows, train_root, len(classes))
    predictions, seconds = {}, {}
    model = create_model().eval()
    load_checkpoint(model, checkpoint, classes, provenance["split_sha256"])
    for condition in CONDITIONS:
        for method in METHODS:
            started = time.perf_counter()
            loader = torch.utils.data.DataLoader(
                ChallengeImages(samples, condition, aspects, args.seed, Views(method, evaluation)),
                batch_size=batch_size, num_workers=workers, pin_memory=True,
                prefetch_factor=1 if workers else None,
            )
            results = {}
            for batch in infer_loader(model, loader):
                results.update(batch)
            if results.keys() != targets.keys():
                raise ValueError("Benchmark prediction IDs differ from the fixed validation subset")
            predictions[condition, method] = results
            seconds[f"{condition}/{method}"] = time.perf_counter() - started
            print(f"{condition}/{method}: {len(results)} 張", flush=True)

    scores, outcome = compare(targets, predictions, args.seed)
    geometry = {}
    for item, size in image_sizes(samples).items():
        canvas, patch, center, offcenter, _ = layout(size, aspects, args.seed, item)
        geometry[item] = (canvas[0], canvas[1], patch[0], patch[1], *center, *offcenter,
                          retained_fraction(canvas, patch, center),
                          retained_fraction(canvas, patch, offcenter))
    publish_csv(output / "validation_geometry.csv",
                ("relative_path", "label", "canvas_width", "canvas_height", "patch_width", "patch_height",
                 "center_x", "center_y", "offcenter_x", "offcenter_y", "center_retained", "offcenter_retained"),
                ((item, classes[targets[item]], *values) for item, values in geometry.items()))
    publish_csv(output / "paired_predictions.csv",
                ("relative_path", "label", "condition", "method", "prediction", "correct", "confidence"),
                ((item, classes[y], condition, method, classes[results[item].label],
                  int(results[item].label == y), results[item].confidence)
                 for (condition, method), results in predictions.items() for item, y in targets.items()))
    publish_csv(output / "scores.csv", scores[0], (row.values() for row in scores))
    benchmark_examples(output, samples, targets, classes, predictions, aspects, args.seed)
    settings = {
        "seed": args.seed, "requested_per_class": args.per_class,
        "selected_by_class": {classes[label]: count for label, count in Counter(targets.values()).items()},
        "duplicate_counts": counts, "validation_subset_source": subset_source,
        "train_image_signature": signature, "diagnostic_sha256": file_sha256(diagnostic_path),
        "batch_size": batch_size, "workers": workers, "methods": {method: VIEWS[method] for method in METHODS},
        "torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": pillow_version,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "eval_transform": repr(evaluation), "torch_deterministic_cudnn": True,
    }
    publish(output / "benchmark.json", json_bytes({
        "provenance": provenance, "settings": settings, "scores": scores, "primary_outcome": outcome,
        "elapsed_seconds_by_condition_and_method": seconds,
        "source_image_retained_quantiles": {
            "center": quantiles([row[-2] for row in geometry.values()]),
            "offcenter": quantiles([row[-1] for row in geometry.values()]),
        },
    }))
    print(f"Benchmark: {outcome['decision']}. See {output / 'benchmark.json'}", flush=True)


def main():
    args = get_args()
    train_root = args.data / "train/train/train"
    classes, rows, split_hash = load_comparison_split(train_root, args.v1_run, args.v2_run)
    test = read_test_images(args.data / "test_images/test_images", args.data / "sample_submission.csv")
    provenance = {
        "split_sha256": split_hash,
        "test_image_signature": image_signature(path for path, _ in test),
        "v2_checkpoint_sha256": file_sha256(args.v2_run / "best.pt"),
    }
    if args.stage == "diagnostic":
        diagnostic(args, classes, rows, train_root, test, provenance)
    else:
        benchmark(args, classes, rows, train_root, provenance)


if __name__ == "__main__":
    main()
