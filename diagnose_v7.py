"""稽核真實 Test 外觀，並用既有 V6 驗證保守去顆粒。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from classification_v2 import create_model, load_checkpoint, transforms_v2
from classification_v5 import METHOD
from classification_v6 import validation_view
from compare_v1_v2 import SPLIT_SHA256, load_comparison_split
from ensemble.data import fresh_validation
from v2.data_checks import CLASS_COUNT, read_json
from v3.inference import Views, infer_view_batches
from v3.reports import read_reference, read_test_images
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v4.analysis import paired_effect
from v6.composite import Targets
from v7.impulse import CONDITIONS, MIN_TRIGGER_FRACTION, add_impulses, clean_impulses, impulse_mask
from v7.review import (audit_rows, cached_v6, inspect_test, review_groups, summarize_groups,
                       write_montages)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("review", "validate"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v5-run", type=Path, default=Path("runs/v5_background"))
    parser.add_argument("--v6-run", type=Path, default=Path("runs/v6_composite/experiment"))
    parser.add_argument("--v6-submission", type=Path)
    parser.add_argument("--audit-run", type=Path, default=Path("runs/distribution_audit"))
    parser.add_argument("--v6-probabilities", type=Path, default=Path("runs/v6_ensemble/test/v6.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/v7_diagnostics"))
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.per_class < 4 or args.batch_size < 1 or args.workers < 0:
        parser.error("per-class >= 4, batch-size >= 1 and workers >= 0 are required")
    result = args.output.resolve()
    if (result.parent != Path("runs").resolve() or not result.name.startswith("v7_diagnostics")
            or any(result.is_relative_to(path.resolve()) for path in
                   (args.data, args.v1_run, args.v2_run, args.v5_run, args.v6_run, args.audit_run))):
        parser.error("--output must be an independent runs/v7_diagnostics... directory")
    args.v6_submission = args.v6_submission or args.v6_run / "submission_v6.csv"
    return args


def sources(args):
    v6 = read_json(args.v6_run / "settings.json")
    classes = read_json(args.v2_run / "classes.json")
    checkpoint = args.v6_run / "v6.pt"
    digest = file_sha256(checkpoint)
    split = file_sha256(args.v2_run / "split.csv")
    if (len(classes) != CLASS_COUNT or len(set(classes)) != CLASS_COUNT
            or v6["classes"] != classes or v6["split_sha256"] != split or split != SPLIT_SHA256
            or v6["method"] != METHOD
            or read_json(args.v6_run / "validation.json")["arms"]["v6"]["checkpoint_sha256"] != digest):
        raise ValueError("V6 checkpoint, class order or original split differs from its completed run")
    test = read_test_images(args.data / "test_images/test_images",
                            args.data / "sample_submission.csv")
    ids = [item for _, item in test]
    archived = read_reference(args.v6_submission, ids, classes)
    info = {
        "test_image_signature": image_signature(path for path, _ in test),
        "split_sha256": split, "v6_checkpoint_sha256": digest,
        "v6_submission_sha256": file_sha256(args.v6_submission),
    }
    return classes, checkpoint, v6, test, ids, archived, info


def scan(paths, workers):
    def collect(values):
        results = []
        for value in values:
            results.append(value)
            if len(results) % 2000 == 0:
                print(f"Test impulse scan: {len(results)}/{len(paths)}", flush=True)
        return results

    if workers == 0:
        return collect(map(inspect_test, paths))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return collect(pool.map(inspect_test, paths, chunksize=32))


def review(args, classes, test, ids, archived, info, probabilities, cache_state):
    out = args.output / "review"
    if out.exists():
        raise FileExistsError(out)
    audit = read_json(args.audit_run / "summary.json")
    if (audit["provenance"]["split_sha256"] != info["split_sha256"]
            or audit["labels"]["allowed_characters"] != classes
            or audit["test"]["files"] != len(ids)):
        raise ValueError("Distribution audit does not match the dataset")
    audit_path = args.audit_run / "test_novelty.csv"
    rows = audit_rows(audit_path, ids, audit["test"]["decoded_images"])
    files = {item: path for path, item in test if item in rows}
    signals = dict(zip(files, scan(list(files.values()), args.workers), strict=True))
    groups = review_groups(signals, rows, args.seed)
    if not groups:
        raise ValueError("No reviewable Test images in the distribution audit")
    out.mkdir(parents=True, exist_ok=False)
    write_montages(out, groups, files, signals, rows, archived, classes, probabilities, ids)
    aggregates, mismatches = summarize_groups(ids, signals, rows, probabilities, archived)
    publish_csv(out / "test_impulse_signals.csv",
                ("id", "impulse_fraction", "filter_active", "visual_band", "combo_support"),
                ((item, signals[item], signals[item] >= MIN_TRIGGER_FRACTION,
                  rows[item]["visual_band"], rows[item]["combo_support"]) for item in signals))
    publish(out / "review.json", json_bytes({
        "provenance": {**info, "distribution_audit_sha256": file_sha256(audit_path)},
        "total_test": len(ids), "inspected": len(signals), "v6_probability_cache": cache_state,
        "cached_v6_vs_archived_mismatch_count": mismatches, "trigger_fraction": MIN_TRIGGER_FRACTION,
        "groups": aggregates, "review_ids": list(dict.fromkeys(item for _, _, members in groups for item in members)),
    }))
    print(out / "review.json", flush=True)


def base_image(path, item, condition, seed, targets):
    if condition == "test":
        with Image.open(path) as source:
            return source.convert("RGB")
    original = validation_view(path, item, "clean" if condition == "clean" else "heldout_composite",
                               seed, targets)
    return add_impulses(original, seed, item) if condition == "textured_impulse" else original


class Images:
    def __init__(self, samples, prepare, views, filtered):
        self.samples, self.prepare, self.views, self.filtered = samples, prepare, views, filtered

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item = self.samples[index]
        image = self.prepare(path, item)
        if self.filtered:
            image, _ = clean_impulses(image)
        return self.views(image), item


def infer(model, dataset, args):
    import torch

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
        prefetch_factor=1 if args.workers else None,
    )
    scores = {}
    for ids, probs, _, _ in infer_view_batches(model, loader):
        for item, row in zip(ids, probs.mean(axis=0), strict=True):
            scores[item] = score(row)
    if scores.keys() != {item for _, item in dataset.samples}:
        raise ValueError("Prediction IDs differ from the fixed sample")
    return scores


def score(probabilities):
    top2 = np.sort(probabilities)[-2:]
    return int(probabilities.argmax()), float(top2[-1]), float(top2[-1] - top2[-2])


def paired_inference(model, samples, args, prepare, views, raw=None):
    fractions = {item: float(impulse_mask(prepare(path, item)).mean()) for path, item in samples}
    active = {item for item, fraction in fractions.items() if fraction >= MIN_TRIGGER_FRACTION}
    if raw is None:
        raw = infer(model, Images(samples, prepare, views, False), args)
    filtered = raw.copy()
    if active:
        # 保留原批次邊界，只跳過完全不觸發的批次。
        blocks = [samples[i:i + args.batch_size] for i in range(0, len(samples), args.batch_size)]
        needed = [sample for block in blocks if any(item in active for _, item in block) for sample in block]
        changes = infer(model, Images(needed, prepare, views, True), args)
        filtered.update({item: changes[item] for item in active})
    return raw, filtered, fractions


def validation_result(labels, cases, seed):
    metrics = {}
    for condition in CONDITIONS:
        raw, filtered, fractions = cases[condition]
        a = np.array([raw[item][0] == y for item, y in labels.items()])
        b = np.array([filtered[item][0] == y for item, y in labels.items()])
        diff = b.astype(int) - a
        active = [item for item in labels if fractions[item] >= MIN_TRIGGER_FRACTION]
        metrics[condition] = {
            "macro_gain": paired_effect(diff, list(labels.values()), seed),
            "raw_accuracy_pct": 100 * float(a.mean()), "filtered_accuracy_pct": 100 * float(b.mean()),
            "n": len(labels), "activated": len(active), "wins": int((diff == 1).sum()),
            "losses": int((diff == -1).sum()),
            "active_prediction_changes": sum(raw[item][0] != filtered[item][0] for item in active),
        }
    noise = metrics["textured_impulse"]
    if min(metrics[key]["macro_gain"]["pp"] for key in ("clean", "texture_control")) < -0.5:
        decision = "guard_regression"
    elif noise["macro_gain"]["ci95_pp"][0] > 0:
        decision = "supports_labeled_impulse_followup"
    elif noise["macro_gain"]["ci95_pp"][1] <= 0:
        decision = "no_evidence_for_filter"
    else:
        decision = "inconclusive"
    return {"decision": decision, "conditions": metrics}


def validate(args, classes, checkpoint, v6, test, ids, archived, info, probabilities):
    import torch

    out = args.output / "validation"
    if out.exists():
        raise FileExistsError(out)
    review_report = read_json(args.output / "review/review.json")
    if any(review_report["provenance"].get(key) != value for key, value in info.items()):
        raise ValueError("V7 review does not match the current Test and V6 artifacts")
    if not torch.cuda.is_available():
        raise RuntimeError("Use a confirmed available GPU for validate")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, evaluation = transforms_v2()
    if v6["eval_transform"] != repr(evaluation):
        raise ValueError("Current evaluation transform differs from V6")

    train_root = args.data / "train/train/train"
    original, rows, split = load_comparison_split(train_root, args.v1_run, args.v2_run)
    signature = image_signature(row.path for row in rows)
    if (original != classes or split != info["split_sha256"] or
            v6["train_image_signature"] != signature or
            read_json(args.v5_run / "settings.json")["train_image_signature"] != signature):
        raise ValueError("Current Train/Val images differ from V5/V6")
    samples, y, _, _, exclusions = fresh_validation(rows, train_root, args.v5_run, classes,
                                                     args.per_class, args.seed)
    labels = {item: int(label) for (_, item), label in zip(samples, y, strict=True)}
    targets = Targets(**v6["targets_from_unlabeled_test"])
    out.mkdir(parents=True, exist_ok=False)
    views = Views(METHOD, evaluation)
    model = create_model().eval()
    load_checkpoint(model, checkpoint, classes, split)

    def for_condition(condition):
        return partial(base_image, condition=condition, seed=args.seed, targets=targets)

    results = {}
    for condition in CONDITIONS:
        print(f"V6 {condition} paired validation", flush=True)
        results[condition] = paired_inference(model, samples, args, for_condition(condition), views)
    report = validation_result(labels, results, args.seed)
    publish_csv(out / "paired_validation.csv",
                ("condition", "relative_path", "character", "raw_v6", "filtered_v6",
                 "impulse_fraction", "filter_active"),
                ((case, item, classes[label], classes[raw[item][0]], classes[filtered[item][0]],
                  fractions[item], fractions[item] >= MIN_TRIGGER_FRACTION)
                 for case, (raw, filtered, fractions) in results.items() for item, label in labels.items()))
    publish(out / "validation_report.json", json_bytes({
        **report, "provenance": info, "seed": args.seed, "per_class_limit": args.per_class,
        "trigger_fraction": MIN_TRIGGER_FRACTION, "validation_exclusions": exclusions,
        "method": METHOD, "gpu": torch.cuda.get_device_name(),
    }))

    selected = review_report["review_ids"]
    paths = {item: path for path, item in test}
    cases = [(paths[item], item) for item in selected]
    indices = {item: index for index, item in enumerate(ids)}
    baseline = None if probabilities is None else {item: score(probabilities[indices[item]]) for item in selected}
    raw, filtered, fractions = paired_inference(model, cases, args, for_condition("test"), views, baseline)
    publish_csv(out / "selected_test_v6_probe.csv",
                ("id", "impulse_fraction", "archived_v6", "raw_v6", "raw_confidence", "raw_margin",
                 "filtered_v6", "filtered_confidence", "filtered_margin", "changed", "raw_source"),
                ((item, fractions[item], classes[archived[item]], classes[raw[item][0]], *raw[item][1:],
                  classes[filtered[item][0]], *filtered[item][1:], raw[item][0] != filtered[item][0],
                  "subset_replay" if baseline is None else "verified_v6_cache")
                 for item in selected))
    print(f"{out / 'validation_report.json'}: {report['decision']}", flush=True)


def main():
    args = arguments()
    classes, checkpoint, v6, test, ids, archived, info = sources(args)
    probabilities, cache_state = cached_v6(
        args.v6_probabilities, ids, classes, info["v6_checkpoint_sha256"],
        info["test_image_signature"], args.v6_submission,
    )
    if args.stage == "review":
        review(args, classes, test, ids, archived, info, probabilities, cache_state)
    else:
        validate(args, classes, checkpoint, v6, test, ids, archived, info, probabilities)


if __name__ == "__main__":
    main()
