"""稽核原 V5 配對結果，使用現有權重進行更困難的驗證。"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from classification_v2 import create_model, load_checkpoint, transforms_v2
from classification_v5 import EPOCHS, METHOD
from compare_v1_v2 import load_comparison_split
from v2.data_checks import read_json
from v3.inference import Views, infer_loader
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v4.analysis import save_sheet
from v4.geometry import seeded, stratified_samples
from v5.challenge import PARENTS, ChallengeImages
from v5.diagnostics import evaluate_challenge, legacy_audit


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "challenge"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v5-run", type=Path, default=Path("runs/v5_background"))
    parser.add_argument("--output", type=Path, default=Path("runs/v5_challenge"))
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not (args.batch_size > 0 and args.per_class > 0 and args.workers >= 0):
        parser.error("batch size and per-class sample count must be positive, workers nonnegative")
    if args.stage == "challenge":
        output = args.output.resolve()
        if (output.parent != Path("runs").resolve() or not output.name.startswith("v5_challenge")
                or any(output.is_relative_to(path.resolve())
                       for path in (args.data, args.v1_run, args.v2_run, args.v5_run))):
            parser.error("--output must be an independent runs/v5_challenge... directory")
        if output.exists():
            parser.error(f"Output already exists: {output}")
    return args


def validation_inputs(args, rows, train_root, classes, settings):
    by_id = {row.path.relative_to(train_root).as_posix(): row for row in rows}
    manifest = args.v5_run / "validation_samples.csv"
    subset = read_json(args.v5_run / "validation_subset.json")
    if subset["image_signature"] != settings["train_image_signature"]:
        raise ValueError("V5 validation subset provenance does not match settings")
    excluded = set(subset["excluded_ids"])
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        listed = list(csv.DictReader(file))
    targets, samples = {}, []
    for item in listed:
        key = item["relative_path"]
        row = by_id.get(key)
        if (key in targets or key in excluded or row is None or row.part != "val"
                or item["label"] != classes[row.label]):
            raise ValueError(f"Invalid V5 validation item: {key}")
        targets[key] = row.label
        samples.append((row.path, key))
    if len(samples) != settings["validation_samples"]:
        raise ValueError("V5 validation sample manifest count differs from settings")
    samples = stratified_samples(samples, targets, args.per_class, args.seed, len(classes))
    targets = {item: targets[item] for _, item in samples}
    train = defaultdict(list)
    for row in rows:
        if row.part == "train":
            train[row.label].append(row)
    distractors = {}
    for item, label in targets.items():
        rng = seeded(args.seed, "distractor:" + item)
        choice = rng.choice(sorted(key for key in train if key != label))
        distractors[item] = rng.choice(train[choice])
    return samples, targets, distractors


def write_preview(output, datasets, classes, targets, seed):
    count = len(datasets["clean"])
    indices = seeded(seed, "preview").sample(list(range(count)), min(6, count))
    rows, facts = [], {}
    for index in indices:
        item = datasets["clean"].samples[index][1]
        rows.append([])
        facts[item] = {}
        for condition, dataset in datasets.items():
            image, stats = dataset.case(index)
            rows[-1].append((f"{condition}\n{classes[targets[item]]}", image))
            facts[item][condition] = stats
    save_sheet(output / "challenge_examples.png", rows)
    publish(output / "preview_geometry.json", json_bytes(facts))


def run_challenge(args):
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("challenge requires a confirmed available CUDA GPU")
    train_root = args.data / "train/train/train"
    classes, rows, split_hash = load_comparison_split(train_root, args.v1_run, args.v2_run)
    old = read_json(args.v5_run / "settings.json")
    _, evaluation = transforms_v2()
    weight_paths = {"v2": args.v2_run / "best.pt", "v5": args.v5_run / f"epoch_{EPOCHS:02d}.pt"}
    weight_hashes = {name: file_sha256(path) for name, path in weight_paths.items()}
    if (old["split_sha256"] != split_hash or old["classes"] != classes
            or old["eval_transform"] != repr(evaluation) or old["method"] != METHOD
            or old["train_image_signature"] != image_signature(row.path for row in rows)
            or old["v2_checkpoint_sha256"] != weight_hashes["v2"]
            or read_json(args.v5_run / "validation.json")["checkpoint_sha256"] != weight_hashes["v5"]):
        raise ValueError("V5 data, preprocessing, or checkpoints differ from the original V5 run")
    samples, targets, distractors = validation_inputs(args, rows, train_root, classes, old)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    views = Views(METHOD, evaluation)
    datasets = {condition: ChallengeImages(samples, condition, views, args.seed, distractors)
                for condition in PARENTS}
    write_preview(args.output, datasets, classes, targets, args.seed)
    alpha_fractions = {item: datasets["geometry"].case(index)[1]["primary_alpha_fraction"]
                       for index, (_, item) in enumerate(samples)}
    publish_csv(args.output / "sample_manifest.csv",
                ("validation_image", "source_label", "training_distractor", "distractor_label",
                 "primary_alpha_fraction"),
                ((item, classes[label], distractors[item].path.relative_to(train_root).as_posix(),
                  classes[distractors[item].label], alpha_fractions[item])
                 for item, label in targets.items()))

    predictions = {}
    started = time.perf_counter()
    for name, path in weight_paths.items():
        model = create_model().eval()
        load_checkpoint(model, path, classes, split_hash)
        for condition, dataset in datasets.items():
            batches = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                prefetch_factor=1 if args.workers else None,
            )
            scores = {}
            for batch in infer_loader(model, batches):
                scores.update(batch)
            if scores.keys() != targets.keys():
                raise ValueError("Challenge prediction IDs differ from the fixed validation subset")
            predictions[condition, name] = scores
            print(f"{name}/{condition}: {len(scores)} images", flush=True)

    results = evaluate_challenge(targets, predictions, args.seed, alpha_fractions)
    publish_csv(args.output / "paired_predictions.csv",
                ("condition", "relative_path", "true_character", "v2_prediction", "v5_prediction"),
                ((condition, item, classes[label], classes[predictions[condition, "v2"][item].label],
                  classes[predictions[condition, "v5"][item].label])
                 for condition in PARENTS for item, label in targets.items()))
    report = {
        "protocol": "v5_challenge_v1", "seed": args.seed, "per_class_limit": args.per_class,
        "samples": len(samples), "classes": classes, "split_sha256": split_hash,
        "checkpoint_sha256": weight_hashes, "method": METHOD, "eval_transform": repr(evaluation),
        "validation_sample_manifest_sha256": file_sha256(args.v5_run / "validation_samples.csv"),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "batch_size": args.batch_size, "workers": args.workers,
        "inference_wall_seconds": time.perf_counter() - started, "results": results,
    }
    publish(args.output / "challenge.json", json_bytes(report))
    for condition, row in results.items():
        print(f"{condition}: V2 {row['v2_accuracy_pct']:.2f}% / V5 {row['v5_accuracy_pct']:.2f}%",
              flush=True)


def main():
    args = arguments()
    if args.stage == "audit":
        print(json.dumps(legacy_audit(args.v5_run), indent=2, ensure_ascii=False))
    else:
        run_challenge(args)


if __name__ == "__main__":
    main()
