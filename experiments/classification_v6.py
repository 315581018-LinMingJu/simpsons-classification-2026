"""用相同步數對照 V5 增強與依 Test 分布校準的合成場景。"""

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from classification_v2 import (create_model, load_checkpoint, loader, save_checkpoint, train_epoch,
                               transforms_v2)
from classification_v5 import (BACKGROUND_PROBABILITY, EPOCHS, LR, METHOD, WEIGHT_DECAY,
                               BackgroundTrain)
from compare_v1_v2 import load_comparison_split
from diagnose_v5 import validation_inputs
from distribution_audit.compare import visual_signature
from distribution_audit.features import FEATURES, ImageReading, visual_features
from v2.data_checks import TEST_COUNT, read_json
from v3.inference import Views, infer_batches, infer_loader
from v3.reports import read_test_images, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v4.analysis import paired_effect, save_sheet
from v4.geometry import seeded, stratified_samples
from v5.background import replace_background
from v6.composite import Targets, composite

SCENE_PROBABILITY = 0.45
MAX_CLEAN_LOSS = 0.5
CONDITIONS = ("clean", "heldout_composite")
ARMS = ("control", "v6")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preview", "train", "predict"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v5-run", type=Path, default=Path("runs/v5_background"))
    parser.add_argument("--audit", type=Path, default=Path("runs/distribution_audit/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/v6_composite"))
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--per-class", type=int, default=16, help="驗證集每類上限")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.per_class, args.train_per_class) < 1 or args.workers < 0:
        parser.error("batch size and sample counts must be positive, workers nonnegative")
    output = args.output.resolve()
    if (output.parent != Path("runs").resolve() or not output.name.startswith("v6_")
            or any(output.is_relative_to(path.resolve())
                   for path in (args.data, args.v1_run, args.v2_run, args.v5_run))):
        parser.error("--output must be an independent runs/v6_... directory")
    return args


class CompositeTrain:
    def __init__(self, transform, targets):
        from torchvision import transforms as T

        self.transform = transform
        self.targets = targets
        # 合成視角不再做第二次隨機裁切，保留有標籤的主圖塊。
        self.scene_transform = T.Compose([T.Resize((224, 224)), *transform.transforms[4:]])

    def __call__(self, image):
        fraction = random.random()
        if fraction < BACKGROUND_PROBABILITY:
            rng = np.random.default_rng(random.getrandbits(64))
            if fraction < SCENE_PROBABILITY:
                return self.scene_transform(composite(image, self.targets, rng)[0])
            image = replace_background(image, rng)
        return self.transform(image)


def validation_view(path, item, condition, seed, targets):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if condition == "clean":
        return image
    rng = np.random.default_rng(seeded(seed, f"v6:heldout:{item}").getrandbits(64))
    return composite(image, targets, rng, held_out=True)[0]


class ValidationImages:
    def __init__(self, samples, condition, seed, targets, views):
        self.samples = samples
        self.condition = condition
        self.seed = seed
        self.targets = targets
        self.views = views

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item = self.samples[index]
        return self.views(validation_view(path, item, self.condition, self.seed, self.targets)), item


def preview(args, samples, targets, summary):
    output = args.output / "preview"
    output.mkdir(parents=True, exist_ok=False)
    bins = summary["reference_fit"]["bins_from_train_p20_p80"]
    sample = seeded(args.seed, "v6:preview").sample(samples, min(120, len(samples)))
    rows, results = [], {"clean": [], "training_composite": [], "heldout_composite": []}
    for path, item in sample:
        clean = validation_view(path, item, "clean", args.seed, targets)
        rng = np.random.default_rng(seeded(args.seed, f"v6:preview:{item}").getrandbits(64))
        aug = composite(clean, targets, rng)[0]
        hard = validation_view(path, item, "heldout_composite", args.seed, targets)
        images = dict(zip(results, (clean, aug, hard), strict=True))
        for condition, image in images.items():
            values = visual_features(image)
            reading = ImageReading(path, None, *image.size, None, values)
            results[condition].append((values, visual_signature(reading, bins)))
        if len(rows) < 6:
            rows.append([(f"{condition} | {item.split('/')[0]}", image)
                         for condition, image in images.items()])
    report = {
        "images": len(sample), "targets_from_unlabeled_test": asdict(targets),
        "comparisons": {
            condition: {
                "medians": dict(zip(FEATURES, np.median([row[0] for row in samples], axis=0).tolist())),
                "dominant_test_combo_count": sum(
                    signature[0:3] == ("little_black", "colored", "high")
                    and signature[3] in ("low", "mid") and signature[4] == "high"
                    for _, signature in samples
                ),
            } for condition, samples in results.items()
        },
    }
    save_sheet(output / "composite_examples.png", rows)
    publish(output / "preview.json", json_bytes(report))
    print(f"CPU preview: {output / 'preview.json'}", flush=True)


def paired_results(labels, predictions, seed):
    ids = list(labels)
    targets = np.asarray(list(labels.values()))
    scores = {}
    for condition in CONDITIONS:
        hits = {
            model: np.asarray([predictions[model, condition][item].label == labels[item]
                               for item in ids], dtype=int)
            for model in ("v5", *ARMS)
        }
        row = {model + "_accuracy_pct": 100 * float(values.mean()) for model, values in hits.items()}
        row["near_ceiling"] = min(1 - float(hits[base].mean()) for base in ("v5", "control")) < 0.05
        for baseline in ("v5", "control"):
            diff = hits["v6"] - hits[baseline]
            row[f"vs_{baseline}"] = {
                "macro_gain": paired_effect(diff, targets, seed),
                "wins": int((diff == 1).sum()), "losses": int((diff == -1).sum()),
                "both_wrong": int(((hits["v6"] == 0) & (hits[baseline] == 0)).sum()),
            }
        scores[condition] = row
    hard = scores["heldout_composite"]
    clean = scores["clean"]
    if clean["vs_v5"]["macro_gain"]["pp"] < -MAX_CLEAN_LOSS:
        decision = "clean_regression"
    elif hard["near_ceiling"]:
        decision = "heldout_near_ceiling"
    elif all(hard[f"vs_{ref}"]["macro_gain"]["ci95_pp"][0] > 0 for ref in ("v5", "control")):
        decision = "supports_composite"
    elif hard["vs_control"]["macro_gain"]["ci95_pp"][1] <= 0:
        decision = "no_evidence_beyond_extra_training"
    else:
        decision = "inconclusive"
    return {"conditions": scores, "decision": decision}


def validate(model, samples, args, targets, evaluation):
    import torch

    expected = {item for _, item in samples}
    views = Views(METHOD, evaluation)
    result = {}
    for condition in CONDITIONS:
        dataset = ValidationImages(samples, condition, args.seed, targets, views)
        batches = torch.utils.data.DataLoader(
            dataset, batch_size=min(args.batch_size, 16), num_workers=args.workers,
            pin_memory=True, prefetch_factor=1 if args.workers else None,
        )
        scores = {}
        for batch in infer_loader(model, batches):
            scores.update(batch)
        if scores.keys() != expected:
            raise ValueError("Validation predictions differ from the fixed set of IDs")
        result[condition] = scores
    return result


def train(args, classes, split_hash, training, validation, labels,
          parent, targets, train_tf, evaluation, provenance):
    import torch
    import torchvision

    output = args.output / "experiment"
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    settings = {
        **provenance, "classes": classes, "split_sha256": split_hash,
        "seed": args.seed, "train_per_class_limit": args.train_per_class,
        "validation_per_class_limit": args.per_class, "train_images": len(training),
        "validation_images": len(validation), "batch_size": args.batch_size, "workers": args.workers,
        "epochs_per_arm": 1, "steps_per_arm": (len(training) + args.batch_size - 1) // args.batch_size,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "precision": "float32",
        "control_background_probability": BACKGROUND_PROBABILITY,
        "v6_composite_probability": SCENE_PROBABILITY, "targets_from_unlabeled_test": asdict(targets),
        "eval_transform": repr(evaluation), "train_transform": repr(train_tf), "method": METHOD,
        "cudnn_deterministic": True, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
    }
    publish(output / "settings.json", json_bytes(settings))
    publish_csv(output / "sample_manifest.csv", ("split", "relative_path", "label"),
                ((part, item, classes[label]) for part, rows in (
                    ("train", [(item, label) for _, item, label in training]),
                    ("val", list(labels.items())),
                ) for item, label in rows))

    model = create_model().eval()
    load_checkpoint(model, parent, classes, split_hash)
    predictions = {}

    def evaluate(name):
        print(f"驗證 {name} 的 Clean / Heldout Composite……", flush=True)
        model.eval()
        for condition, scores in validate(model, validation, args, targets, evaluation).items():
            predictions[name, condition] = scores

    evaluate("v5")
    records = {}
    for name in ARMS:
        load_checkpoint(model, parent, classes, split_hash)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        transform = BackgroundTrain(train_tf) if name == "control" else CompositeTrain(train_tf, targets)
        batches = loader([(path, label) for path, _, label in training],
                         transform, settings, args.workers, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss = train_epoch(model, batches, optimizer, torch.nn.CrossEntropyLoss())
        del optimizer
        if not np.isfinite(loss):
            raise ValueError(f"Non-finite training loss: {name}")
        checkpoint = output / f"{name}.pt"
        save_checkpoint(checkpoint, {"model": model.state_dict(), "classes": classes,
                                     "split_sha256": split_hash, "epoch": 1})
        records[name] = {"loss": loss, "train_seconds": time.perf_counter() - started,
                         "checkpoint_sha256": file_sha256(checkpoint)}
        publish(output / f"{name}_epoch.json", json_bytes(records[name]))
        evaluate(name)

    report = paired_results(labels, predictions, args.seed)
    report.update({"arms": records, "images_per_condition": len(validation),
                   "elapsed_seconds": time.perf_counter() - start})
    publish_csv(output / "paired_predictions.csv",
                ("relative_path", "character", "condition", "v5", "control", "v6"),
                ((item, classes[label], condition,
                  *(classes[predictions[model_name, condition][item].label]
                    for model_name in ("v5", *ARMS)))
                 for condition in CONDITIONS for item, label in labels.items()))
    publish(output / "validation.json", json_bytes(report))
    print(f"V6 verdict: {report['decision']}", flush=True)


def predict(args, classes, split_hash, evaluation):
    output = args.output / "experiment"
    settings = read_json(output / "settings.json")
    report = read_json(output / "validation.json")
    checkpoint = output / "v6.pt"
    destination = output / "submission_v6.csv"
    if (settings["classes"] != classes or settings["split_sha256"] != split_hash
            or settings["eval_transform"] != repr(evaluation) or settings["method"] != METHOD
            or report["arms"]["v6"]["checkpoint_sha256"] != file_sha256(checkpoint)):
        raise ValueError("V6 prediction inputs differ from the completed experiment")
    if destination.exists():
        raise FileExistsError(destination)
    test = read_test_images(args.data / "test_images/test_images",
                            args.data / "sample_submission.csv")
    model = create_model().eval()
    load_checkpoint(model, checkpoint, classes, split_hash)
    predictions = {}
    for batch in infer_batches(model, test, METHOD, evaluation,
                               min(args.batch_size, 16), args.workers):
        predictions.update(batch)
    submission(destination, [item for _, item in test], predictions, classes)
    print(f"{destination} | validation: {report['decision']}", flush=True)


def main():
    args = arguments()
    train_root = args.data / "train/train/train"
    classes, rows, split_hash = load_comparison_split(train_root, args.v1_run, args.v2_run)

    if args.stage != "preview":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Use a confirmed available CUDA GPU for train or predict")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        train_tf, evaluation = transforms_v2()

    if args.stage == "predict":
        predict(args, classes, split_hash, evaluation)
        return

    summary = read_json(args.audit)
    if (summary["provenance"]["split_sha256"] != split_hash
            or summary["labels"]["allowed_characters"] != classes
            or summary["labels"]["test_ground_truth_available"] is not False
            or summary["test"]["decoded_images"] != TEST_COUNT):
        raise ValueError("Expected the full distribution audit from the same split and class mapping")
    targets = Targets.from_summary(summary)
    pool = [(row.path, row.path.relative_to(train_root).as_posix()) for row in rows if row.part == "train"]
    by_path = {row.path: row.label for row in rows if row.part == "train"}
    pool_labels = {item: by_path[path] for path, item in pool}
    sampled = stratified_samples(pool, pool_labels, args.train_per_class, args.seed, len(classes))
    if args.stage == "preview":
        preview(args, sampled, targets, summary)
        return

    old = read_json(args.v5_run / "settings.json")
    parent = args.v5_run / f"epoch_{EPOCHS:02d}.pt"
    parent_hash = file_sha256(parent)
    if (old["classes"] != classes or old["split_sha256"] != split_hash
            or old["train_image_signature"] != image_signature(row.path for row in rows)
            or old["train_transform"] != repr(train_tf) or old["eval_transform"] != repr(evaluation)
            or old["method"] != METHOD or old["background_probability"] != BACKGROUND_PROBABILITY
            or read_json(args.v5_run / "validation.json")["checkpoint_sha256"] != parent_hash):
        raise ValueError("V5 weights, data or transforms differ from the completed V5 run")
    validation, labels, _ = validation_inputs(args, rows, train_root, classes, old)
    train(args, classes, split_hash, [(path, item, pool_labels[item]) for path, item in sampled],
          validation, labels, parent, targets, train_tf, evaluation, {
              "v5_checkpoint_sha256": parent_hash, "distribution_audit_sha256": file_sha256(args.audit),
              "train_image_signature": old["train_image_signature"],
              "v5_validation_manifest_sha256": file_sha256(args.v5_run / "validation_samples.csv"),
          })


if __name__ == "__main__":
    main()
