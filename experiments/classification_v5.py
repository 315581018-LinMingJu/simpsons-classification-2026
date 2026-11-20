"""從 V2 權重微調外圍黑底適應，並與 V3 Multi-scale 對照。"""

import argparse
import hashlib
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from classification_v2 import (create_model, load_checkpoint, loader, save_checkpoint, train_epoch,
                               transforms_v2)
from classification_v4 import paired_data
from compare_v1_v2 import load_comparison_split
from v2.data_checks import read_json
from v3.inference import Views, infer_batches, infer_loader
from v3.reports import read_test_images, submission
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v4.analysis import paired_effect
from v5.background import DARK_MAX, MAX_COVERAGE, MIN_COVERAGE, exterior_mask, replace_background

EPOCHS = 2
LR = 3e-5
WEIGHT_DECAY = 1e-4
BACKGROUND_PROBABILITY = 0.6
METHOD = "multiscale"
MAX_CLEAN_LOSS_PP = 0.5


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "predict"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--v3-run", type=Path, default=Path("runs/v3_tta_155"))
    parser.add_argument("--output", type=Path, default=Path("runs/v5_background"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.per_class) < 1 or args.workers < 0:
        parser.error("batch size and per-class sample count must be positive, workers nonnegative")
    output = args.output.resolve()
    if (output.parent != Path("runs").resolve() or not output.name.startswith("v5_")
            or any(output.is_relative_to(path.resolve())
                   for path in (args.data, args.v1_run, args.v2_run, args.v3_run))):
        parser.error("--output must be an independent runs/v5_... directory")
    return args


class BackgroundTrain:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if random.random() < BACKGROUND_PROBABILITY:
            image = replace_background(image, np.random.default_rng(random.getrandbits(64)))
        return self.transform(image)


class BackgroundValidation:
    def __init__(self, samples, transform, seed, stress):
        self.samples = samples
        self.transform = transform
        self.seed = seed
        self.stress = stress

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.stress:
            key = hashlib.sha256(f"{self.seed}:heldout:{item}".encode()).digest()
            image = replace_background(image, np.random.default_rng(int.from_bytes(key, "big")), held_out=True)
        return self.transform(image), item


def paired_outcome(targets, baseline, candidate, seed):
    conditions = {}
    labels = [targets[item] for item in targets]
    for condition in ("clean", "heldout_background"):
        a = np.array([baseline[condition][item].label == y for item, y in targets.items()], dtype=int)
        b = np.array([candidate[condition][item].label == y for item, y in targets.items()], dtype=int)
        delta = b - a
        conditions[condition] = {
            "gain_macro_pp": paired_effect(delta, labels, seed),
            "v2_accuracy_pct": 100 * float(a.mean()), "v5_accuracy_pct": 100 * float(b.mean()),
            "wins": int((delta == 1).sum()), "losses": int((delta == -1).sum()), "images": len(labels),
        }
    clean = conditions["clean"]["gain_macro_pp"]
    stress = conditions["heldout_background"]["gain_macro_pp"]
    if clean["pp"] < -MAX_CLEAN_LOSS_PP:
        decision = "clean_regression"
    elif stress["ci95_pp"][0] > 0:
        decision = "supports_background_adaptation"
    elif stress["ci95_pp"][1] <= 0:
        decision = "does_not_support_background_adaptation"
    else:
        decision = "inconclusive"
    return {"decision": decision, "conditions": conditions}


def validate(model, samples, targets, args, evaluation):
    import torch

    scores = {}
    for condition in ("clean", "heldout_background"):
        dataset = BackgroundValidation(samples, Views(METHOD, evaluation), args.seed,
                                       stress=condition == "heldout_background")
        batches = torch.utils.data.DataLoader(
            dataset, batch_size=min(args.batch_size, 16), num_workers=args.workers,
            pin_memory=True, prefetch_factor=1 if args.workers else None,
        )
        result = {}
        for batch in infer_loader(model, batches):
            result.update(batch)
        if result.keys() != targets.keys():
            raise ValueError("Validation prediction IDs differ from the fixed subset")
        scores[condition] = result
    return scores


def train(args, train_root, classes, rows, split_hash, checkpoint, train_tf, evaluation):
    import torch
    import torchvision

    args.output.mkdir(parents=True, exist_ok=False)
    samples, targets, duplicates, signature, subset_source = paired_data(args, rows, train_root, len(classes))
    coverage = []
    for path, _ in samples:
        with Image.open(path) as image:
            coverage.append(float(exterior_mask(np.asarray(image.convert("RGB"))).mean()))
    if not any(coverage):
        raise ValueError("No fixed validation image has an exterior matte to change")
    metadata = {
        "seed": args.seed, "batch_size": args.batch_size, "workers": args.workers,
        "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW",
        "precision": "float32", "train_images": sum(row.part == "train" for row in rows),
        "cudnn_deterministic": True, "background_probability": BACKGROUND_PROBABILITY,
        "dark_max": DARK_MAX, "matte_coverage_bounds": [MIN_COVERAGE, MAX_COVERAGE],
        "validation_per_class": args.per_class, "validation_samples": len(samples),
        "validation_counts_by_class": {classes[i]: n for i, n in Counter(targets.values()).items()},
        "duplicate_counts": duplicates, "validation_subset_source": subset_source,
        "validation_with_matte": sum(value > 0 for value in coverage),
        "validation_median_replaced_fraction": float(np.median(coverage)),
        "split_sha256": split_hash, "train_image_signature": signature, "classes": classes,
        "v2_checkpoint_sha256": file_sha256(checkpoint),
        "train_transform": repr(train_tf), "eval_transform": repr(evaluation), "method": METHOD,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
    }
    publish(args.output / "settings.json", json_bytes(metadata))
    publish_csv(args.output / "validation_samples.csv", ("relative_path", "label", "matte_fraction"),
                ((item, classes[targets[item]], fraction)
                 for (_, item), fraction in zip(samples, coverage, strict=True)))
    model = create_model()
    load_checkpoint(model, checkpoint, classes, split_hash)
    model.eval()
    print("驗證 V2 對照權重……", flush=True)
    baseline = validate(model, samples, targets, args, evaluation)

    train_batches = loader([(row.path, row.label) for row in rows if row.part == "train"],
                           BackgroundTrain(train_tf), metadata, args.workers, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        loss = train_epoch(model, train_batches, optimizer, criterion)
        save_checkpoint(args.output / f"epoch_{epoch:02d}.pt", {
            "model": model.state_dict(), "classes": classes, "split_sha256": split_hash, "epoch": epoch,
        })
        publish(args.output / f"epoch_{epoch:02d}.json", json_bytes({
            "epoch": epoch, "loss": loss, "seconds": time.perf_counter() - started,
        }))
        print(f"epoch {epoch}/{EPOCHS}: loss={loss:.5f}", flush=True)
    model.eval()
    print("驗證固定最後一輪 V5 權重……", flush=True)
    candidate = validate(model, samples, targets, args, evaluation)
    outcome = paired_outcome(targets, baseline, candidate, args.seed)
    outcome["checkpoint_sha256"] = file_sha256(args.output / f"epoch_{EPOCHS:02d}.pt")
    publish_csv(args.output / "paired_predictions.csv",
                ("condition", "relative_path", "true_character", "v2_prediction", "v5_prediction"),
                ((condition, item, classes[label], classes[baseline[condition][item].label],
                  classes[candidate[condition][item].label])
                 for condition in baseline for item, label in targets.items()))
    publish(args.output / "validation.json", json_bytes(outcome))
    print(f"V5 validation: {outcome['decision']}", flush=True)


def predict(args, classes, split_hash, evaluation):
    settings = read_json(args.output / "settings.json")
    report = read_json(args.output / "validation.json")
    model_path = args.output / f"epoch_{EPOCHS:02d}.pt"
    if (settings["split_sha256"] != split_hash or settings["classes"] != classes
            or settings["eval_transform"] != repr(evaluation)
            or settings["method"] != METHOD or report["checkpoint_sha256"] != file_sha256(model_path)):
        raise ValueError("V5 checkpoint or settings are inconsistent with the completed run")
    output = args.output / "submission_v5.csv"
    if output.exists():
        raise FileExistsError(output)
    samples = read_test_images(args.data / "test_images/test_images",
                               args.data / "sample_submission.csv")
    model = create_model().eval()
    load_checkpoint(model, model_path, classes, split_hash)
    predictions = {}
    for batch in infer_batches(model, samples, METHOD, evaluation, min(args.batch_size, 16), args.workers):
        predictions.update(batch)
        if len(predictions) % 1024 == 0:
            print(f"已推論 {len(predictions)}/{len(samples)} 張", flush=True)
    submission(output, [item for _, item in samples], predictions, classes)
    print(f"{output} ({len(predictions)} 張，驗證判定 {report['decision']})", flush=True)


def main():
    args = arguments()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V5 train and predict require a confirmed available CUDA GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    train_root = args.data / "train/train/train"
    classes, rows, split_hash = load_comparison_split(train_root, args.v1_run, args.v2_run)
    train_tf, evaluation = transforms_v2()
    if read_json(args.v2_run / "run_config.json")["eval_transform"] != repr(evaluation):
        raise ValueError("V2 evaluation transform differs from the current runtime")
    if args.stage == "train":
        train(args, train_root, classes, rows, split_hash, args.v2_run / "best.pt", train_tf, evaluation)
    else:
        predict(args, classes, split_hash, evaluation)


if __name__ == "__main__":
    main()
