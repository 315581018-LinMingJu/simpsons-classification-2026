"""Simpsons V2 experiment and exact-duplicate leakage diagnostic."""

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from tempfile import NamedTemporaryFile
from itertools import islice
from pathlib import Path

from v2.data_checks import CLASS_COUNT, TEST_COUNT, duplicate_diagnostic, load_inputs, read_json


class Images:
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), target


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "smoke", "train", "diagnostic", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v1", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--reference-run", type=Path, default=Path("runs/v2_primary"),
                        help="Completed primary run for the duplicate-removal diagnostic")
    parser.add_argument("--v1-best", type=Path, help="Optional V1 best.pt for paired diagnostic scoring")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.v1_best and args.mode != "diagnostic":
        parser.error("--v1-best is for diagnostic mode")
    defaults = {"audit": "v2_audit", "diagnostic": "v2_dedup"}
    args.run = args.run or Path("runs") / defaults.get(args.mode, "v2_primary")
    args.template = args.template or args.data / "sample_submission.csv"
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_output(run, v1):
    if run.resolve().is_relative_to(v1.resolve()):
        raise ValueError("V2 output cannot be placed inside the V1 run")
    run.mkdir(parents=True, exist_ok=False)


def transforms_v2():
    from torchvision import transforms as T

    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.55, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomAffine(
            degrees=15, translate=(0.12, 0.12), scale=(0.82, 1.0),
            interpolation=T.InterpolationMode.BILINEAR, fill=(124, 116, 104),
        )], p=0.5),
        T.RandomGrayscale(p=0.12),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])
    evaluation = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    return train, evaluation


def create_model(pretrained=False):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, CLASS_COUNT)
    return model.cuda()


def loader(samples, transform, config, workers, shuffle=False):
    import torch

    generator = torch.Generator().manual_seed(config["seed"])
    return torch.utils.data.DataLoader(
        Images(samples, transform), batch_size=config["batch_size"], shuffle=shuffle,
        num_workers=workers, pin_memory=True, generator=generator,
    )


def score(model, batches, limit=None):
    import torch

    model.eval()
    hits, count = 0, 0
    with torch.inference_mode():
        for images, labels in islice(batches, limit):
            predictions = model(images.cuda(non_blocking=True)).argmax(1)
            hits += (predictions == labels.cuda(non_blocking=True)).sum().item()
            count += len(labels)
    if not count:
        raise ValueError("Validation contains no images")
    return {"correct": hits, "images": count, "accuracy": hits / count}


def train_epoch(model, batches, optimizer, criterion, limit=None):
    model.train()
    loss_sum, count = 0.0, 0
    for images, labels in islice(batches, limit):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        count += len(labels)
    return loss_sum / count


def save_checkpoint(path, checkpoint):
    import torch

    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(model, path, classes, split_hash=None):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != classes:
        raise ValueError(f"Class order in checkpoint does not match V1: {path}")
    if split_hash is not None and checkpoint.get("split_sha256") != split_hash:
        raise ValueError(f"Checkpoint was trained on a different split: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def predict(args, model, test, evaluation, config, classes, split_hash):
    saved = read_json(args.run / "run_config.json")
    summary = read_json(args.run / "summary.json")
    if saved["mode"] != "train" or summary["completed_epochs"] != config["epochs"]:
        raise ValueError("Prediction requires a completed primary V2 run")
    if saved["eval_transform"] != repr(evaluation):
        raise ValueError("Evaluation transform differs from the trained run")
    checkpoint = load_checkpoint(model, args.run / "best.pt", classes, split_hash)
    if checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError("Checkpoint epoch does not match the completed run")
    output = args.run / "submission_v2.csv"
    if output.exists():
        raise FileExistsError(output)

    import torch

    model.eval()
    rows = []
    with torch.inference_mode():
        for images, ids in loader(test, evaluation, config, args.workers):
            indices = model(images.cuda(non_blocking=True)).argmax(1).cpu().tolist()
            rows.extend((item, classes[index]) for item, index in zip(ids, indices, strict=True))
    if len(rows) != TEST_COUNT or [item for item, _ in rows] != [item for _, item in test]:
        raise ValueError("Prediction IDs differ from the validated submission template")
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=output.parent, delete=False,
                                encoding="utf-8", newline="") as file:
            temporary = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(["id", "character"])
            writer.writerows(rows)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()
    print(f"Submission: {output} ({len(rows)} IDs)")


def main():
    args = arguments()
    classes, config, v1_history, rows, test, split_bytes = load_inputs(
        args.data, args.v1, args.template
    )
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    v1_best = max(v1_history, key=lambda row: row["val_accuracy"])
    baseline = {
        "split_sha256": split_hash,
        "v1_best_epoch": v1_best["epoch"],
        "v1_best_val_accuracy": v1_best["val_accuracy"],
        "v1_epoch_seconds_total": sum(row["seconds"] for row in v1_history),
        "sample_counts": {**Counter(row.part for row in rows), "test": len(test)},
    }
    print(json.dumps(baseline), flush=True)
    removed_train, overlapping_val = set(), set()
    if args.mode in {"audit", "diagnostic"}:
        duplicates, removed_train, overlapping_val = duplicate_diagnostic(rows)
        print(json.dumps({key: value for key, value in duplicates.items() if key != "groups"}),
              flush=True)
        if args.mode == "audit":
            prepare_output(args.run, args.v1)
            write_json(args.run / "audit.json", {**baseline, **duplicates})
            return
        if not removed_train:
            raise ValueError("No cross-partition duplicates found; no dedup retraining needed")

    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this mode; use audit for CPU-only checks")
    arch = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"Installed PyTorch lacks native {arch} support")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = True
    train_tf, eval_tf = transforms_v2()
    if args.mode == "predict":
        predict(args, create_model(), test, eval_tf, config, classes, split_hash)
        return

    train = [(r.path, r.label) for r in rows if r.part == "train" and str(r.path) not in removed_train]
    val = [(r.path, r.label) for r in rows if r.part == "val"]
    if len({label for _, label in train}) != CLASS_COUNT:
        raise ValueError("Duplicate removal eliminated a training class")
    val_dl = loader(val, eval_tf, config, args.workers)
    diagnostic_scores = {}
    if args.mode == "diagnostic":
        primary = read_json(args.reference_run / "run_config.json")
        primary_summary = read_json(args.reference_run / "summary.json")
        if primary["mode"] != "train" or primary_summary["completed_epochs"] != config["epochs"]:
            raise ValueError("Diagnostic requires a completed primary run")
        for key, value in {
            "seed": config["seed"], "batch_size": config["batch_size"],
            "train_transform": repr(train_tf), "eval_transform": repr(eval_tf),
        }.items():
            if primary[key] != value:
                raise ValueError(f"Primary run has different {key}")
        subset_loaders = {
            "duplicate_exposed": loader(
                [(path, label) for path, label in val if str(path) in overlapping_val],
                eval_tf, config, args.workers,
            )
        }
        clean = [(path, label) for path, label in val if str(path) not in overlapping_val]
        if clean:
            subset_loaders["no_exact_train_match"] = loader(clean, eval_tf, config, args.workers)
        comparison = create_model()
        checkpoint = load_checkpoint(comparison, args.reference_run / "best.pt", classes, split_hash)
        if checkpoint["epoch"] != primary_summary["best_epoch"]:
            raise ValueError("Primary checkpoint epoch does not match the completed run")
        del checkpoint
        diagnostic_scores["v2_primary_best"] = {
            name: score(comparison, batches) for name, batches in subset_loaders.items()
        }
        if args.v1_best:
            load_checkpoint(comparison, args.v1_best, classes)
            diagnostic_scores["v1_best"] = {
                name: score(comparison, batches) for name, batches in subset_loaders.items()
            }
        del comparison
        # Match primary initialization after constructing the scoring model.
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])

    model = create_model(pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    train_dl = loader(train, train_tf, config, args.workers, shuffle=True)
    if args.mode == "smoke":
        print({"train_loss": train_epoch(model, train_dl, optimizer, criterion, limit=2),
               "validation": score(model, val_dl, limit=1)}, flush=True)
        return

    prepare_output(args.run, args.v1)
    (args.run / "split.csv").write_bytes(split_bytes)
    write_json(args.run / "classes.json", classes)
    write_json(args.run / "run_config.json", {
        **config, **baseline, "mode": args.mode, "train_images_used": len(train),
        "val_images_used": len(val), "train_transform": repr(train_tf),
        "eval_transform": repr(eval_tf), "precision": "float32",
        "weights": "IMAGENET1K_V1", "workers": args.workers, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    })
    if args.mode == "diagnostic":
        write_json(args.run / "duplicates.json", duplicates)

    history = []
    best = -1
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        tick = time.perf_counter()
        loss = train_epoch(model, train_dl, optimizer, criterion)
        result = score(model, val_dl)
        row = {"epoch": epoch, "train_loss": loss, "val_accuracy": result["accuracy"],
               "seconds": time.perf_counter() - tick, "val_images": result["images"]}
        if row["val_accuracy"] > best:
            best = row["val_accuracy"]
            best_epoch = epoch
            save_checkpoint(args.run / "best.pt", {
                "model": model.state_dict(), "classes": classes, "epoch": epoch,
                "best_acc": best, "split_sha256": split_hash,
            })
        history.append(row)
        write_json(args.run / "history.json", history)
        print(json.dumps(row), flush=True)
    summary = {
        "completed_epochs": len(history), "best_epoch": best_epoch, "best_val_accuracy": best,
        "epoch_seconds_total": sum(row["seconds"] for row in history),
        "training_wall_seconds": time.perf_counter() - started,
    }
    if args.mode == "diagnostic":
        load_checkpoint(model, args.run / "best.pt", classes, split_hash)
        diagnostic_scores["v2_dedup_best"] = {
            name: score(model, batches) for name, batches in subset_loaders.items()
        }
        summary.update({"subset_scores": diagnostic_scores,
                        "primary_best_val_accuracy": primary_summary["best_val_accuracy"]})
    else:
        summary["v1_best_val_accuracy"] = v1_best["val_accuracy"]
        summary["accuracy_delta_percentage_points"] = (best - v1_best["val_accuracy"]) * 100
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
