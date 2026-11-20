"""在相同驗證圖片上比較 V1 與 V2 的抗干擾表現。"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageOps, __version__ as pillow_version

from classification_v2 import create_model, load_checkpoint, transforms_v2
from v2.data_checks import duplicate_diagnostic, load_split, read_json

SPLIT_SHA256 = "d6b86a9b8cd01f8a1a631902bcb1e2b9c394c345cc80443ac940d53982ca5054"
FILL = (124, 116, 104)
PERTURBATIONS = {
    "rotate_translate": {"absolute_degrees": (6, 12), "shift_fraction": (-0.06, 0.06)},
    "grayscale": {"gray_blend": (0.65, 0.85)},
    "zoom_out": {"scale": (0.72, 0.86)},
    "occlusion": {"side_fraction": (0.27, 0.35), "position_fraction": (0.15, 0.85)},
}
DISPLAY_NAMES = {
    "clean_full": "Clean (full)",
    "clean": "Clean (filtered)",
    "rotate_translate": "Rotate + translate",
    "grayscale": "Grayscale blend",
    "zoom_out": "Zoom out",
    "occlusion": "Occlusion",
}


def load_comparison_split(train_root, v1, v2, expected_hash=SPLIT_SHA256):
    split = (v1 / "split.csv").read_bytes()
    split_hash = hashlib.sha256(split).hexdigest()
    if split_hash != expected_hash:
        raise ValueError(f"V1 split SHA-256 mismatch: {split_hash}")
    if (v2 / "split.csv").read_bytes() != split:
        raise ValueError("V2 split differs from the original V1 split")
    classes = read_json(v1 / "classes.json")
    if read_json(v2 / "classes.json") != classes:
        raise ValueError("V1 and V2 run class mappings differ")
    return classes, load_split(train_root, classes, split), split_hash


def perturb(image, condition, seed, relative_path):
    if condition in ("clean", "clean_full"):
        return image
    # 以相對路徑決定參數，各 DataLoader worker 取得相同結果。
    key = json.dumps([seed, str(relative_path), condition], ensure_ascii=False)
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big"))
    settings = PERTURBATIONS[condition]
    width, height = image.size
    if condition == "rotate_translate":
        angle = rng.uniform(*settings["absolute_degrees"]) * rng.choice((-1, 1))
        shift = tuple(round(rng.uniform(*settings["shift_fraction"]) * axis)
                      for axis in image.size)
        return image.rotate(angle, translate=shift, resample=Image.Resampling.BILINEAR,
                            fillcolor=FILL)
    if condition == "grayscale":
        return Image.blend(image, ImageOps.grayscale(image).convert("RGB"),
                           rng.uniform(*settings["gray_blend"]))
    if condition == "zoom_out":
        scale = rng.uniform(*settings["scale"])
        small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                             Image.Resampling.BILINEAR)
        result = Image.new("RGB", image.size, FILL)
        result.paste(small, ((width - small.width) // 2, (height - small.height) // 2))
        return result

    fraction = rng.uniform(*settings["side_fraction"])
    patch_w, patch_h = (max(1, round(axis * fraction)) for axis in image.size)
    x = round(rng.uniform(*settings["position_fraction"]) * (width - patch_w))
    y = round(rng.uniform(*settings["position_fraction"]) * (height - patch_h))
    result = image.copy()
    result.paste(FILL, (x, y, x + patch_w, y + patch_h))
    return result


class ValidationImages:
    def __init__(self, samples, train_root, condition, seed, preprocess):
        self.samples = samples
        self.train_root = train_root
        self.condition = condition
        self.seed = seed
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        with Image.open(row.path) as image:
            transformed = perturb(image.convert("RGB"), self.condition, self.seed,
                                  row.path.relative_to(self.train_root).as_posix())
        return self.preprocess(transformed), row.label


def score_condition(models, batches, forward, condition, scope, clean=None):
    samples = v1_hits = v2_hits = disagreements = 0
    v1_seconds = v2_seconds = 0.0
    for images, labels in batches:
        v1, seconds = forward(models[0], images)
        v1_seconds += seconds
        v2, seconds = forward(models[1], images)
        v2_seconds += seconds
        for target, a, b in zip(labels, v1, v2, strict=True):
            samples += 1
            v1_hits += a == target
            v2_hits += b == target
            disagreements += a != b
    if not samples:
        raise ValueError(f"No validation samples for {condition}")
    v1_pct, v2_pct = 100 * v1_hits / samples, 100 * v2_hits / samples
    return {
        "condition": condition, "scope": scope, "samples": samples,
        "v1_correct": v1_hits, "v2_correct": v2_hits,
        "v1_accuracy_pct": v1_pct, "v2_accuracy_pct": v2_pct,
        "v2_minus_v1_pp": 100 * (v2_hits - v1_hits) / samples,
        "prediction_disagreements": disagreements,
        "prediction_disagreement_pct": 100 * disagreements / samples,
        "v1_drop_from_clean_pp": None if clean is None else clean["v1_accuracy_pct"] - v1_pct,
        "v2_drop_from_clean_pp": None if clean is None else clean["v2_accuracy_pct"] - v2_pct,
        "v1_inference_seconds": v1_seconds, "v2_inference_seconds": v2_seconds,
    }


def write_reports(output, settings, results, examples, train_root, seed, preview):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output / "comparison.json").write_text(
        json.dumps({"settings": settings, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, (acc_ax, drop_ax) = plt.subplots(1, 2, figsize=(12, 4.6))
    names = [DISPLAY_NAMES[row["condition"]] for row in results]
    disturbed = [row for row in results if row["condition"] in PERTURBATIONS]
    for model, label, offset, color in (("v1", "V1", -0.19, "#3265a8"),
                                        ("v2", "V2", 0.19, "#c15c30")):
        acc_ax.barh([i + offset for i in range(len(results))],
                    [row[f"{model}_accuracy_pct"] for row in results], height=0.36,
                    label=label, color=color)
        drop_ax.barh([i + offset for i in range(len(disturbed))],
                     [row[f"{model}_drop_from_clean_pp"] for row in disturbed], height=0.36,
                     label=label, color=color)
    acc_ax.set_yticks(range(len(names)), names)
    acc_ax.set(xlabel="Accuracy (%)", xlim=(0, 100), title="Validation accuracy")
    drop_ax.set_yticks(range(len(disturbed)), [DISPLAY_NAMES[r["condition"]] for r in disturbed])
    drop_ax.axvline(0, color="0.4", linewidth=0.8)
    drop_ax.set(xlabel="Clean minus perturbed (percentage points)", title="Accuracy drop")
    for axis in (acc_ax, drop_ax):
        axis.invert_yaxis()
        axis.legend()
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    conditions = ("clean", *PERTURBATIONS)
    fig, axes = plt.subplots(len(examples), len(conditions), squeeze=False,
                             figsize=(12, 2.5 * len(examples)))
    for column, condition in enumerate(conditions):
        images = ValidationImages(examples, train_root, condition, seed, preview)
        for row, (image, _) in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image)
            axis.axis("off")
            if row == 0:
                axis.set_title(DISPLAY_NAMES[condition])
    fig.tight_layout()
    fig.savefig(output / "examples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("data/extracted/train/train/train"))
    parser.add_argument("--v1-run", type=Path, default=Path("runs/v1_baseline"))
    parser.add_argument("--v2-run", type=Path, default=Path("runs/v2_primary"))
    parser.add_argument("--output", type=Path, default=Path("runs/v2_robustness"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size < 1:
        parser.error("--workers must be nonnegative and --batch-size must be positive")
    if args.output.exists() or any(args.output.resolve().is_relative_to(path.resolve()) for path in
                                   (args.v1_run, args.v2_run, args.train_root)):
        parser.error("--output must be a new directory outside both model runs and the training tree")

    import matplotlib
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model comparison")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classes, rows, split_hash = load_comparison_split(args.train_root, args.v1_run, args.v2_run)
    _, evaluation = transforms_v2()
    checkpoints = [args.v1_run / "best.pt", args.v2_run / "best.pt"]
    models = [create_model().eval(), create_model().eval()]
    metadata = []
    for model, path, expected_hash in zip(models, checkpoints, (None, split_hash), strict=True):
        checkpoint = load_checkpoint(model, path, classes, expected_hash)
        metadata.append({"path": str(path), "epoch": checkpoint["epoch"],
                         "reported_best_accuracy_fraction": checkpoint["best_acc"],
                         "embedded_split_sha256": checkpoint.get("split_sha256")})
        del checkpoint

    print("正在辨識與訓練集完全相同的驗證圖……", flush=True)
    duplicate_report, _, exposed = duplicate_diagnostic(rows)
    full = [row for row in rows if row.part == "val"]
    filtered = [row for row in full if str(row.path) not in exposed]
    if not filtered:
        raise ValueError("No validation images remain after excluding exact duplicates")

    def forward(model, images):
        torch.cuda.synchronize(images.device)
        start = time.perf_counter()
        logits = model(images)
        torch.cuda.synchronize(images.device)
        elapsed = time.perf_counter() - start
        return logits.argmax(1).cpu().tolist(), elapsed

    results = []
    filtered_clean = None
    with torch.inference_mode():
        warmup = torch.zeros((args.batch_size, 3, 224, 224), device="cuda")
        for model in models:
            model(warmup)
        torch.cuda.synchronize()
        del warmup
        for condition in DISPLAY_NAMES:
            scope = "full" if condition == "clean_full" else "no_exact_train_match"
            partition = full if scope == "full" else filtered
            dataset = ValidationImages(partition, args.train_root, condition, args.seed, evaluation)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
            batches = ((images.cuda(non_blocking=True), labels.tolist()) for images, labels in loader)
            clean = filtered_clean if condition in PERTURBATIONS else None
            result = score_condition(models, batches, forward, condition, scope, clean)
            results.append(result)
            if condition == "clean":
                filtered_clean = result
            print(f"{condition}: n={result['samples']} V1={result['v1_accuracy_pct']:.4f}% "
                  f"V2={result['v2_accuracy_pct']:.4f}%", flush=True)

    examples = random.Random(args.seed).sample(filtered, k=min(3, len(filtered)))
    settings = {
        "seed": args.seed, "batch_size": args.batch_size, "workers": args.workers,
        "train_root": str(args.train_root), "split_sha256": split_hash, "classes": classes,
        "checkpoints": dict(zip(("v1", "v2"), metadata, strict=True)),
        "exact_duplicate_definition": "SHA-256 of decoded RGB pixels with dimensions",
        "duplicate_counts": {key: value for key, value in duplicate_report.items() if key != "groups"},
        "perturbations": PERTURBATIONS, "fill_rgb": FILL,
        "perturbation_seed_key": "SHA-256 of JSON [seed, relative POSIX path, condition]",
        "eval_transform": repr(evaluation), "precision": "float32",
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "pillow": pillow_version, "matplotlib": matplotlib.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "timing": "CUDA-synchronized model forward only, after full-batch warmup, V1 then V2",
        "example_paths": [row.path.relative_to(args.train_root).as_posix() for row in examples],
    }
    preview = torchvision.transforms.Compose(evaluation.transforms[:2])
    write_reports(args.output, settings, results, examples, args.train_root, args.seed, preview)


if __name__ == "__main__":
    main()
