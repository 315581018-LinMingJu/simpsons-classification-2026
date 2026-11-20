"""使用 V14-A3 與 V14-B4 的十視角機率產生 V16 候選。"""

import argparse
import fcntl
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import classification_v14 as v14
from classification_v2 import transforms_v2
from classification_v8 import TestImages
from classification_v9 import sources as test_sources
from v2.data_checks import TEST_COUNT, read_json
from v3.inference import SCALES, Prediction, Views, infer_view_batches
from v3.reports import read_reference, submission
from v3.storage import file_sha256, image_signature, json_bytes, publish, publish_csv
from v8.data import CONDITIONS, EvaluationImages, environment as package_environment, selected_signature
from v8.experiment import data_loader, init_cuda
from v8.models import make_model
from v12.model import environment, permitted
from v16.probabilities import cache_or_infer, fuse, read_cache, replay_mismatches, validate_matrix
from v16.report import screen

EPOCHS = {"a": 3, "b": 4}
METHOD = "multiscale"
SOURCE_FILES = ("classification_v16.py", "classification_v8.py", "classification_v9.py",
                "v3/reports.py", "v16/probabilities.py", "v16/report.py")


@dataclass(frozen=True)
class Arm:
    checkpoint: Path
    settings: dict
    epoch: int
    archive_dir: Path
    archive: dict
    fingerprint: dict


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "probe", "validate", "predict"))
    p.add_argument("--output", type=Path, default=Path("runs/v16"))
    p.add_argument("--v14-run", type=Path, default=Path("runs/v14_continuation"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--archive-a", type=Path, help="V14 A Epoch 3 的 summary.json 與 submission.csv 所在目錄")
    p.add_argument("--archive-b", type=Path, help="V14 B Epoch 4 的 summary.json 與 submission.csv 所在目錄")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    root = Path("runs").resolve() / "v16"
    out = args.output.resolve()
    inputs = (args.data, args.v14_run, args.v8_run, args.archive_a, args.archive_b)
    if (root.is_symlink() or not out.is_relative_to(root) or any(
            out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out)
            for path in inputs if path is not None)):
        p.error("--output must be independent under runs/v16")
    if args.workers < 0:
        p.error("--workers must be nonnegative")
    args.output = out
    return args


def probe():
    import torch

    info = environment()
    if not info["cuda_available"] or info["devices_visible"] != 1:
        return {**info, "status": "requires_one_visible_cuda_gpu"}
    try:
        info["free_gib_at_probe"] = torch.cuda.mem_get_info()[0] / 2**30
        with torch.inference_mode():
            model = make_model("convnext_tiny", 50).cuda().eval()
            output = model(torch.zeros((1, 3, 224, 224), device="cuda")).softmax(dim=1)
            validate_matrix(output.cpu().numpy(), 1, 50)
    except (RuntimeError, ValueError) as exc:
        return {**info, "status": "fp32_probe_failed", "error": str(exc)}
    return {**info, "status": "ready"}


def load_arm(args, name, prepare_sha, test, ids, classes):
    epoch = EPOCHS[name]
    folder = args.v14_run / name
    path = folder / f"epoch_{epoch:02d}.pt"
    settings = read_json(folder / "settings.json")
    if (settings.get("arm") != name or settings.get("prepare_sha256") != prepare_sha
            or settings.get("protocol") != v14.FORMAT):
        raise ValueError(f"V14 {name.upper()} settings do not match preparation")
    digest = file_sha256(path)
    matches = [row for row in read_json(folder / "history.json") if row.get("epoch") == epoch]
    if len(matches) != 1 or matches[0].get("checkpoint_sha256") != digest:
        raise ValueError(f"V14 {name.upper()} checkpoint SHA does not match history")
    state = v14.read_epoch(path, settings, epoch)
    if state["training"] != {k: v for k, v in matches[0].items() if k != "checkpoint_sha256"}:
        raise ValueError(f"V14 {name.upper()} checkpoint and recorded training metrics differ")
    if tuple(state["model"]["classifier.2.weight"].shape) != (len(classes), 768):
        raise ValueError(f"V14 {name.upper()} ConvNeXt-Tiny class count does not match")
    del state

    archive_dir = getattr(args, f"archive_{name}") or folder / f"candidate_epoch_{epoch:02d}"
    summary = read_json(archive_dir / "summary.json")
    csv_path = archive_dir / "submission.csv"
    csv_sha = file_sha256(csv_path)
    expected = {"arm": name, "epoch": epoch, "checkpoint_sha256": digest, "submission_sha256": csv_sha,
                "test_provenance": test.provenance, "images": len(ids), "method": METHOD, "denoise": True}
    if any(summary.get(k) != v for k, v in expected.items()) or summary.get("denoise") is not True:
        raise ValueError(f"V14 {name.upper()} archived submission provenance differs")
    batch_size = summary.get("batch_size")
    if type(batch_size) is not int or batch_size not in (4, 8, 16, 32):
        raise ValueError(f"V14 {name.upper()} archived Test batch size is invalid")
    archive = read_reference(csv_path, ids, classes)
    seconds = summary.get("seconds")
    fingerprint = {"epoch": epoch, "checkpoint_sha256": digest, "submission_sha256": csv_sha,
                   "summary_sha256": file_sha256(archive_dir / "summary.json"), "test_batch_size": batch_size,
                   "archived_test_seconds": seconds if type(seconds) in (int, float)
                   and math.isfinite(seconds) and seconds > 0 else None}
    return Arm(path, settings, epoch, archive_dir, archive, fingerprint)


def prepare_inputs(args):
    info = v14.context(args)
    record = v14.prepared(SimpleNamespace(output=args.v14_run), info)
    prepare_sha = file_sha256(args.v14_run / "prepare.json")
    classes = info.inputs["classes"]
    screen_ids = [case.item for case in info.screen]
    if (len(classes) != 50 or len(set(classes)) != 50 or screen_ids != record["screen_ids"]
            or record["screen_images"] != len(screen_ids)):
        raise ValueError("V14 class order or fixed Screen IDs differ")
    if type(record["eval_batch_size"]) is not int or record["eval_batch_size"] not in (4, 8, 16, 32):
        raise ValueError("Invalid V14 Screen batch size")
    sources = SimpleNamespace(data=args.data, v8_run=args.v8_run, checkpoint=info.checkpoint,
                              v8_submission=args.v8_run / "candidates/convnext_tta_denoise/submission_v8.csv")
    test = test_sources(sources)
    if (test.inputs["classes"] != classes or test.inputs["split_sha256"] != info.inputs["split_sha256"]
            or test.provenance["checkpoint_sha256"] != info.identity["checkpoint_sha256"]):
        raise ValueError("V14 Screen and Test class order or data split differ")
    test_ids = [item for _, item in test.samples]
    arms = {name: load_arm(args, name, prepare_sha, test, test_ids, classes) for name in EPOCHS}
    code_root = Path(__file__).resolve().parent
    identity = {
        "protocol": "simpsons-v16-v1", "parent_prepare_sha256": prepare_sha, "v14": info.identity,
        "test": test.provenance, "arms": {name: arm.fingerprint for name, arm in arms.items()},
        "code_sha256": {name: file_sha256(code_root / name) for name in SOURCE_FILES},
        "classes": classes, "screen_ids": screen_ids, "screen_labels": [case.label for case in info.screen],
        "screen_batch_size": record["eval_batch_size"], "method": METHOD, "denoise": True,
        "precision": "float32", "scales": list(SCALES), "view_count": 10,
        "screen_conditions": list(CONDITIONS), "packages": package_environment()["packages"],
    }
    return SimpleNamespace(info=info, test=test, ids={"screen": screen_ids, "test": test_ids},
                           arms=arms, identity=identity, classes=classes)


def safe_path(output, relative):
    path = output / relative
    if path.resolve() != path:
        raise ValueError(f"V16 output paths may not be redirected: {path}")
    return path


def cache_spec(args, source, name, scope, condition=None):
    relative = Path("cache") / scope / name / f"{condition or 'all'}.npz"
    path = safe_path(args.output, relative)
    identity = {"prepare_sha256": file_sha256(args.output / "prepare.json"), "arm": name,
                "scope": scope, "condition": condition,
                "batch_size": (source.identity["screen_batch_size"] if scope == "screen"
                               else source.arms[name].fingerprint["test_batch_size"])}
    return path, identity


def prepare(args, source):
    path = safe_path(args.output, "prepare.json")
    publish(path, json_bytes(source.identity))
    status, missing_cost = {}, []
    for name, arm in source.arms.items():
        cache, metadata = cache_spec(args, source, name, "test")
        if cache.is_file():
            read_cache(cache, source.ids["test"], len(source.classes), metadata)
            status[name] = "verified"
        else:
            status[name] = "missing"
            missing_cost.append(arm.fingerprint["archived_test_seconds"])
    v14.print_json({"decision": "prepared", "checkpoint_sha256": {
        name: arm.fingerprint["checkpoint_sha256"] for name, arm in source.arms.items()},
        "screen_images": len(source.ids["screen"]), "test_images": len(source.ids["test"]),
        "test_cache": status, "test_missing_view_forwards": 10 * len(source.ids["test"]) * len(missing_cost),
        "test_estimated_seconds_from_v14": sum(missing_cost) if all(x is not None for x in missing_cost) else None})


def verify_images(args, source, scope):
    if scope == "screen":
        if selected_signature(source.info.root, source.info.screen) != source.info.identity["screen_signature"]:
            raise ValueError("V14 Screen images changed during inference")
    elif (image_signature(p for p, _ in source.test.samples) != source.test.provenance["test_image_signature"]
          or file_sha256(args.data / "sample_submission.csv") != source.test.provenance["template_sha256"]):
        raise ValueError("V14 Test images or template changed during inference")


def verify_unchanged(args, source, scope):
    verify_images(args, source, scope)
    for arm in source.arms.values():
        for path, field in ((arm.checkpoint, "checkpoint_sha256"),
                            (arm.archive_dir / "submission.csv", "submission_sha256"),
                            (arm.archive_dir / "summary.json", "summary_sha256")):
            if file_sha256(path) != arm.fingerprint[field]:
                raise ValueError(f"V14 source changed during V16 inference: {path}")


def collect(args, source, name, scope):
    arm = source.arms[name]
    model = None
    seed = source.info.inputs["seed"]
    ids = source.ids[scope]
    matrices, seconds = {}, {}

    def batches(condition, batch_size):
        nonlocal model
        if model is None:
            permitted(args.allow_imagenet)
            init_cuda(seed)
            model = make_model("convnext_tiny", len(source.classes))
            checkpoint = v14.read_epoch(arm.checkpoint, arm.settings, arm.epoch)
            model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
            if file_sha256(arm.checkpoint) != arm.fingerprint["checkpoint_sha256"]:
                raise ValueError(f"V14 {name.upper()} checkpoint changed")
            model.cuda().eval()
        views = Views(METHOD, transforms_v2()[1])
        if scope == "screen":
            dataset = EvaluationImages(source.info.root, source.info.screen, condition, views, seed, True)
        else:
            dataset = TestImages(source.test.samples, views, True)
        loader = data_loader(dataset, batch_size, args.workers)
        yield from infer_view_batches(model, loader, warmup=scope == "test")
        verify_images(args, source, scope)

    for condition in CONDITIONS if scope == "screen" else (None,):
        path, metadata = cache_spec(args, source, name, scope, condition)
        label = f"V16 {name.upper()} {scope}/{condition or 'all'}"
        p, elapsed, _ = cache_or_infer(path, ids, len(source.classes), metadata,
                                      lambda: batches(condition, metadata["batch_size"]), label)
        matrices[condition], seconds[condition or "all"] = p, elapsed
    return matrices, seconds


def screen_result(args, source, matrices, seconds):
    labels = np.asarray(source.identity["screen_labels"])
    result = screen(matrices["a"], matrices["b"], labels, source.info.inputs["seed"])
    result.update(prepare_sha256=file_sha256(args.output / "prepare.json"),
                  seconds=seconds)
    return result


def validate(args, source):
    matrices, seconds = {}, {}
    for name in EPOCHS:
        matrices[name], seconds[name] = collect(args, source, name, "screen")
    verify_unchanged(args, source, "screen")
    result = screen_result(args, source, matrices, seconds)
    publish(safe_path(args.output, "validation.json"), json_bytes(result))
    v14.print_json({"reused_screen": True, "weight_a": result["weight_a"], "selection": result["selection"],
                    "trials": {name: {"clean_pct": row["conditions"]["clean"]["accuracy_pct"],
                                      "hard_pct": row["hard_accuracy_pct"], "hard_vs_a": row["hard_effect"],
                                      "clean_disagreement_pct_vs_a": row["conditions"]["clean"]["disagreement"]["pp"]}
                               for name, row in result["trials_vs_a"].items()}})


def checked_replay(args, source, name, matrix):
    ids, classes = source.ids["test"], source.classes
    differences = replay_mismatches(matrix, ids, source.arms[name].archive)
    if differences:
        path = safe_path(args.output, f"replay_mismatch_{name}.csv")
        publish_csv(path, ("id", "v14_label", "current_label"),
                    ((item, classes[old], classes[new]) for item, old, new in differences))
        raise SystemExit(f"V14 {name.upper()} Top-1 有 {len(differences)} 筆不一致。詳見 {path}。")
    print(f"V16 {name.upper()} replay | {len(ids):,}/{len(ids):,} exact Top-1 matches", flush=True)


def predict(args, source):
    path = safe_path(args.output, "validation.json")
    if not path.is_file():
        raise SystemExit("先執行 V16 validate。")
    report = read_json(path)
    prepare_sha = file_sha256(args.output / "prepare.json")
    screen_probs, screen_seconds = {}, {}
    for name in EPOCHS:
        screen_probs[name], screen_seconds[name] = {}, {}
        for condition in CONDITIONS:
            cache, metadata = cache_spec(args, source, name, "screen", condition)
            p, elapsed = read_cache(cache, source.ids["screen"], len(source.classes), metadata)
            screen_probs[name][condition], screen_seconds[name][condition] = p, elapsed
    if report != screen_result(args, source, screen_probs, screen_seconds):
        raise ValueError("V16 validation selection does not match the verified Screen probabilities")
    weight = report["weight_a"]
    ids, classes = source.ids["test"], source.classes
    destination = safe_path(args.output, "candidate")
    if destination.exists():
        raise FileExistsError(f"V16 candidate already exists: {destination}")
    matrices, timings = {}, {}
    for name in EPOCHS:
        values, timings[name] = collect(args, source, name, "test")
        matrices[name] = values[None]
        checked_replay(args, source, name, matrices[name])
    verify_unchanged(args, source, "test")
    probabilities = fuse(matrices["a"], matrices["b"], weight)
    validate_matrix(probabilities, len(ids), len(classes))
    predictions = {item: Prediction(int(label), float(probabilities[i, label]), 0.0, 0.0)
                   for i, (item, label) in enumerate(zip(ids, probabilities.argmax(axis=1), strict=True))}
    with TemporaryDirectory(prefix=".candidate-", dir=args.output) as temporary:
        staged = Path(temporary)
        csv = staged / "submission_v16.csv"
        submission(csv, ids, predictions, classes, count=TEST_COUNT)
        saved = read_reference(csv, ids, classes)
        rows, moves, changed = v14.test_comparison(ids, source.arms["a"].archive, saved, classes)
        publish_csv(staged / "class_changes.csv",
                    ("character", "a_count", "v16_count", "removed", "added", "net"), rows)
        publish_csv(staged / "transitions.csv", ("a_label", "v16_label", "count"), moves)
        a_b_changes = sum(source.arms["a"].archive[i] != source.arms["b"].archive[i] for i in ids)
        result = {"submission": str(destination / csv.name), "submission_sha256": file_sha256(csv),
                  "images": len(ids), "replay_mismatches": {"a": 0, "b": 0},
                  "weight_a": weight, "weight_b": round(1 - weight, 2), "selection": report["selection"],
                  "changed_vs_a": changed, "changed_vs_a_pct": 100 * changed / len(ids),
                  "a_b_disagreement": a_b_changes, "a_b_disagreement_pct": 100 * a_b_changes / len(ids),
                  "seconds": timings, "prepare_sha256": prepare_sha, "validation_sha256": file_sha256(path),
                  "checkpoint_sha256": {name: arm.fingerprint["checkpoint_sha256"]
                                        for name, arm in source.arms.items()}}
        publish(staged / "summary.json", json_bytes(result))
        staged.rename(destination)
    v14.print_json(result)


def main():
    args = arguments()
    if args.stage == "probe":
        result = probe()
        v14.print_json(result)
        return 0 if result["status"] == "ready" else 1
    args.output.mkdir(parents=True, exist_ok=True)
    with safe_path(args.output, "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V16 輸出已有階段正在使用。")
        source = prepare_inputs(args)
        path = safe_path(args.output, "prepare.json")
        if args.stage == "prepare":
            prepare(args, source)
        elif not path.is_file() or read_json(path) != source.identity:
            raise SystemExit("V16 prepare 缺少或來源已變更。請先執行 prepare，必要時另指定 V16 輸出目錄。")
        elif args.stage == "validate":
            validate(args, source)
        else:
            predict(args, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
