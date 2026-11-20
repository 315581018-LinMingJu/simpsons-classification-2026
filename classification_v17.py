"""V17 使用 V14-A Epoch 3／4 的 ConvNeXt-Tiny 模型權重平均。"""

import argparse
import fcntl
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import classification_v14 as v14
import classification_v16 as v16
from classification_v2 import transforms_v2
from classification_v8 import TestImages
from classification_v9 import sources as test_sources
from v2.data_checks import TEST_COUNT, read_json
from v3.inference import SCALES, Prediction, Views, infer_view_batches
from v3.reports import read_reference, submission
from v3.storage import file_sha256, json_bytes, publish, publish_csv
from v8.data import CONDITIONS, HARD_WEIGHTS, EvaluationImages, environment as package_environment
from v8.experiment import data_loader, init_cuda
from v8.models import make_model
from v12.model import permitted
from v16.probabilities import cache_or_infer, read_cache
from v16.report import compare
from v17.weights import average, epoch_model

METHOD = "multiscale"
STATES = ("a3", "a4", "v17")
MAX_DROP = {"clean_pp": 2.0, "hard_pp": 3.0, "any_hard_condition_pp": 10.0}
FORMAT = "simpsons-v17-a3-a4-weight-average-v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "probe", "validate", "predict"))
    p.add_argument("--output", type=Path, default=Path("runs/v17"))
    p.add_argument("--data", type=Path, default=Path("data/extracted"))
    p.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    p.add_argument("--v14-run", type=Path, default=Path("runs/v14_continuation"))
    p.add_argument("--archive-a3", type=Path, help="V14-A Epoch 3 的 summary.json 與 submission.csv 所在目錄")
    p.add_argument("--v16-candidate", type=Path, help="V16 A80/B20 的 submission_v16.csv 與 summary.json 所在目錄")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-imagenet", action="store_true")
    args = p.parse_args(argv)
    root = Path("runs").resolve() / "v17"
    out = args.output.resolve()
    if (root.is_symlink() or not out.is_relative_to(root) or any(
            out.is_relative_to(path.resolve()) or path.resolve().is_relative_to(out)
            for path in (args.data, args.v8_run, args.v14_run, args.archive_a3, args.v16_candidate)
            if path is not None)):
        p.error("--output must be independent under runs/v17")
    if args.workers < 0:
        p.error("--workers must be nonnegative")
    args.output = out
    return args


def v16_reference(args, test, ids, classes, a3_sha, v14_identity):
    base = Path("runs/v16")
    if args.v16_candidate is None:
        paths = [base / "candidate", *sorted(base.glob("*/candidate"))]
        matches = []
        for folder in paths:
            path = folder / "summary.json"
            if path.is_file():
                summary = read_json(path)
                if (summary.get("weight_a") == .8 and summary.get("weight_b") == .2
                        and summary.get("checkpoint_sha256", {}).get("a") == a3_sha):
                    matches.append(folder)
        if len(matches) != 1:
            raise ValueError("找不到唯一的 V16 A80/B20 封存，請使用 --v16-candidate 指定候選目錄。")
        folder = matches[0]
    else:
        folder = args.v16_candidate
    summary_path = folder / "summary.json"
    csv_path = folder / "submission_v16.csv"
    plan_path = folder.parent / "prepare.json"
    validation_path = folder.parent / "validation.json"
    summary = read_json(summary_path)
    plan = read_json(plan_path)
    validation = read_json(validation_path)
    digest = file_sha256(csv_path)
    plan_sha = file_sha256(plan_path)
    validation_sha = file_sha256(validation_path)
    parent = plan.get("arms", {})
    a, b = parent.get("a", {}), parent.get("b", {})
    if (plan.get("protocol") != "simpsons-v16-v1" or plan.get("v14") != v14_identity
            or plan.get("test") != test.provenance or plan.get("classes") != classes
            or a.get("epoch") != 3 or a.get("checkpoint_sha256") != a3_sha
            or b.get("epoch") != 4 or not b.get("checkpoint_sha256")
            or summary.get("checkpoint_sha256") != {"a": a3_sha, "b": b["checkpoint_sha256"]}
            or summary.get("prepare_sha256") != plan_sha or summary.get("validation_sha256") != validation_sha
            or validation.get("prepare_sha256") != plan_sha
            or any(row.get("weight_a") != .8 or row.get("weight_b") != .2 for row in (summary, validation))
            or summary.get("replay_mismatches") != {"a": 0, "b": 0}
            or summary.get("images") != len(ids) or summary.get("submission_sha256") != digest):
        raise ValueError("V16 A80/B20 source, replay evidence, or submission SHA differs")
    fingerprint = {"csv_sha256": digest, "summary_sha256": file_sha256(summary_path),
                   "prepare_sha256": plan_sha, "validation_sha256": validation_sha}
    return read_reference(csv_path, ids, classes), fingerprint, {
        csv_path: digest, summary_path: fingerprint["summary_sha256"], plan_path: plan_sha,
        validation_path: validation_sha,
    }


def prepare_inputs(args):
    info = v14.context(args)
    record = v14.prepared(SimpleNamespace(output=args.v14_run), info)
    prepare_sha = file_sha256(args.v14_run / "prepare.json")
    classes = info.inputs["classes"]
    screen_ids = [case.item for case in info.screen]
    if (len(classes) != 50 or len(set(classes)) != 50 or screen_ids != record["screen_ids"]
            or len(screen_ids) != record["screen_images"] or record["eval_batch_size"] not in (4, 8, 16, 32)):
        raise ValueError("V14 classes, fixed Screen IDs, or validation batch size differ")
    v8_csv = args.v8_run / "candidates/convnext_tta_denoise/submission_v8.csv"
    test = test_sources(SimpleNamespace(data=args.data, v8_run=args.v8_run, checkpoint=info.checkpoint,
                                       v8_submission=v8_csv))
    if (test.inputs["classes"] != classes or test.inputs["split_sha256"] != info.inputs["split_sha256"]
            or test.provenance["checkpoint_sha256"] != info.identity["checkpoint_sha256"]):
        raise ValueError("V14 Screen and Test class order, split, or reference weights differ")
    test_ids = [item for _, item in test.samples]
    a3 = v16.load_arm(SimpleNamespace(v14_run=args.v14_run, archive_a=args.archive_a3),
                      "a", prepare_sha, test, test_ids, classes)
    folder = args.v14_run / "a"
    history = read_json(folder / "history.json")
    rows = {row["epoch"]: row for row in history}
    if len(rows) != len(history) or list(rows) != list(range(1, len(history) + 1)) or 4 not in rows:
        raise ValueError("V14-A epoch history must contain a contiguous Epoch 3 / 4 trajectory")
    checkpoints = {"a3": a3.checkpoint, "a4": folder / "epoch_04.pt"}
    hashes = {name: file_sha256(path) for name, path in checkpoints.items()}
    if hashes != {"a3": a3.fingerprint["checkpoint_sha256"], "a4": rows[4]["checkpoint_sha256"]}:
        raise ValueError("V14-A checkpoint SHA differs from the recorded epochs")
    v16_labels, v16_fingerprint, references = v16_reference(args, test, test_ids, classes, hashes["a3"], info.identity)
    references.update({a3.archive_dir / "summary.json": a3.fingerprint["summary_sha256"],
                       a3.archive_dir / "submission.csv": a3.fingerprint["submission_sha256"],
                       args.v14_run / "prepare.json": prepare_sha,
                       folder / "settings.json": file_sha256(folder / "settings.json"),
                       folder / "history.json": file_sha256(folder / "history.json"),
                       v8_csv: test.provenance["submission_sha256"]})
    code = Path(__file__).resolve().parent
    identity = {
        "protocol": FORMAT, "architecture": "convnext_tiny", "averaging": {"a3": .5, "a4": .5},
        "v14": info.identity, "v14_prepare_sha256": prepare_sha, "split_sha256": test.inputs["split_sha256"],
        "checkpoint_sha256": hashes, "a3_archive": a3.fingerprint, "v16_archive": v16_fingerprint,
        "test": test.provenance, "classes": classes, "screen_ids": screen_ids,
        "screen_labels": [case.label for case in info.screen], "screen_conditions": list(CONDITIONS),
        "screen_batch_size": record["eval_batch_size"], "test_batch_size": a3.fingerprint["test_batch_size"],
        "precision": "float32", "method": METHOD, "denoise": True, "scales": list(SCALES), "view_count": 10,
        "max_drop_vs_a3": MAX_DROP, "hard_weights": HARD_WEIGHTS, "packages": package_environment()["packages"],
        "code_sha256": {name: file_sha256(code / name) for name in (
            "classification_v17.py", "v17/weights.py", "classification_v16.py", "v16/probabilities.py",
            "v16/report.py", "classification_v8.py", "classification_v9.py", "v3/reports.py")},
    }
    return SimpleNamespace(info=info, test=test, ids={"screen": screen_ids, "test": test_ids}, identity=identity,
                           classes=classes, settings=a3.settings, rows=rows, checkpoints=checkpoints,
                           references=references, a3_labels=a3.archive, v16_labels=v16_labels)


def load_parent(source, name):
    epoch = {"a3": 3, "a4": 4}[name]
    return epoch_model(source.checkpoints[name], source.settings, epoch, source.rows[epoch],
                       source.identity["checkpoint_sha256"][name])


def verify_unchanged(args, source, scope):
    v16.verify_images(args, source, scope)
    references = {**source.references,
                  **{path: source.identity["checkpoint_sha256"][name] for name, path in source.checkpoints.items()}}
    for path, expected in references.items():
        if file_sha256(path) != expected:
            raise ValueError(f"V17 source changed: {path}")


def save_average(path, source):
    import torch

    # 避免 CPU 對大量小張量啟動過多執行緒。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = make_model("convnext_tiny", len(source.classes))
    weights, stats = average(load_parent(source, "a3"), load_parent(source, "a4"), model.state_dict())
    with NamedTemporaryFile(dir=path.parent) as temporary:
        torch.save({"identity": source.identity, "model": weights, "tensors": stats}, temporary)
        temporary.flush()
        os.link(temporary.name, path)


def prepared(args, source, create=False):
    path = v16.safe_path(args.output, "averaged.pt")
    manifest = v16.safe_path(args.output, "prepare.json")
    if manifest.is_file():
        record = read_json(manifest)
        if record["identity"] != source.identity or file_sha256(path) != record["averaged_sha256"]:
            raise ValueError("V17 prepared source or averaged checkpoint changed; choose a fresh --output")
        return record
    if not create:
        raise SystemExit("先執行 V17 prepare。")

    import torch

    if not path.exists():
        save_average(path, source)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["identity"] != source.identity:
        raise ValueError("V17 averaged checkpoint identity differs; choose a fresh --output")
    verify_unchanged(args, source, "screen")
    record = {"identity": source.identity, "averaged_sha256": file_sha256(path), "tensors": saved["tensors"]}
    publish(manifest, json_bytes(record))
    return record


def cache_spec(args, name, scope, condition, manifest_sha):
    path = v16.safe_path(args.output, Path("cache") / scope / name / f"{condition or 'all'}.npz")
    metadata = {"prepare_sha256": manifest_sha, "model": name, "scope": scope, "condition": condition}
    return path, metadata


def collect(args, source, record, name, scope):
    import torch

    manifest_sha = file_sha256(args.output / "prepare.json")
    model = None
    ids = source.ids[scope]
    matrices, timings = {}, {}

    def batches(condition):
        nonlocal model
        seed = source.info.inputs["seed"]
        if model is None:
            permitted(args.allow_imagenet)
            init_cuda(seed)
            model = make_model("convnext_tiny", len(source.classes))
            if name == "v17":
                path = v16.safe_path(args.output, "averaged.pt")
                state = torch.load(path, map_location="cpu", weights_only=True)
                if (state["identity"] != source.identity or file_sha256(path) != record["averaged_sha256"]):
                    raise ValueError("V17 averaged checkpoint changed")
                weights = state["model"]
            else:
                weights = load_parent(source, name)
            model.load_state_dict(weights, strict=True)
            model.cuda().eval()
        views = Views(METHOD, transforms_v2()[1])
        if scope == "screen":
            dataset = EvaluationImages(source.info.root, source.info.screen, condition, views, seed, True)
        else:
            dataset = TestImages(source.test.samples, views, True)
        loader = data_loader(dataset, source.identity[f"{scope}_batch_size"], args.workers)
        yield from infer_view_batches(model, loader, warmup=scope == "test")
        v16.verify_images(args, source, scope)

    for condition in CONDITIONS if scope == "screen" else ("all",):
        path, metadata = cache_spec(args, name, scope, condition, manifest_sha)
        matrices[condition], timings[condition], _ = cache_or_infer(
            path, ids, len(source.classes), metadata, lambda: batches(condition), f"V17 {name} {scope}/{condition}")
    return matrices, timings


def screen_result(matrices, labels, seed):
    predictions = {name: {c: matrix.argmax(axis=1) for c, matrix in conditions.items()}
                   for name, conditions in matrices.items()}
    trials = {name: compare(predictions["a3"], predictions[name], labels, seed) for name in STATES}
    vs_a4 = compare(predictions["a4"], predictions["v17"], labels, seed)
    averaged = trials["v17"]
    gaps = {"clean_pp": averaged["conditions"]["clean"]["accuracy_effect"]["pp"],
            "hard_pp": averaged["hard_effect"]["pp"],
            "any_hard_condition_pp": min(averaged["conditions"][c]["accuracy_effect"]["pp"] for c in HARD_WEIGHTS)}
    breaches = [key for key, maximum in MAX_DROP.items() if gaps[key] < -maximum - 1e-9]
    return {"decision": "stop_severe_regression" if breaches else "allow_candidate",
            "guardrail_breaches": breaches, "gaps_vs_a3": gaps, "max_drop_vs_a3": MAX_DROP,
            "reused_screen": True, "samples_per_condition": len(labels),
            "trials_vs_a3": trials, "average_vs_a4": vs_a4}


def validation_result(args, source, matrices, timings):
    report = screen_result(matrices, np.asarray(source.identity["screen_labels"]), source.info.inputs["seed"])
    return {**report, "prepare_sha256": file_sha256(args.output / "prepare.json"), "seconds": timings}


def validate(args, source, record):
    matrices, timings = {}, {}
    for name in STATES:
        matrices[name], timings[name] = collect(args, source, record, name, "screen")
    verify_unchanged(args, source, "screen")
    report = validation_result(args, source, matrices, timings)
    publish(v16.safe_path(args.output, "validation.json"), json_bytes(report))
    v14.print_json({**{k: report[k] for k in ("decision", "gaps_vs_a3", "guardrail_breaches")},
                    "scores": {name: {"clean_pct": row["conditions"]["clean"]["accuracy_pct"],
                                      "hard_pct": row["hard_accuracy_pct"]}
                               for name, row in report["trials_vs_a3"].items()}})


def verified_validation(args, source):
    path = v16.safe_path(args.output, "validation.json")
    if not path.is_file():
        raise SystemExit("先執行 V17 validate。")
    report = read_json(path)
    matrices, timings = {}, {}
    manifest_sha = file_sha256(args.output / "prepare.json")
    for name in STATES:
        matrices[name], timings[name] = {}, {}
        for condition in CONDITIONS:
            cache, metadata = cache_spec(args, name, "screen", condition, manifest_sha)
            matrices[name][condition], timings[name][condition] = read_cache(
                cache, source.ids["screen"], len(source.classes), metadata)
    if report != validation_result(args, source, matrices, timings):
        raise ValueError("V17 validation report differs from the verified Screen probabilities")
    if report["decision"] != "allow_candidate":
        raise SystemExit(f"V17 停止產生候選。護欄觸發項目 {report['guardrail_breaches']}")
    return file_sha256(path)


def predict(args, source, record):
    validation_sha = verified_validation(args, source)
    dest = v16.safe_path(args.output, "candidate")
    if dest.exists():
        raise FileExistsError(f"V17 candidate already exists: {dest}")
    p, seconds = collect(args, source, record, "v17", "test")
    verify_unchanged(args, source, "test")
    ids, classes, probabilities = source.ids["test"], source.classes, p["all"]
    labels = probabilities.argmax(axis=1)
    predictions = {item: Prediction(int(label), float(probabilities[i, label]), 0.0, 0.0)
                   for i, (item, label) in enumerate(zip(ids, labels, strict=True))}
    with TemporaryDirectory(prefix=".candidate-", dir=args.output) as temporary:
        staged = Path(temporary)
        csv = staged / "submission_v17.csv"
        submission(csv, ids, predictions, classes, count=TEST_COUNT)
        saved = read_reference(csv, ids, classes)
        result = {"submission": str(dest / csv.name), "submission_sha256": file_sha256(csv), "images": len(ids),
                  "averaged_sha256": record["averaged_sha256"], "validation_sha256": validation_sha,
                  "prepare_sha256": file_sha256(args.output / "prepare.json"), "test_seconds": seconds["all"]}
        for name, reference in (("a3", source.a3_labels), ("v16", source.v16_labels)):
            counts, moves, changed = v14.test_comparison(ids, reference, saved, classes)
            publish_csv(staged / f"class_changes_vs_{name}.csv",
                        ("character", "reference_count", "v17_count", "removed", "added", "net"), counts)
            publish_csv(staged / f"transitions_vs_{name}.csv", ("reference_label", "v17_label", "count"), moves)
            result[f"changed_vs_{name}"] = changed
            result[f"changed_vs_{name}_pct"] = 100 * changed / len(ids)
        publish(staged / "summary.json", json_bytes(result))
        staged.rename(dest)
    v14.print_json(result)


def main(argv=None):
    args = arguments(argv)
    if args.stage == "probe":
        result = v16.probe()
        v14.print_json(result)
        return 0 if result["status"] == "ready" else 1
    args.output.mkdir(parents=True, exist_ok=True)
    with v16.safe_path(args.output, "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V17 輸出已有階段正在使用。")
        source = prepare_inputs(args)
        record = prepared(args, source, create=args.stage == "prepare")
        if args.stage == "prepare":
            v14.print_json({"decision": "prepared", "averaged_sha256": record["averaged_sha256"],
                            "checkpoint_sha256": source.identity["checkpoint_sha256"], "tensors": record["tensors"],
                            "screen_images": len(source.ids["screen"]), "test_images": len(source.ids["test"])})
        elif args.stage == "validate":
            validate(args, source, record)
        else:
            predict(args, source, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
