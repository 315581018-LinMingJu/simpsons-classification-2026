"""V11 以合法 Train／Val 驗證角色檢索。"""

import argparse
import fcntl
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import __version__ as pillow_version

from classification_v2 import transforms_v2
from v2.data_checks import COUNTS, TEST_COUNT, read_json
from v3.inference import Views, infer_loader
from v3.storage import file_sha256, json_bytes, publish
from v7.impulse import clean_impulses
from v8.data import benchmark, open_rgb
from v8.experiment import init_cuda, read_checkpoint
from v8.models import make_model
from v11.data import make_plan, plan_counts, render_query, validate_plan
from v11.decision import PROFILES, choose_screen, metrics
from v11.retrieval import Index, extract_batch, query_views, reference_view

PROTOCOL = "v11_role_retrieval_diagnostic_v1"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "pilot"))
    parser.add_argument("--output", type=Path, default=Path("runs/v11"))
    parser.add_argument("--data", type=Path, default=Path("data/extracted"))
    parser.add_argument("--v8-run", type=Path, default=Path("runs/v8_full"))
    parser.add_argument("--cpu-workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--allow-imagenet", action="store_true")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    root = Path("runs").resolve() / "v11"
    protected = (args.data.resolve(), args.v8_run.resolve())
    if root.is_symlink() or not out.is_relative_to(root) or any(
        out.is_relative_to(path) or path.is_relative_to(out) for path in protected
    ):
        parser.error("Use an independent directory under runs/v11")
    return args


def sources(args):
    report, training, validation = benchmark(SimpleNamespace(output=args.v8_run, data=args.data))
    inputs = report["inputs"]
    if inputs["train_per_class"] != 0 or len(training) != COUNTS["train"]:
        raise ValueError("V11 requires the full V8 Train manifest for source isolation")
    folder = args.v8_run / "candidates/convnext_tta_denoise"
    submission = folder / "submission_v8.csv"
    settings = read_json(folder / "settings.json")
    checkpoint = args.v8_run / "models/convnext_all_layers/best.pt"
    digest = file_sha256(checkpoint)
    if (settings.get("checkpoint_sha256") != digest or settings.get("submission_sha256") != file_sha256(submission)
            or settings.get("architecture") != "convnext_tiny" or settings.get("method") != "multiscale"
            or settings.get("denoise") is not True or settings.get("images") != TEST_COUNT
            or type(settings.get("batch_size")) is not int or settings["batch_size"] < 1):
        raise ValueError("V8 checkpoint and named Multi-scale + Denoise archive do not match")
    root = Path(__file__).resolve().parent
    source_files = ("classification_v11.py", "v11/data.py", "v11/retrieval.py", "v11/decision.py",
                    "v8/data.py", "v8/models.py", "v8/experiment.py", "v3/inference.py", "v3/storage.py",
                    "v7/impulse.py", "v5/challenge.py", "v5/background.py", "v4/geometry.py", "v4/analysis.py",
                    "classification_v2.py")
    identity = {"protocol": PROTOCOL, "checkpoint_sha256": digest,
                "v8_submission_sha256": settings["submission_sha256"],
                "v8_settings_sha256": file_sha256(folder / "settings.json"),
                "benchmark_sha256": file_sha256(args.v8_run / "benchmark/prepare.json"),
                "manifest_sha256": report["manifest_sha256"], "v8_image_signature": report["selected_image_signature"],
                "validation_seed": inputs["seed"], "v8_eval_batch_size": settings["batch_size"],
                "code_sha256": {name: file_sha256(root / name) for name in source_files},
                "pillow": pillow_version, "numpy": np.__version__}
    train_root = args.data / "train/train/train"
    return SimpleNamespace(training=training, validation=validation, inputs=inputs, checkpoint=checkpoint,
                           settings=settings, identity=identity, root=train_root)


def print_summary(result):
    print(json_bytes(result).decode(), end="", flush=True)


def audit(args, info):
    path = args.output / "audit.json"
    if path.exists():
        result = read_json(path)
        if result["identity"] != info.identity:
            raise ValueError("V11 audit sources changed. Use a new runs/v11 subdirectory.")
    else:
        started = time.perf_counter()
        print(f"V11 CPU audit | {len(info.training)} original Train fingerprints", flush=True)
        indexed, queries, exclusions = make_plan(info.training, info.validation, info.root,
                                                  info.inputs["seed"], args.cpu_workers)
        ready, counts, by_class = plan_counts(indexed, queries, len(info.inputs["classes"]))
        result = {"identity": info.identity, "index": indexed, "queries": queries, "counts": counts,
                  "index_size": len(indexed), "per_class": by_class, "exclusions": exclusions,
                  "decision": "ready" if ready else "stop_insufficient_source_isolation",
                  "seconds": time.perf_counter() - started}
        publish(path, json_bytes(result))
    print_summary({key: result[key] for key in ("decision", "index_size", "counts", "exclusions", "seconds")})


def chunks(items, size):
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


def build_index(info, manifest, model, normalize, batch_size):
    features = []
    for rows in chunks(manifest, batch_size):
        pairs = [reference_view(open_rgb(info.root / row["item"])) for row in rows]
        features.extend(extract_batch(model, pairs, normalize))
    return Index(manifest, features)


def collect(info, queries, index, model, views, batch_size):
    import torch

    evidence = []
    for rows in chunks(queries, batch_size):
        images = [clean_impulses(render_query(info.root, row, info.inputs["seed"]))[0] for row in rows]
        positive_batch = [row["item"] for row in rows]
        tensors = [torch.stack(col) for col in zip(*(views(image) for image in images), strict=True)]
        (baseline,) = infer_loader(model, [(tensors, positive_batch)], warmup=False)
        if [item for item, _ in baseline] != positive_batch:
            raise ValueError("V8 prediction order changed")
        regions = [query_views(image) for image in images]
        features = []
        for part in chunks([view for bank in regions for view in bank], batch_size):
            features.extend(extract_batch(model, part, views.normalize))
        offset = 0
        for row, (_, score), image_regions in zip(rows, baseline, regions, strict=True):
            group = features[offset:offset + len(image_regions)]
            offset += len(image_regions)
            matches = index.search(group)
            negative = index.search(group, absent_label=row["label"]) if row["role"] == "positive" else []
            evidence.append({**row, "baseline": score.label, "v8_confidence": score.confidence,
                             "matches": matches, "negative_matches": negative})
    return evidence


def write_evidence(path, evidence):
    rows = [{**row, "matches": [asdict(match) for match in row["matches"]],
             "negative_matches": [asdict(match) for match in row["negative_matches"]]} for row in evidence]
    publish(path, json_bytes(rows))


def pilot(args, info):
    import torch
    import torchvision

    path = args.output / "audit.json"
    if not path.is_file():
        raise SystemExit("先執行 V11 audit。")
    plan = read_json(path)
    if plan["identity"] != info.identity:
        raise ValueError("V11 audit and current inputs differ")
    validate_plan(plan["index"], plan["queries"], info.training, info.validation)
    ready, counts, _ = plan_counts(plan["index"], plan["queries"], len(info.inputs["classes"]))
    if not ready or plan["counts"] != counts or plan["decision"] != "ready":
        raise SystemExit("來源隔離樣本不足，V11 停在 CPU 診斷。")
    report_path = args.output / "pilot.json"
    old_report = read_json(report_path) if report_path.exists() else None
    if old_report is not None:
        if old_report["identity"] != info.identity or old_report["audit_sha256"] != file_sha256(path):
            raise ValueError("Existing V11 result uses different sources")
        print_summary(old_report["summary"])
        return
    checkpoint = read_checkpoint(info.checkpoint, info.inputs, args.allow_imagenet)
    if (checkpoint["architecture"] != "convnext_tiny" or checkpoint["config"]["mode"] != "full"
            or checkpoint["config"]["benchmark_sha256"] != info.identity["manifest_sha256"]
            or checkpoint["origin"] != info.settings.get("origin")):
        raise ValueError("Use the V8 all-layers checkpoint associated with the original benchmark")
    seed = info.inputs["seed"]
    init_cuda(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = make_model("convnext_tiny", len(info.inputs["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda().eval()
    views = Views("multiscale", transforms_v2()[1])
    batch_size = info.identity["v8_eval_batch_size"]
    print(f"V11 | {len(plan['index'])} reference images", flush=True)
    index = build_index(info, plan["index"], model, views.normalize, batch_size)
    print("V11 | Screen", flush=True)
    screen = collect(info, [row for row in plan["queries"] if row["fold"] == "screen"],
                     index, model, views, batch_size)
    screen_path = args.output / "screen_evidence.json"
    write_evidence(screen_path, screen)
    screen_result = choose_screen(screen, seed)
    threshold = screen_result["locked_threshold"]
    evidence_hashes = {"screen": file_sha256(screen_path)}
    publish(args.output / "screen.json", json_bytes({
        "audit_sha256": file_sha256(path), "evidence_sha256": evidence_hashes["screen"], **screen_result,
    }))
    confirm_result = None
    if threshold:
        print(f"V11 | Confirm locked to {threshold}", flush=True)
        confirm = collect(info, [row for row in plan["queries"] if row["fold"] == "confirm"],
                          index, model, views, batch_size)
        confirm_path = args.output / "confirm_evidence.json"
        write_evidence(confirm_path, confirm)
        evidence_hashes["confirm"] = file_sha256(confirm_path)
        confirm_result = metrics(confirm, threshold, seed, confirm=True)
    if file_sha256(info.checkpoint) != info.identity["checkpoint_sha256"]:
        raise ValueError("V8 checkpoint changed during the pilot")
    summary = {
        "decision": ("recommend_larger_diagnostic" if confirm_result and confirm_result["decision"] == "pass"
                     else "stop_confirm" if confirm_result else "stop_screen"),
        "screen": screen_result, "confirm": confirm_result,
        "seconds": time.perf_counter() - started, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    report = {"identity": info.identity, "audit_sha256": file_sha256(path), "evidence_sha256": evidence_hashes,
              "profiles": PROFILES, "summary": summary, "scope": "competition_train_and_validation_only",
              "environment": {"python": sys.version, "torch": str(torch.__version__),
                              "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
                              "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name()}}
    publish(report_path, json_bytes(report))
    print_summary(summary)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "stage.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此 V11 輸出已有執行中的階段。")
        info = sources(args)
        if args.stage == "audit":
            audit(args, info)
        else:
            pilot(args, info)


if __name__ == "__main__":
    main()
