"""驗證統計、候選 CSV 與測試圖片拼圖。"""

import csv
import io
import textwrap
from collections import Counter
from pathlib import Path

from v2.data_checks import TEST_COUNT
from v3.inference import VIEWS
from v3.storage import json_bytes, publish, publish_csv

FIELDS = ("id", "character")


def read_test_images(folder, template, count=TEST_COUNT):
    with template.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Expected template columns id,character")
        ids = [row["id"] for row in reader]
    if len(ids) != count or len(set(ids)) != count or any(
        not item or item in {".", ".."} or Path(item).name != item for item in ids
    ):
        raise ValueError(f"Expected {count} unique, nonempty test filename IDs")
    paths = {path.stem: path for path in folder.glob("*.jpg") if path.is_file()}
    if set(paths) != set(ids):
        raise ValueError(f"Test IDs and image files differ ({len(set(ids) - paths.keys())} missing, "
                         f"{len(paths.keys() - set(ids))} extra)")
    return [(paths[item], item) for item in ids]


def read_reference(path, ids, classes):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Expected V2 reference columns id,character")
        rows = [(row["id"], row["character"]) for row in reader]
    class_index = {name: i for i, name in enumerate(classes)}
    if [item for item, _ in rows] != ids or any(name not in class_index for _, name in rows):
        raise ValueError("V2 reference submission IDs, order, or class names differ")
    return {item: class_index[name] for item, name in rows}


def submission(path, ids, predictions, classes, count=TEST_COUNT):
    if len(ids) != count or len(set(ids)) != count or not all(ids) or set(ids) != predictions.keys():
        raise ValueError("Missing, extra, empty, or duplicate submission IDs")
    if any(not 0 <= pred.label < len(classes) for pred in predictions.values()):
        raise ValueError("Invalid submission class index")
    publish_csv(path, FIELDS, ((item, classes[predictions[item].label]) for item in ids))


def validation_report(output, targets, predictions, settings):
    baseline = predictions["baseline"]
    baseline_hits = sum(baseline[item].label == label for item, label in targets.items())
    n = len(targets)
    results = []
    for method, scores in predictions.items():
        hits = sum(scores[item].label == label for item, label in targets.items())
        changes = sum(scores[item].label != baseline[item].label for item in targets)
        results.append({
            "method": method, "views": VIEWS[method], "samples": n,
            "accuracy_pct": 100 * hits / n, "delta_vs_baseline_pp": 100 * (hits - baseline_hits) / n,
            "changed_pct": 100 * changes / n,
            "mean_confidence": sum(scores[item].confidence for item in targets) / n,
            "inference_seconds": sum(scores[item].seconds for item in targets),
        })
    publish_csv(output / "comparison.csv", results[0], (row.values() for row in results))
    publish(output / "comparison.json", json_bytes({"settings": settings, "results": results}))
    return results


def test_report(output, samples, predictions, classes, reference, settings):
    ids = [item for _, item in samples]
    v1, v2 = (predictions[key] for key in ("v1_baseline", "baseline"))
    methods = list(VIEWS)
    output.mkdir(parents=True, exist_ok=True)
    distributions = []
    for method, scores in predictions.items():
        counts = Counter(scores[item].label for item in ids)
        distributions.extend((method, name, counts[i], 100 * counts[i] / len(ids))
                             for i, name in enumerate(classes))
    publish_csv(output / "prediction_distribution.csv", ("method", "character", "count", "pct"),
                distributions)
    mismatches = [item for item in ids if v2[item].label != reference[item]]
    summary = {
        "settings": settings, "samples": len(ids),
        "baseline_reference_mismatch_count": len(mismatches),
        "baseline_reference_mismatch_ids": mismatches,
        "v1_v2_changed_count": sum(v1[item].label != v2[item].label for item in ids),
        "methods": {method: {
            "changed_vs_v2": sum(scores[item].label != v2[item].label for item in ids),
            "view_disagreements": sum(scores[item].view_disagreement > 0 for item in ids),
            "mean_confidence": sum(scores[item].confidence for item in ids) / len(ids),
            "inference_seconds": sum(scores[item].seconds for item in ids),
        } for method, scores in predictions.items()},
    }
    publish(output / "test_summary.json", json_bytes(summary))

    fields = ("id", "method", "v1_character", "v2_character", "v2_reference_character", "character",
              "v1_confidence", "v2_confidence", "confidence", "view_disagreement",
              "changed_vs_v2", "v1_v2_changed")
    publish_csv(output / "test_diagnostics.csv", fields, (
        (item, method, classes[v1[item].label], classes[v2[item].label], classes[reference[item]],
         classes[pred.label], v1[item].confidence, v2[item].confidence, pred.confidence,
         pred.view_disagreement, int(pred.label != v2[item].label),
         int(v1[item].label != v2[item].label))
        for method in methods for item in ids for pred in (predictions[method][item],)
    ))

    low, changes, votes = {}, {}, {}
    for method in methods:
        scores = predictions[method]
        low[method] = sorted(ids, key=lambda item: scores[item].confidence)[:4]
        if method != "baseline":
            changed = [item for item in ids if scores[item].label != v2[item].label]
            changes[method] = sorted(changed, key=lambda item: scores[item].confidence)[:4]
        if VIEWS[method] > 1:
            disputed = [item for item in ids if scores[item].view_disagreement > 0]
            votes[method] = sorted(disputed, key=lambda item: -scores[item].view_disagreement)[:4]
    v1_changes = [item for item in ids if v1[item].label != v2[item].label]
    images = {item: path for path, item in samples}
    for filename, selection in (
        ("low_confidence.png", low),
        ("prediction_disagreements.png", changes),
        ("tta_disagreements.png", votes),
        ("v1_v2_disagreements.png", {"multiscale": v1_changes[:4]}),
    ):
        montage(output / filename, selection, images, predictions, classes)
    return summary


def montage(path, selection, images, predictions, classes):
    import matplotlib
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(selection), 4, squeeze=False,
                                figsize=(13, 3.6 * len(selection)), layout="constrained")
    for row, (method, ids) in enumerate(selection.items()):
        for axis in axes[row]:
            axis.axis("off")
        if not ids:
            axes[row, 0].text(0.5, 0.5, f"{method}\nNo matching images", ha="center")
        for column, item in enumerate(ids):
            axis = axes[row, column]
            with Image.open(images[item]) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((320, 240))
            axis.imshow(thumb)
            lines = [f"{method} | ID {item}"]
            for label, key in (("V1", "v1_baseline"), ("V2", "baseline"), ("TTA", method)):
                pred = predictions[key][item]
                name = classes[pred.label].replace("_", " ")
                lines.append(textwrap.fill(f"{label}: {name} ({pred.confidence:.1%})", 33))
            if VIEWS[method] > 1:
                lines.append(f"View disagreement: {predictions[method][item].view_disagreement:.0%}")
            axis.set_title("\n".join(lines), fontsize=8)
    image_data = io.BytesIO()
    figure.savefig(image_data, format="png", dpi=125)
    plt.close(figure)
    publish(path, image_data.getvalue())
