"""以共用視角比較 V8 ConvNeXt 的推論配方。"""

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from classification_v2 import transforms_v2
from v3.inference import SCALES, Views
from v3.storage import json_bytes, publish
from v7.impulse import MIN_TRIGGER_FRACTION, clean_impulses
from v8.data import open_rgb, stress

CROPS = ("tl", "tr", "bl", "br", "center")
SCALE_VIEWS = {scale: tuple(f"{scale}_{corner}" for corner in CROPS) for scale in SCALES}
TEN = sum(SCALE_VIEWS.values(), ())
REFERENCE = "v8_multiscale_denoise"


@dataclass(frozen=True)
class Recipe:
    denoise: bool
    views: tuple
    anchor: bool = False


RECIPES = {
    REFERENCE: Recipe(True, TEN),
    "multiscale_raw": Recipe(False, TEN),
    "full_raw": Recipe(False, ("full",)),
    "full_denoise": Recipe(True, ("full",)),
    "center256_denoise": Recipe(True, ("center256",)),
    "five240_denoise": Recipe(True, SCALE_VIEWS[240]),
    "five288_denoise": Recipe(True, SCALE_VIEWS[288]),
    "two_centers_denoise": Recipe(True, ("240_center", "288_center")),
    "full_anchor240_denoise": Recipe(True, ("full", *SCALE_VIEWS[240]), True),
}


def describe_recipes(names):
    return {name: {"denoise": RECIPES[name].denoise, "views": list(RECIPES[name].views),
                   "full_frame_weight": 0.5 if RECIPES[name].anchor else None}
            for name in names}


def requested(names):
    return sorted({(RECIPES[name].denoise, view) for name in names for view in RECIPES[name].views})


@dataclass(frozen=True)
class Example:
    path: Path
    item: str
    distractor: Path | None = None


class ViewImages:
    def __init__(self, examples, plan, condition, seed):
        evaluation = transforms_v2()[1]
        self.examples, self.condition, self.seed = examples, condition, seed
        self.evaluation = evaluation
        self.legacy = Views("multiscale", evaluation)
        self.plan = plan
        self.by_filter = {filtered: {view for denoise, view in plan if denoise == filtered}
                          for filtered in (False, True)}
        self.shared = self.by_filter[True] & self.by_filter[False]

    def __len__(self):
        return len(self.examples)

    def tensor_views(self, image, wanted):
        views = {}
        if "full" in wanted:
            views["full"] = self.legacy.normalize(self.legacy.full_resize(image))
        if "center256" in wanted:
            views["center256"] = self.evaluation(image)
        for keys, resize in zip(SCALE_VIEWS.values(), self.legacy.scales, strict=True):
            if wanted.isdisjoint(keys):
                continue
            crops = self.legacy.crop(resize(image))
            views.update({key: self.legacy.normalize(crop) for key, crop in zip(keys, crops, strict=True)
                          if key in wanted})
        return views

    def __getitem__(self, index):
        example = self.examples[index]
        raw = open_rgb(example.path)
        if self.condition != "test":
            secondary = open_rgb(example.distractor) if self.condition == "distractor" else None
            raw = stress(raw, self.condition, self.seed, example.item, secondary)
        filtered, fraction = clean_impulses(raw) if self.by_filter[True] else (raw, 0.0)
        triggered = fraction >= MIN_TRIGGER_FRACTION
        if triggered:
            raw_views = self.tensor_views(raw, self.by_filter[False])
            filtered_views = self.tensor_views(filtered, self.by_filter[True])
            tensors = [filtered_views[key] if denoise else raw_views[key] for denoise, key in self.plan]
        else:
            shared = self.tensor_views(raw, self.by_filter[False] | self.by_filter[True])
            tensors = [None if not denoise and key in self.shared else shared[key] for denoise, key in self.plan]
        return tensors, example.item, triggered, fraction


def collate_views(batch):
    from torch.utils.data import default_collate

    tensors, ids, triggers, fractions = zip(*batch, strict=True)
    # raw 與 selective 共用未觸發列，避免複製大量影像張量進共享記憶體。
    slots = [[tensor for tensor in slot if tensor is not None] for slot in zip(*tensors, strict=True)]
    return [default_collate(slot) if slot else None for slot in slots], ids, default_collate(triggers), default_collate(fractions)


def infer(model, batches, plan, *, device="cuda"):
    import torch

    # V8 等價的 selective 視角用原批次大小；raw 重算只跑觸發列。
    filtered_indices = {view: i for i, (filtered, view) in enumerate(plan) if filtered}
    order = sorted(range(len(plan)), key=lambda i: not plan[i][0])
    use_cuda = torch.device(device).type == "cuda"
    with torch.inference_mode():
        model(torch.zeros((1, 3, 224, 224), device=device))
        for tensors, ids, triggers, fractions in batches:
            if use_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            active = triggers.numpy()
            probs, labels = {}, {}
            forwards = 0

            def forward(tensor):
                logits = model(tensor.to(device, non_blocking=use_cuda))
                return logits.softmax(1).cpu().numpy(), logits.argmax(1).cpu().numpy()

            for i in order:
                filtered, view = plan[i]
                shared = filtered_indices.get(view) if not filtered else None
                if shared is None:
                    probs[i], labels[i] = forward(tensors[i])
                    forwards += len(ids)
                else:
                    probs[i], labels[i] = probs[shared].copy(), labels[shared].copy()
                    if active.any():
                        probs[i][active], labels[i][active] = forward(tensors[i])
                        forwards += int(active.sum())
            yield (list(ids), np.stack([probs[i] for i in range(len(plan))]),
                   np.stack([labels[i] for i in range(len(plan))]), fractions.numpy(),
                   forwards, time.perf_counter() - started)


@dataclass(frozen=True)
class Bank:
    probabilities: np.ndarray
    labels: np.ndarray
    fractions: np.ndarray
    forwards: int
    inference_seconds: float
    wall_seconds: float


def cache_bank(path, examples, plan, metadata, class_count, get_batches):
    ids = [example.item for example in examples]
    signature = json_bytes({**metadata, "views": plan, "class_count": class_count}).decode()
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if archive["signature"].item() != signature or archive["ids"].tolist() != ids:
                raise ValueError(f"V9 cache metadata or IDs differ: {path}")
            result = Bank(archive["probabilities"], archive["labels"], archive["fractions"],
                          int(archive["forwards"]), float(archive["inference_seconds"]),
                          float(archive["wall_seconds"]))
    else:
        start = time.perf_counter()
        chunks = list(get_batches())
        if [item for batch in chunks for item in batch[0]] != ids:
            raise ValueError("V9 inference changed the requested image order")
        result = Bank(np.concatenate([batch[1] for batch in chunks], axis=1),
                      np.concatenate([batch[2] for batch in chunks], axis=1),
                      np.concatenate([batch[3] for batch in chunks]),
                      sum(batch[4] for batch in chunks), sum(batch[5] for batch in chunks),
                      time.perf_counter() - start)

    probabilities, labels, fractions = result.probabilities, result.labels, result.fractions
    if (probabilities.shape != (len(plan), len(ids), class_count)
            or probabilities.dtype != np.float32 or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=2), 1, atol=1e-5, rtol=0)
            or labels.shape != (len(plan), len(ids)) or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= class_count))
            or fractions.shape != (len(ids),) or not np.isfinite(fractions).all()
            or np.any((fractions < 0) | (fractions > 1))
            or result.forwards <= 0 or not np.isfinite(result.inference_seconds)
            or not 0 <= result.inference_seconds <= result.wall_seconds):
        raise ValueError(f"Invalid probabilities or auxiliary columns: {path}")
    if not path.exists():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, signature=signature, ids=np.asarray(ids), **vars(result))
        publish(path, buffer.getvalue())
    return result


@dataclass(frozen=True)
class Scores:
    labels: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    disagreement: np.ndarray


def aggregate(bank, plan, recipe):
    indices = [plan.index((recipe.denoise, view)) for view in recipe.views]
    probabilities = bank.probabilities[indices]
    view_labels = bank.labels[indices]
    if recipe.anchor:
        mean = probabilities[0] * 0.5 + probabilities[1:].mean(axis=0) * 0.5
    else:
        mean = probabilities.mean(axis=0)
    labels = view_labels[0] if len(indices) == 1 else mean.argmax(axis=1)
    top_two = np.sort(np.partition(mean, -2, axis=1)[:, -2:], axis=1)
    # 採未加權視角眾數，保留與 V8 相同的一致性定義。
    disagreement = np.array([1 - np.unique(row, return_counts=True)[1].max() / len(row)
                             for row in view_labels.T])
    return Scores(labels, mean[np.arange(len(labels)), labels], top_two[:, 1] - top_two[:, 0], disagreement)


def scores_for(path, examples, names, metadata, class_count, condition, seed, model, batch_size, workers):
    plan = requested(names)

    def batches():
        import torch

        dataset = ViewImages(examples, plan, condition, seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
            prefetch_factor=1 if workers else None, collate_fn=collate_views,
            generator=torch.Generator().manual_seed(seed),
        )
        yield from infer(model, loader, plan)

    bank = cache_bank(path, examples, plan, metadata, class_count, batches)
    return {name: aggregate(bank, plan, RECIPES[name]) for name in names}, bank
