"""V3 的影像視角、機率彙整與批次推論。"""

import time
from dataclasses import dataclass

import numpy as np

VIEWS = {"baseline": 1, "full_resize": 1, "five_crop": 5, "multiscale": 10}
SCALES = (240, 288)


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float
    view_disagreement: float
    seconds: float


class Views:
    def __init__(self, method, evaluation):
        from torchvision import transforms as T

        self.method = method
        self.evaluation = evaluation
        self.normalize = T.Compose(evaluation.transforms[2:])
        self.full_resize = T.Resize((224, 224))
        self.crop = T.FiveCrop(224)
        self.scales = [T.Resize(scale) for scale in (SCALES if method == "multiscale" else (256,))]

    def __call__(self, image):
        if self.method == "baseline":
            return (self.evaluation(image),)
        if self.method == "full_resize":
            return (self.normalize(self.full_resize(image)),)
        return tuple(self.normalize(crop) for resize in self.scales for crop in self.crop(resize(image)))


def summarize_views(probabilities, view_labels, seconds):
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities from checkpoint inference")
    mean = probabilities.mean(axis=0)
    # 單視角用原始 logits 的 argmax，與 V2 推論完全相同。
    labels = view_labels[0] if len(view_labels) == 1 else mean.argmax(axis=1)
    return [
        Prediction(
            int(label), float(mean[i, label]),
            1 - int(np.unique(view_labels[:, i], return_counts=True)[1].max()) / len(view_labels),
            seconds / len(labels),
        )
        for i, label in enumerate(labels)
    ]


def view_loader(samples, method, evaluation, batch_size, workers):
    import torch

    from classification_v2 import Images

    return torch.utils.data.DataLoader(
        Images(samples, Views(method, evaluation)), batch_size=batch_size,
        num_workers=workers, pin_memory=True, prefetch_factor=1 if workers else None,
    )


def infer_batches(model, samples, method, evaluation, batch_size, workers, *, warmup=True):
    loader = view_loader(samples, method, evaluation, batch_size, workers)
    yield from infer_loader(model, loader, warmup=warmup)


def infer_loader(model, loader, *, warmup=True):
    for ids, probabilities, view_labels, seconds in infer_view_batches(model, loader, warmup=warmup):
        yield list(zip(ids, summarize_views(probabilities, view_labels, seconds), strict=True))


def infer_view_batches(model, loader, *, warmup=True):
    import torch

    with torch.inference_mode():
        if warmup:
            model(torch.zeros((1, 3, 224, 224), device="cuda"))
        for views, ids in loader:
            torch.cuda.synchronize()
            started = time.perf_counter()
            probs, labels = [], []
            for view in views:
                logits = model(view.cuda(non_blocking=True))
                probs.append(logits.softmax(dim=1))
                labels.append(logits.argmax(dim=1))
            probabilities = torch.stack(probs).cpu().numpy()
            view_labels = torch.stack(labels).cpu().numpy()
            seconds = time.perf_counter() - started
            yield ids, probabilities, view_labels, seconds
