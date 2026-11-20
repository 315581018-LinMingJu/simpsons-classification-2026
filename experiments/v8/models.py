"""白名單權重來源與強骨幹的訓練範圍。"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from v3.storage import file_sha256

RULES_URL = "https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules"
VISION_SOURCE = "https://github.com/pytorch/vision/tree/v0.24.1"
VISION_LICENSE = "https://github.com/pytorch/vision/blob/v0.24.1/LICENSE"


@dataclass(frozen=True)
class WeightSource:
    architecture: str
    enum: str
    filename: str
    sha256: str
    mib: float

    @property
    def url(self):
        return "https://download.pytorch.org/models/" + self.filename


SOURCES = {
    "convnext_tiny": WeightSource("convnext_tiny", "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
                                  "convnext_tiny-983f1562.pth",
                                  "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d", 109.119),
    "vit_b_16": WeightSource("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_V1",
                             "vit_b_16-c867db91.pth",
                             "c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9", 330.285),
}


def head_status(url):
    try:
        with urlopen(Request(url, method="HEAD"), timeout=12) as response:
            return {"http": response.status, "content_length": response.headers.get("Content-Length")}
    except OSError as exc:
        return {"error": str(exc)}


def verify_weight(path, source):
    if file_sha256(path) != source.sha256:
        raise ValueError(f"SHA-256 mismatch for official weight {source.filename}. Use a verified copy.")
    return path


def official_weight(name, directory, local=None):
    source = SOURCES[name]
    if local is not None:
        return verify_weight(local, source)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / source.filename
    if path.exists():
        return verify_weight(path, source)
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, delete=False) as file:
            temporary = Path(file.name)
            digest = hashlib.sha256()
            with urlopen(source.url, timeout=45) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    file.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != source.sha256:
            raise ValueError(f"Downloaded bytes do not match {source.filename}")
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify_weight(path, source)
        return path
    finally:
        if temporary is not None:
            temporary.unlink()


def make_model(name, classes, pretrained=None):
    import torch
    from torchvision import models

    model = {"convnext_tiny": models.convnext_tiny, "vit_b_16": models.vit_b_16}[name](weights=None)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True), strict=True)
    head = model.classifier if name == "convnext_tiny" else model.heads
    head[-1] = torch.nn.Linear(head[-1].in_features, classes)
    return model


def trainable_modules(model, name, mode):
    head = model.classifier if name == "convnext_tiny" else model.heads
    if mode == "full":
        return head, [model]
    if mode == "head":
        return head, [head]
    if name == "convnext_tiny":
        return head, [head, model.features[-1]]
    return head, [head, *list(model.encoder.layers.children())[-2:], model.encoder.ln]


def configure(model, name, mode, lr, decay):
    import torch

    head, modules = trainable_modules(model, name, mode)
    model.requires_grad_(False)
    for module in modules:
        module.requires_grad_(True)
    head_parameters = list(head.parameters())
    head_ids = {id(value) for value in head_parameters}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_parameters, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr / 10})
    return modules, torch.optim.AdamW(groups, weight_decay=decay)


def permit_weights(origin, allowed):
    if origin["kind"] == "imagenet1k" and not allowed:
        raise ValueError("Confirm both competition and assignment rules, then pass --allow-imagenet.")
