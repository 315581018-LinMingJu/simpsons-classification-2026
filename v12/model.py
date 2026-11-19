"""官方 ConvNeXt-Base 權重、單卡預檢與固定解凍範圍。"""

import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import __version__ as pillow_version

from v8.models import WeightSource, verify_weight

SOURCE = WeightSource(
    "convnext_base", "ConvNeXt_Base_Weights.IMAGENET1K_V1", "convnext_base-6075fbad.pth",
    "6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd", 338.064,
)
ORIGIN = {"kind": "imagenet1k", "weight": SOURCE.enum, "url": SOURCE.url, "sha256": SOURCE.sha256}
RATES = {"head": (1e-3, None), "tune": (2e-4, 2e-5)}


def permitted(allowed):
    if not allowed:
        raise SystemExit("確認 Kaggle 與作業允許 ImageNet-1K 後，才使用 --allow-imagenet。")


def weights_path(directory, local=None):
    import torch

    if local is not None:
        return verify_weight(local, SOURCE)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SOURCE.filename
    if path.exists():
        return verify_weight(path, SOURCE)
    with NamedTemporaryFile(dir=directory) as temporary:
        torch.hub.download_url_to_file(SOURCE.url, temporary.name, hash_prefix=SOURCE.sha256)
        verify_weight(Path(temporary.name), SOURCE)
        try:
            os.link(temporary.name, path)
        except FileExistsError:
            verify_weight(path, SOURCE)
    return path


def make_base(classes, weights=None):
    import torch
    from torchvision.models import convnext_base

    model = convnext_base(weights=None)
    if weights is not None:
        model.load_state_dict(torch.load(verify_weight(weights, SOURCE), map_location="cpu", weights_only=True),
                              strict=True)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, classes)
    return model


def configure(model, phase):
    import torch

    head_lr, backbone_lr = RATES[phase]
    model.requires_grad_(False)
    modules = [model.classifier]
    groups = [{"params": list(model.classifier.parameters()), "lr": head_lr}]
    if backbone_lr is not None:
        # 最後兩個解析度階段與其先前的降採樣層。
        backbone = list(model.features.children())[-4:]
        modules.extend(backbone)
        groups.append({"params": [p for layer in backbone for p in layer.parameters()], "lr": backbone_lr})
    for module in modules:
        module.requires_grad_(True)
    return modules, torch.optim.AdamW(groups, weight_decay=1e-4)


def environment():
    import torch
    import torchvision

    info = {"python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "pillow": pillow_version, "numpy": np.__version__, "built_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "devices_visible": torch.cuda.device_count(),
            "convnext_base": hasattr(torchvision.models, "convnext_base")}
    if info["cuda_available"] and info["devices_visible"] == 1:
        info.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                    total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                    bf16=torch.cuda.is_bf16_supported(), cudnn=torch.backends.cudnn.version())
    return info


def gpu_probe():
    import torch
    import torch.nn.functional as F

    env = environment()
    if not env.get("convnext_base") or not env.get("bf16"):
        return {**env, "status": "requires_one_bf16_cuda_gpu_and_torchvision_convnext"}
    try:
        env["free_gib_at_probe"] = torch.cuda.mem_get_info()[0] / 2**30
        inputs = torch.ones((2, 3, 16, 16), device="cuda")
        weight = torch.ones((4, 3, 3, 3), device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.conv2d(inputs, weight).float().mean()
        loss.backward()
        torch.cuda.synchronize()
        okay = bool(torch.isfinite(weight.grad).all())
    except RuntimeError as exc:
        return {**env, "status": "cuda_kernel_failed", "error": str(exc)}
    return {**env, "status": "ready" if okay else "cuda_gradient_not_finite"}
