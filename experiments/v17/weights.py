"""以相鄰 checkpoint 的模型張量建立固定 50/50 平均。"""

import torch


def epoch_model(path, settings, epoch, row, digest):
    from classification_v14 import read_epoch
    from v3.storage import file_sha256

    saved = read_epoch(path, settings, epoch)
    expected = {k: v for k, v in row.items() if k != "checkpoint_sha256"}
    if saved["training"] != expected or row["checkpoint_sha256"] != digest or file_sha256(path) != digest:
        raise ValueError(f"V14 epoch {epoch} checkpoint or history differs: {path}")
    return saved["model"]


def average(a3, a4, architecture):
    if a3.keys() != architecture.keys() or a4.keys() != architecture.keys():
        raise ValueError("Model tensor names differ from the ConvNeXt-Tiny state dictionary")
    result = {}
    count = {"averaged_float": 0, "copied_nonfloat": 0}
    with torch.no_grad():
        for name, expected in architecture.items():
            a, b = a3[name], a4[name]
            if any(not isinstance(t, torch.Tensor) or t.shape != expected.shape or t.dtype != expected.dtype
                   or t.layout != torch.strided or t.device.type != "cpu" or t.is_complex() or t.is_quantized
                   for t in (a, b)):
                raise ValueError(f"Incompatible model tensor: {name}")
            if a.is_floating_point():
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    raise ValueError(f"Non-finite source tensor: {name}")
                # 先各乘一半再相加，低精度張量在 FP32 中計算。
                dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
                mean = (a.to(dtype).mul(.5) + b.to(dtype).mul(.5)).to(a.dtype)
                if not bool(torch.isfinite(mean).all()):
                    raise ValueError(f"Non-finite averaged tensor: {name}")
                result[name] = mean
                count["averaged_float"] += 1
            else:
                if not torch.equal(a, b):
                    raise ValueError(f"Non-floating model tensor differs between epochs: {name}")
                result[name] = a.clone()
                count["copied_nonfloat"] += 1
    return result, count
