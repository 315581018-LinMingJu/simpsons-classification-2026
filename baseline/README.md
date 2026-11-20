# 官方 Baseline（ResNet-18）

`baseline.py` 原封不動取自本課程 Kaggle 競賽頁面「完整 baseline.py」，作為報告的基準方法。
主辦方公布的 Public 實測分數為 0.58418。

本專案在研究室 Server（RTX 5090）重新執行，只調整 DataLoader worker 數以加速，其餘使用預設值
（ImageNet ResNet-18、10 epochs、seed 2026、每類 20% 驗證、patience 3）：

```bash
python baseline/baseline.py train --train-dir data/extracted/train --output runs/report/baseline_resnet18 --workers 16
```

`final/classification_v20.py` 的 `baseline_split()` 逐行重現這裡的切分，因此所有方法使用同一組驗證集。
