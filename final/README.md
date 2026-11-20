# 最終方法：V20（測試風格增強＋ConvNeXt）與 V21（驗證集挑選融合）

所有指令都從**專案根目錄**執行，資料放在 `data/extracted/`（Kaggle 下載解壓縮後的結構）：

```
data/extracted/train/train/train/<character>/*.jpg
data/extracted/test_images/test_images/<id>.jpg
data/extracted/sample_submission.csv
```

## 1. 方法

### 資料切分
`baseline_split()` 逐行重現課程 `baseline.py`：類別名稱排序、每類檔案排序、同一個 `random.Random(2026)` 依序洗牌、每類 20% 為驗證集。Train 77,547 張、Val 19,382 張。`train` 會輸出 `split.csv` 供核對。

### 測試風格增強（`testify()`）
1. **作業增強 #1（`distort()`）**：水平／垂直翻轉、旋轉／仿射／透視（黑色填補）、亮度對比飽和色相、灰階、反相、色階壓縮、Solarize、銳化、彈性變形、高斯模糊、四種雜訊（Gaussian、Speckle、Poisson、Salt-and-Pepper）。機率比課程範例高，以符合實際測試圖多數經過幾何變形的觀察。
2. **作業增強 #2（`composite()`）**：近黑（三通道皆 <20）或近白（皆 >235）像素視為透明，外加 10–30 px 邊框，貼到隨機裁切的 DTD 紋理照片上。
3. 70% 機率再做一次 JPEG 壓縮（品質 70–95）。

訓練時 85% 的圖片套用上述流程，其餘只做水平翻轉。

### 模型與訓練
| 項目 | 設定 |
|---|---|
| 骨幹 | timm `convnext_base.fb_in22k_ft_in1k` / `convnext_large.fb_in22k_ft_in1k`，完整微調 |
| 輸入 | 整張圖縮放成正方形（224 或 288），ImageNet mean/std |
| 最佳化 | AdamW，lr 1e-4，weight decay 0.05，batch 64 |
| 排程 | 線性 warmup 1 epoch（最多 1,000 step）＋ cosine 到 1% |
| 正則化 | label smoothing 0.1，drop path 0.2，梯度裁切 1.0 |
| 精度 | BF16 autocast，channels_last |
| 推論 | 原圖＋水平翻轉，softmax 機率平均 |
| 模型挑選 | 每 epoch 評估 Clean 與 Test-style 驗證集，保存 Test-style 最高者為 `best.pt` |

### 融合（V21）
三個成員各自計算驗證集機率，比較所有組合後選 Test-style 驗證正確率最高的組合，再對測試機率等權平均。

## 2. 結果（RTX 5090 單卡）

| 成員 | 骨幹＠解析度 | Epochs | 最佳 epoch | Clean | Test-style | 時間（約） | VRAM |
|---|---|---|---|---|---|---|---|
| main | Base@224 | 10 | 9 | 99.917% | 98.824% | 32.3 分鐘（訓練 25.9） | 10.7 GiB |
| b288 | Base@288 | 12 | 11 | 99.938% | 99.278% | 每 epoch 訓練 4.6 分鐘 | 15.6 GiB |
| large | Large@224 | 10 | 10 | 99.938% | 99.200% | 每 epoch 訓練 3.7 分鐘 | 16.7 GiB |

| 融合組合 | Clean | Test-style | Kaggle Public |
|---|---|---|---|
| main（V20 單模型） | 99.917% | 98.813%* | 0.98517 |
| main＋b288＋large | 99.954% | 99.293% | 未提交 |
| **b288＋large（V21，採用）** | 99.943% | **99.345%** | **0.98887** |

\* `valprobs` 以 softmax 機率平均做 TTA；訓練中的驗證以 logits 相加，數值略有差異（98.824%）。

精確時間記錄在各 run 的 `history.json`（`train_seconds`、`epoch_seconds`）。

每個 run 的 `confusion_val_clean.csv`、`confusion_val_testlike.csv` 是 50×50 混淆矩陣（列為真實、欄為預測）。

## 3. 重現指令

```bash
# 環境（不更動既有 torch）
pip install --no-deps timm==1.0.31 && pip install huggingface_hub safetensors pyyaml

# 背景照片：DTD（約 625 MB）
python final/classification_v20.py backgrounds
BG=$(find data/backgrounds -type d -name images | head -n 1)

# （選用）產生 36 張合成圖檢查增強效果 → runs/v20/preview.jpg
python final/classification_v20.py preview --backgrounds "$BG"

# 三個成員（各自訓練 → 預測 → 驗證集機率），可分別放在不同 GPU
BG="$BG" bash final/run_member.sh 0 main  convnext_base.fb_in22k_ft_in1k  224 10 64
BG="$BG" bash final/run_member.sh 1 b288  convnext_base.fb_in22k_ft_in1k  288 12 64
BG="$BG" bash final/run_member.sh 2 large convnext_large.fb_in22k_ft_in1k 224 10 64

# 依驗證集比較融合組合，輸出 submission_v21.csv
python final/classification_v21.py ensemble runs/v20/main runs/v20/b288 runs/v20/large --output runs/v21/all3
python final/classification_v21.py ensemble runs/v20/b288 runs/v20/large --output runs/v21/b288_large
```

輸出位置：
- `runs/v20/<名稱>/`：`config.json`、`split.csv`、`history.json`、`best.pt`、`last.pt`、混淆矩陣、`val_probabilities.npz`
- `runs/v20/<名稱>/candidate_epoch_NN/`：`submission_v20.csv`、`test_probabilities.npz`、`summary.json`
- `runs/v21/<組合>/ensemble-N/`：`submission_v21.csv`、`summary.json`（各成員與融合後的驗證正確率）

`train` 中斷後重新執行同一指令，會從 `last.pt` 接著跑；已存在的 candidate 目錄不會被覆蓋。

## 4. 測試

```bash
cd final && python -m pytest -q tests
```
CPU 測試涵蓋：baseline 切分與參考演算法一致、黑白像素遮罩與邊框、程序化背景、融合輸出格式（共 4 項）。GPU 階段在研究室 Server 以相同檔案（SHA-256：v20 `b539d567…655f`、v21 `b5c2e09a…0ca4`）實際執行。
