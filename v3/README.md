# V3 多視角推論與測試圖片診斷

適用於 `ml-2026-iaii-classification2`。V3 使用 V2 的 ResNet18 最佳權重做四種獨立推論。測試集診斷另載入 V1 `best.pt`，使用原本的評估前處理取得 V1 標籤與信心。所有權重和舊版 CSV 都以唯讀方式使用。

## RTX 5090 執行順序

沿用實驗室 Python 3.10、PyTorch 2.14.1+cu130、torchvision、NumPy、Pillow 和 Matplotlib。若環境沒有 Matplotlib，只需補裝 Matplotlib，沿用已驗證的 CUDA 套件環境。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
source .venv/bin/activate
python -c 'import torch, torchvision, numpy, PIL, matplotlib; print(torch.__version__, torch.cuda.get_device_name())'

python classification_v3.py baseline-check
python classification_v3.py diagnostic
python classification_v3.py validation
python classification_v3.py predict
```

預設讀取 `data/extracted/`、`runs/v1_baseline/` 與 `runs/v2_primary/`。兩版各需 `best.pt`、`classes.json` 和 `split.csv`。V2 還需原 `run_config.json`，測試階段需 `submission_v2.csv` 做 Baseline 一致性檢查。可用 `--v2-submission` 指定參考 CSV 的其他位置。驗證階段不需讀取 V1 權重或測試圖。

| 階段 | 工作 | RTX 5090 首次執行粗估 |
| --- | --- | --- |
| `baseline-check` | 以 V2 原設定對全量測試圖比對 Baseline，先確認權重和推論流程 | 約 30 秒至 3 分鐘 |
| `diagnostic` | 固定種子抽 128 張測試圖，先驗證推論與報表流程 | 約 15–90 秒 |
| `validation` | 解碼約 9.7 萬張圖排除跨集合完全重複，再對約 1.9 萬張驗證圖比較四種方法 | 約 8–30 分鐘 |
| `predict` | 對 10,791 張測試圖完成推論、全量診斷及候選 CSV | 約 4–18 分鐘 |

這些是估計，尚未在實驗室實測。JPEG 尺寸、CPU 解碼、磁碟與 GPU 負載都會影響耗時。預設 `--batch-size 16 --workers 2`，每次只把一個視角的一批張量移到 GPU。CPU 端每批最多持有 10 個視角，worker 各預取一批。若遇到主機共享記憶體限制，先試 `--workers 0`；若 GPU 記憶體不足，可降至 `--batch-size 8`，並為不同批次大小使用新 `--output`。

## 方法

| 方法 | 視角 | 幾何前處理 |
| --- | ---: | --- |
| `baseline` | 1 | V2 原始短邊 Resize(256) 加 CenterCrop(224)，以原始 logits 的 argmax 取類別 |
| `full_resize` | 1 | 整張影像縮放至 224×224，保留邊緣內容，長寬比可能拉伸 |
| `five_crop` | 5 | 短邊 Resize(256)，擷取中心和四角的 224×224 |
| `multiscale` | 10 | 短邊分別縮放至 240、288，每個尺度各取中心與四角 |

所有方法沿用 V2 的 RGB 讀取及 ImageNet 正規化。多視角方法先對每個視角各自做 Softmax，再等權平均機率；最大平均機率對應的角色為預測。不同裁切可能捕捉到不同角色。依方法分開比較，結果可持平、變差或改善。

## 結果與續跑

- **`baseline-check`** 只跑 V2 原始 Baseline，依範本順序、V2 紀錄的 batch size、相同前處理與 cuDNN 設定推論。將全量核對結果寫到 `runs/v3_tta/baseline_check.json`。即使 CUDA 核心或版本不同，是否一致仍以逐 ID 實際結果為準。
- **`diagnostic`** 將小樣本統計及圖片存到 `runs/v3_tta/diagnostic/`。可用 `--limit` 和 `--seed` 調整抽樣。更改抽樣並保留原報告時，另指定新的 `--output`。
- **`validation`** 在 `runs/v3_tta/` 寫入 `comparison.csv`、`comparison.json` 和去重清單。四種方法使用同一批排除跨集合 RGB 像素與尺寸完全相同的驗證圖，使用原切分的真實驗證標籤。報表列出準確率、相對 Baseline 的百分點差、改判比例、平均信心及耗時。耗時累加每批主機到 GPU 傳輸、前向運算、Softmax 及 GPU 到主機傳輸，排除讀圖、解碼與預熱。去重未涵蓋近似圖或相鄰影格。
- **`predict`** 在 `runs/v3_tta/` 產生全量診斷與 `submission_v3_baseline.csv`、`submission_v3_full_resize.csv`、`submission_v3_five_crop.csv`、`submission_v3_multiscale.csv`。每份 CSV 都依範本的字串 ID 順序寫入 10,791 筆、只有 `id,character`，並驗證 ID 和 50 類映射。範本的角色欄不作為答案。會先通過全量 Baseline 比對，再執行其他方法。有差異時，從 `baseline_check.json` 查差異 ID，並核對權重、參考 CSV 和執行環境。抽樣 Baseline 另存快取，以維持正式比對的原始批次邊界。
- 測試診斷提供各方法所有角色的 `prediction_distribution.csv`、逐圖 `test_diagnostics.csv` 與 `test_summary.json`。拼圖包括 `low_confidence.png`、`prediction_disagreements.png`、`tta_disagreements.png`、`v1_v2_disagreements.png`。每格標示 ID、V1／V2／該方法預測與信心。視角分歧定義為 `1 − 最常見視角 top-1 的票數 / 視角數`，與最終機率平均的類別是不同統計。

成功的批次會逐批寫入 SQLite 快取。本機相同輸入、環境與 batch size 下，`predict` 可沿用 `baseline-check` 的全量 Baseline、`diagnostic` 的其餘推論，被中斷後也可續跑。驗證另有快取，並沿用已算好的去重清單。報表及候選檔內容相同時直接沿用，內容不同時明確停止，可指定如 `--output runs/v3_tta_repeat1` 保存新結果。輸入影像快取依路徑、大小與時間戳追蹤，請使用完整且未變更的原資料。執行同一個輸出目錄時，請讓各階段依序完成。若曾執行舊版 V3 並留下快取，為新版各階段統一指定 `--output runs/v3_tta_verified`，保留舊結果。

## 解讀限制

驗證集與競賽測試集存在已觀察到的表現差距。本地準確率和 TTA 改判比例無法預測 Kaggle Public Score，分布表中某些類別增加也無法在無標籤時判定對錯。信心是模型輸出的機率分數，未經校準。拼圖可用於檢視偏心、邊框、多角色等假設，單憑幾張圖不能確定造成差距的原因。候選檔只會寫到本機。

## 根據結果選擇下一項實驗

1. **先確認全量 Baseline。** 若與既有 V2 Submission 有任何不同，先用 `baseline_check.json` 確認差異範圍，核對 checkpoint、V2 產出紀錄和套件版本。保留已知的 V2 提交結果作為比較基準。
2. **有明確的裁切機制才優先試一個 TTA 候選。** 比對測試集改判比例、角色分布和視角分歧，並在改判圖中找角色被中心裁切移除的案例，也查看不同裁切只剩背景或截到其他角色的反例。若便宜的 Full Resize 與較貴方法有相近支持，優先選 Full Resize。若需要 Kaggle 驗證，只由使用者選擇並提交一個候選。高信心、低分歧與 Clean 準確率都不能單獨當作篩選依據。
3. **TTA 證據不足或分數下降時，先提高本地驗證的鑑別力。** 下一項實驗固定一份帶標籤、無完全重複的驗證子集，加入保留同一角色的偏心、縮小與長寬比變化。在同圖上比較 Baseline 和最便宜的候選，記錄兩方向的改判對錯。這能檢驗裁切假設，但仍無法重現真實測試集的背景和多角色情況。
4. **用困難子集的互補錯誤決定是否集成。** 若 V1 在 V2 失敗的案例上確實救回足夠樣本，再試一次固定等權的 V1／V2 機率平均。僅有 47.44% 的測試改判不能證明互補。若兩版多在相同案例失敗，再考慮以較強預訓練骨幹做短跑次，固定 V2 增強和驗證協定，確認改善後才延長訓練。

可回傳 `baseline_check.json`、`comparison.csv`、`test_summary.json`、`prediction_distribution.csv` 和拼圖觀察摘要。描述找到的正反例即可，無須傳圖、權重或金鑰。

## 合成測試

```bash
python -m unittest discover -s tests -p 'test_classification_v3.py' -v
```

測試涵蓋視角規格、機率平均、去重、快取續跑、候選檔驗證與圖表，使用合成圖片及替代推論，不需要真實權重或 GPU。
