# V4 幾何分布診斷與困難驗證

## 為什麼先做這個實驗

- V1 和 V2 都使用 ImageNet 預訓練 ResNet18、相同切分及中心裁切。V2 加入仿射、灰階與局部遮擋等增強，Public Score 從 0.51961 到 0.58541。V3 沿用 V2 checkpoint，Multi-scale 提高到 0.59715。這些增幅無法單獨歸因於某一種增強或裁切。
- V3 本地準確率 99.5644%，與 Public Score 相差約 39.85 個百分點。去除跨集合完全相同的圖片後仍接近 99%，表示完全去重不足以建立有鑑別力的驗證。較容易的畫面、同一來源的近似影格、資料選樣或標註規則差異都是待驗證假設。
- 測試集的 Full Resize、Five Crop、Multi-scale 改判率分別為 23.99%、12.99%、15.12%，顯示測試預測對取景更敏感。偏心、小角色、背景、外框、透視或多角色仍待確認。優先取得真實影像尺寸分布、各比例組的改判率、盲看拼圖的正反例，以及帶標籤的同圖救回與退步數。

| 方向 | 效益假設與主要風險 | 資源與先後 |
| --- | --- | --- |
| 配對的困難驗證 | 檢驗 Multi-scale 的收益是否隨取景惡化而增加，以及保留全圖是否已足夠。合成畫布與真實場景仍有差異。 | **V4 優先**，現有權重，CPU 診斷加單 GPU 推論 |
| V1／V2 集成 | 若兩版在帶標籤困難切片上互相救回，機率平均可能有益。V1 灰階結果較弱，測試集改判無法證明互補。 | 現有權重，約增加一個模型的推論，待帶標籤結果支持 |
| 角色定位或區域聚合 | 可減少背景與裁切，多角色畫面也可能截到錯誤角色。通用偵測器未必能定位動畫角色。 | 既有模型可先做區域評估。偵測訓練需要框標註與額外工時 |
| 更強骨幹或針對性增強 | ConvNeXt-Tiny 或其他 ImageNet 模型可能改善特徵，乾淨切分仍可能無法區分。若 V4 顯示保留視野有益，可先只改幾何相關增強。 | 需重新訓練。等困難驗證建立後再投資單 GPU 訓練 |

## V4 設計與判定

**假設**是當來源圖塊偏離中心，Multi-scale 相對 V2 CenterCrop 的提升大於來源圖塊置中時。沿用原 V1／V2 切分與類別，優先取用影像簽章吻合的 V3 去重清單。沒有可用清單時，才在 V4 目錄重新計算 RGB 完全重複。固定 seed 2026，每類抽至多 24 張驗證圖。

同一張圖跑 Clean、置中縮小、同尺寸偏移三個條件。置中與偏移使用相同灰色背景、縮小幅度與畫布，保持來源圖的長寬比。畫布比例從真實測試圖的第 5、25、50、75、95 百分位等機率選取，再限制於 2:3 到 3:2。來源圖塊以畫布寬高的 54–72% 為適配上限，保留整張來源圖。偏移把圖塊位置移到各軸可平移範圍的 4% 或 96%。`validation_geometry.csv` 記錄原中心裁切保留的來源圖塊面積，可檢查干擾強度，並不代表角色框面積。

三個條件皆使用同一個 V2 checkpoint，分別跑 V2 Baseline、V3 Full Resize 及 V3 Multi-scale。Full Resize 是單視角全圖對照。主要結果是 `(Multi-scale − Baseline)偏移 − (Multi-scale − Baseline)置中` 的百分點。類別等權，類別內對同圖配對重抽 2,000 次，報告雙尾 95% bootstrap 區間。

- 支持假設需偏移組 Multi-scale 相對 Baseline 的改善和主要差值的區間下限都高於 0，且 Clean 點估計沒有下降超過 0.5 個百分點。也應確認報告中的偏移保留率確實下降。
- 偏移改善或主差值的區間上限不大於 0 時，資料不支持此假設。其餘結果列為未定。若 Full Resize 有近似的改善，一併比較配對錯誤和推論成本。
- 合成結果用於篩選機制，無法估計 Kaggle Private 或 Public Score。多角色、透視與複雜背景仍需看真實圖片，近似重複也可能讓區間過窄。若沒有機制支持，先核對拼圖與帶標籤的錯誤分組，再決定下一個模型實驗。

## RTX 5090 伺服器

沿用原 Python、PyTorch、torchvision、Pillow、NumPy、Matplotlib。資料與報告使用本機既有路徑，所有新輸出在 `runs/v4_geometry/`。若 V3 在其他目錄，兩階段都加相同的 `--v3-run runs/...`。若要另跑一組參數，兩階段都加相同的新 `--output runs/v4_...`。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
source .venv/bin/activate

python -u classification_v4.py diagnostic
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
read -r -p "輸入已確認允許使用的一張 GPU index：" gpu_index
CUDA_VISIBLE_DEVICES="$gpu_index" python -u classification_v4.py benchmark
```

- CPU 診斷掃描全部 10,791 張測試圖的尺寸，並各抽樣最多 384 張訓練、驗證、測試圖，粗估 **1–5 分鐘**。
- 困難驗證預設 batch size 16、workers 2、最多 1,200 張標註圖，每張共 36 個視角，預估單張 RTX 5090 **3–12 分鐘、約 2–6 GiB GPU 記憶體**。若需重新掃描約 9.7 萬張圖片去重，另預留 **3–15 分鐘** CPU 與磁碟 I/O。尚未實測估計。程式依序將一個視角的 batch 送入 GPU。共享記憶體不足可用 `--workers 0`，VRAM 不足可用 `--batch-size 8`。
- 已建立的 benchmark 目錄會保留。再次執行請用新輸出目錄。V4 只做推論與診斷，不產生候選 Submission。

## 結果

`diagnostic/diagnostic.json` 有尺寸、長寬比與抽樣影像訊號摘要。若 V3 完整報表的測試圖片簽章與 V2 權重雜湊吻合，會加入 `v3_changes_by_aspect.csv`。色彩、邊界亮度及 EXIF 朝向是影像訊號。`split_montage.png` 對照 Train／Val／Test，`test_cases.png` 包含隨機、比例極端、低信心與改判案例。可另存 `test_review.csv` 的副本，以 `yes/no/uncertain` 紀錄偏心、多角色、外框或背景等觀察。各組來源有不同篩選條件，請分開解讀。

`benchmark/benchmark.json` 提供主要判定、區間、每類樣本數與耗時。`scores.csv` 提供各條件、方法的 macro／micro 準確率及救回和退步數。`validation_geometry.csv`、`paired_predictions.csv` 與 `paired_examples.png` 可查單圖失敗模式。

下一步只需回傳 **`diagnostic/diagnostic.json` 與 `benchmark/benchmark.json`**。若有 `v3_changes_by_aspect.csv` 或肉眼看到的正反例，可一併摘要。無須傳權重、個別圖片或金鑰。

```bash
python -m unittest discover -s tests -p 'test_classification_v4.py' -v
```
