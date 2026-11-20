# V8 強骨幹單卡篩選

V7 的 Public Score 為 0.68211，距離排行榜第 15 名約 0.98300 差 30.089 個百分點。V8 優先檢驗較強的骨幹能否在有標籤的分布偏移下超過 ResNet18，並以最快能得到的實驗結果決定是否投入完整訓練。

## 資料與規則

全量稽核顯示 Train 全為 224×224 JPEG，Test 全為副檔名 `.jpg` 的 PNG，尺寸 244–284。外圈非多數黑色的比率由 Train 的 1.18% 升至 Test 的 94.19%。兩類明亮、高頻細節多且少黑邊的視覺組合，在 Train 只有 97 張，在 Test 有 5,847 張。這些結果支持分別檢驗背景、角色位置與顆粒的敏感度。生成流程細節、是否使用了黑框捷徑，以及這些變化造成多少分類錯誤，仍待驗證。Pillow 以實際格式解碼成 RGB。

Codex 的網路代理拒絕 Kaggle Rules 頁面，GitHub 專案也沒有可核對的作業規則。即使 Server 的 HTTP 探測成功，仍須閱讀內容。只有在確認競賽及作業均允許 ImageNet 預訓練後，才在訓練和載入該權重衍生的 V8 checkpoint 時加上 `--allow-imagenet`。[競賽 Rules](https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules)

| 順序 | 骨幹 | 選擇理由 | 官方權重與資料 |
| --- | --- | --- | --- |
| 1 | ConvNeXt-Tiny | 約 2,860 萬原始參數與 4.46 G ops，優先檢驗多階段卷積特徵能否在小角色與複雜背景下保留輪廓 | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`，ImageNet-1K，109.1 MiB。[torchvision 實作及權重資訊](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/convnext.py) |
| 2 | ViT-B/16 | 約 8,657 萬原始參數與 17.56 G ops，檢驗全圖注意力對偏心及多角色畫面是否有幫助 | `ViT_B_16_Weights.IMAGENET1K_V1`，ImageNet-1K，330.3 MiB。[torchvision 實作及權重資訊](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/vision_transformer.py) |

這兩份官方檔案已在 Codex 完整串流並核對 SHA-256。Server 每次載入會核對完整 SHA-256。torchvision 程式碼採 BSD-3-Clause，權重及訓練資料的使用須遵守各自條款及競賽規則。本輪把預訓練來源限制在官方 ImageNet-1K，DINOv2 的預訓練資料涵蓋 LVD-142M，需另外核對競賽授權範圍。[torchvision LICENSE](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE) · [ImageNet](https://www.image-net.org/) · [DINOv2](https://arxiv.org/abs/2304.07193)

## Phase A｜先在 CPU 準備資料

```bash
cd /usr1/Aa315581018/simpsons-classification-2026 && git pull --ff-only origin main && .venv/bin/python -u classification_v8.py prepare
```

預檢使用既有 V1/V2 的同份 `split.csv`、V5 的去重證據與 `runs/distribution_audit/summary.json`。固定種子後每類抽 160 張 Train，共 8,000 張；從原 Val 排除已知跨集合完全重複、驗證集內像素重複以及 V5/V6 曾用的子集與其完全相同圖片，再每類抽 16 張，分給 `screen` 和 `confirm` 各 400 張。原始驗證檔保留；可另外用 `validate --fold original` 跑原 19,382 張的 Clean，比較時須記得原切分的重複影像。相似影格與接近重複仍可能存在。

每張抽樣 Val 圖建立 Clean 與五種固定的困難條件。場景包含偏心縮小、旋轉與透視，之後分別測局部顆粒、裁切、較小的異類角色與反色。異類角色來自原 Train。所有配置使用同一組底圖、合成種子及干擾參數。Hard 分數依序用 30%、25%、20%、20%、5% 加權。已固定的權重只用來比較本地壓力測試，並未估計真實 Test 類型比例。真實 Test 的解析度與亮度等無標籤稽核摘要會參與資料設計，沒有擷取 Test 像素來訓練。Test 沒有真實標籤，照片背景與多角色的指定標籤仍是合成驗證的主要限制。

輸出 `runs/v8_strong_model/benchmark/prepare.json`、`train.csv`、`validation.csv`。回傳終端列出的 **train、screen、confirm 張數**及權重端點狀態即可。若找不到既有產物，先核對上述預設路徑，或傳入 `--v1-run`、`--v2-run`、`--v5-run`、`--audit` 指向已完成的本機產物。

## Phase B｜單卡短跑

先查資源，依研究室規範確認一張可使用的卡。以下輸入該卡編號，命令會只讓程式看到該卡。tmux 可選，遠端執行建議使用。PyTorch、torchvision 和 CUDA 沿用既有 `.venv`。

```bash
nvidia-smi
read -r -p '已確認可使用的 GPU 編號: ' V8_GPU
export CUDA_VISIBLE_DEVICES="$V8_GPU"
```

**僅在確認 ImageNet-1K 權重合規後**執行 ConvNeXt-Tiny。凍結骨幹一個 epoch，啟用 BF16 AMP、每批 16、累積 4（有效批量 64）與 4 個 DataLoader workers。驗證預設為 224×224 全圖單視角 FP32。首次驗證會用原 V6 權重、固定的 V7 去顆粒閾值與原十視角 Multi-scale 重建 V7 配方，後續同一個 fold 可重用 V7 結果。此對照是有標籤的本地重跑。

```bash
.venv/bin/python -u classification_v8.py train --tag convnext_head \
  --architecture convnext_tiny --init imagenet --allow-imagenet
.venv/bin/python -u classification_v8.py validate --tag convnext_head --allow-imagenet \
  --checkpoint runs/v8_strong_model/models/convnext_head/best.pt
```

若 ConvNeXt 尚無明確優勢，而且 GPU 時間足夠，使用相同資料與策略短跑 ViT-B/16。

```bash
.venv/bin/python -u classification_v8.py train --tag vit_head \
  --architecture vit_b_16 --init imagenet --allow-imagenet
.venv/bin/python -u classification_v8.py validate --tag vit_head --allow-imagenet \
  --checkpoint runs/v8_strong_model/models/vit_head/best.pt
```

**篩選規則。** Screen 的 Hard 距 V7 最多 3 pp、Clean 最多退步 2 pp，才值得投入一次末段微調。只選 Screen 最好的可行骨幹。以下以 ConvNeXt 為例，ViT 則填入 `vit_b_16` 及 `vit_head` 權重。末段微調採分類頭學習率 0.0001、骨幹可訓練部分為其十分之一；ConvNeXt 解凍最後 stage，ViT 解凍最後兩個 Encoder block 與尾端 LayerNorm。

```bash
.venv/bin/python -u classification_v8.py train --tag convnext_partial \
  --architecture convnext_tiny --mode partial --lr 0.0001 --allow-imagenet \
  --from-checkpoint runs/v8_strong_model/models/convnext_head/best.pt
.venv/bin/python -u classification_v8.py validate --tag convnext_partial --allow-imagenet \
  --checkpoint runs/v8_strong_model/models/convnext_partial/best.pt
```

可在篩選後只做一個配對消融。預設 `robust` 約 45% 使用 V2 增強，約 55% 使用去外圍黑底、縮放、位移、透視、稽核校準的程序背景、亮度／對比／色彩擾動和遮擋；其中少量使用較小的第二角色、JPEG 劣化或局部顆粒。`context` 省略後三種，`basic` 只用 V2 增強。要檢查額外干擾，複製同一骨幹的 head 訓練命令，換新 tag 並加 `--augmentation context`；要檢查整體場景增強，改用 `--augmentation basic`。驗證加 `--compare-tag convnext_head` 可輸出同圖配對差值及各困難條件變化。各條件仍有相關性，局部消融的方針以時間充足時的單次對照為限。

完成 Screen 選擇後，對選好的 checkpoint 跑一次 `validate --fold confirm`。以下示範以 ConvNeXt 末段微調通過 Screen 為例。

```bash
.venv/bin/python -u classification_v8.py validate --fold confirm --tag convnext_partial --allow-imagenet \
  --checkpoint runs/v8_strong_model/models/convnext_partial/best.pt
```

**進入 Phase C 的條件。** Confirm 的 Hard 比 V7 至少高 2 pp，按圖配對、類別內重抽的 95% CI 下界大於 0；Clean 降幅最多 1 pp，各主要困難條件降幅最多 8 pp（低權重的罕見反色另列）。若未達門檻，保留 V7，避免把這批 Confirm 反覆用於選參數。這是投入更多 GPU 的依據，Kaggle 分數仍需由正式提交驗證。

各階段短回報只需貼 `comparison.json` 的 `v8_clean`、`v7_clean`、`hard_effect`、`accuracy_gain_pp`、判定及 `peak_reserved_gib`。完整的各類別準確率、參數量、checkpoint 位元組數及各條件推論時間在 `reports/<fold>/<tag>/summary.json`。訓練曲線和時間線在 `models/<tag>/history.json`。

### 恢復、OOM 與權重備案

- 每個階段保留獨立的 `best.pt` 與 `last.pt`。中斷後回到最後完成的 epoch。以同一 `--tag`、架構、階段、批量與學習率加 `--resume`；移除 `--init imagenet`、`--weights-file`、`--from-checkpoint`，保留合規旗標。 `--epochs` 代表總 epoch 數。
- OOM 時改用較小的 `--batch-size 8 --accum 8`，或 `4/16`。已有 checkpoint 時開新 tag，以 `--from-checkpoint .../last.pt` 延續權重並重建 optimizer；若第一個 epoch 就失敗，使用新 tag 從相同初始化重跑。CPU/I/O 記憶體吃緊可降至 `--workers 2`。
- 已核准使用 ImageNet 但 Server 下載失敗時，傳入從官方來源取得的本機 `--weights-file /path/to/official.pth`，仍校驗完整 SHA-256。權重以原子方式保存於 `runs/v8_strong_model/weights/`。若規則無法確認或檔案取不到，最快的合規備案是保留 V7。需要試驗無外部權重時，同一支程式可用 `train --tag convnext_scratch --architecture convnext_tiny --init scratch --mode full --lr 0.001`，從隨機初始化用競賽 Train 短跑，這個結果應獨立看待。

## Phase C｜通過門檻後才投入

先用另一個 `--output runs/v8_full --train-per-class 0` 執行 `prepare`，再以選好的 checkpoint 作 `--from-checkpoint`，在原 Train 全量微調。沿用固定 seed，先用 `--mode partial`、`--lr 0.0001`、`--epochs 2` 與 batch 16／accum 4。用另一個 tag 的 `validate --method multiscale` 評估 TTA，固定 TTA 後才另試 `--denoise`，用 `--compare-tag` 比較同圖結果。由於 Confirm 已用於 Phase C 的進入判定，後續用於觀察泛化的結果應標明這個限制。

最好推論配方確定後，用獨立的 `predict` 讀取該 checkpoint。範例如下，依實際結果加入 `--method multiscale` 或 `--denoise`。

```bash
.venv/bin/python -u classification_v8.py predict --output runs/v8_full --tag candidate --allow-imagenet \
  --checkpoint runs/v8_full/models/convnext_full/best.pt
```

候選 CSV 檢查 10,791 個唯一 ID、原模板順序及 50 個合法名稱，保存在 `runs/v8_full/candidates/candidate/submission_v8.csv`。各 V8 tag 與各驗證 fold 有獨立位置。程式沒有 Kaggle 提交步驟。

## 時間與資源預估

以下是規劃值，依實驗室共享 I/O、CPU 併發與實際 GPU 利用率而變。CUDA 實測請以 `history.json` 和 `summary.json` 為準，VRAM 取 PyTorch peak reserved。Phase B 使用一張已確認可用的 RTX 5090。

| 工作 | 預估耗時 | VRAM 規劃 |
| --- | --- | --- |
| A CPU 預檢 | 1–5 分鐘 | 0 |
| V7 一個 fold 對照 | 3–12 分鐘，重複驗證可重用 | 3–7 GiB |
| ConvNeXt 一個 head epoch 加 Screen | 8–25 分鐘 | 4–8 GiB |
| ViT 一個 head epoch 加 Screen | 12–35 分鐘 | 5–12 GiB |
| 入選者一個 partial epoch 加 Screen | 12–40 分鐘 | 8–16 GiB |
| C 全量 partial 微調 | 30–100 分鐘／epoch | 8–16 GiB |

預留約 5 GiB 給兩種官方權重、Phase B checkpoint、報告及寫入臨時檔。使用單卡逐階段確認；即使更大的批次放得下，也應優先完成第一個可比較的 ConvNeXt 結果。
