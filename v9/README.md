# V9｜ConvNeXt-Tiny 免訓練推論短跑

V8 的 Public Score 是 0.75563，Screen Hard 是 90.26%。本輪沿用該 checkpoint，先檢查 ResNet18 時期選定的裁切法在 ConvNeXt 上的效果。Stage 分開，只有 Screen 入選的一個配方會進 Confirm。所有新產物保存在 `runs/v9/`。沿用 Server 既有的 Python、PyTorch、torchvision 和 CUDA。

## 同步與 CPU 前處理稽核

```bash
cd /usr1/Aa315581018/simpsons-classification-2026 && git pull --ff-only origin main && .venv/bin/python -u classification_v9.py audit
```

預設 checkpoint 為 `runs/v8_full/models/convnext_full/best.pt`。程式在 `runs/v8_full/candidates/*/` 找與該 checkpoint 雜湊吻合的唯一 Multi-scale + Denoise 封存。若你的 V8 路徑不同，在每個階段傳入相同的 `--v8-run`、`--checkpoint` 和 `--v8-submission`。指定的 Submission 須附同目錄下 V8 產生的 `settings.json`。程式會核對 checkpoint、Submission、模板順序、Test 檔案快照與原切分。GPU 階段沿用 V8 官方權重規則的 `--allow-imagenet`（確認競賽與作業規則後使用）。

V8 Train 和 Test 皆以 Pillow 解碼並轉成 RGB，再使用 ImageNet mean/std。Train 預設 `robust` 包含 V2 隨機縮放裁切、幾何變換及灰階，也包含合成全景的 224×224 縮放、亮度／對比／色彩變化。去黑框只發生在部分合成分支，採用連通外圍近黑像素估計。V8 Test 直接對原 RGB 圖套用固定去顆粒條件後縮放和裁切。Test 原本的明亮背景、黑框、品質與位置都會影響模型可見的內容。

`preprocessing.json` 比較完整既有 Train／Test 稽核，量測裁切保留的畫面比例，並對每類一張 Screen 合成圖估計主圖塊 alpha 保留量。正方形縮放到短邊 240 和 288，各自的 224 裁切面積約佔 87% 和 60%。矩形保留的畫面比例會更低。真實 Test 缺少角色遮罩，畫面保留率無法換算成角色保留率。完整縮放保留整個畫面，但會改變非正方形影像的長寬比。

## RTX 5090 最短執行順序

先用 `nvidia-smi` 確認一張可使用的 GPU，再輸入該卡編號。透過 SSH 操作時可使用 tmux。

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' V9_GPU
export CUDA_VISIBLE_DEVICES="$V9_GPU"

.venv/bin/python -u classification_v9.py screen --allow-imagenet
.venv/bin/python -u classification_v9.py diagnostic --allow-imagenet
```

Screen 使用 V8 的原始去重標註樣本及同一組 Clean、場景、顆粒、裁切、多角色與反色。數值報告含各條件準確率、配對勝負次數、改判率及 Hard 差距。所有方法在同批影像上重用同一視角機率，未觸發去顆粒時共用該列推論，觸發門檻沿用既有 0.003。

| 配方 | 視角與目的 |
| --- | --- |
| `v8_multiscale_denoise` | 240、288 各五裁切，平均 10 個 Softmax。既有 V8 對照 |
| `multiscale_raw` | 同上，量測 selective denoise 的影響 |
| `full_raw` / `full_denoise` | 全圖 224×224，分開測原圖及去顆粒 |
| `center256_denoise` | 短邊 256，中心裁切 224 |
| `five240_denoise` / `five288_denoise` | 各單一尺度的五裁切，檢驗裁切強度 |
| `two_centers_denoise` | 240 和 288 的中心裁切，量測移除四角的影響 |
| `full_anchor240_denoise` | 50% 全圖機率 + 50% 短邊 240 的五裁切平均，保留全圖訊號 |

`diagnostic` 預設固定種子抽約 768 張真實 Test，按 V8 封存的完整批次抽樣以維持重跑批次形狀。會依視覺尾端、Train 覆蓋、外圈黑色比例及去顆粒觸發分組，列出信心、前兩名差值、視角分歧與改判率。視角分歧是偏離該方法的視角眾數比例；單視角沒有可比較的其他視角，因此分歧記為 0。檔案也分開記錄 V8 當次重跑及歷史 CSV 的標籤。Test 信心、視角分歧與改判只能顯示模型對輸入有多敏感。各群報告應同時看樣本數。

Screen 要求 Hard 較 V8 高至少 0.75 pp、Clean 降幅至多 0.5 pp、每項主要 Hard 降幅至多 5 pp。達標者依 Hard 增益選一個，平手時優先少視角，結果保存為 `frozen.json`。有入選者再做一次 Confirm：

```bash
.venv/bin/python -u classification_v9.py confirm --allow-imagenet
```

Confirm 只跑 V8 對照與已鎖定配方。要求配對 Hard 提升至少 1 pp、按原圖在各類內重抽 95% CI 下界大於 0，並符合相同 Clean／各項 Hard 降幅限制。V8 的 Confirm 如已用於過往決策，V9 結果也會受先前使用的影響。合成分布與真實 Test 仍有差距。Confirm 未通過時，保留已提交的 V8。

若 Confirm 通過，且抽樣 V8 重跑精確重現封存 CSV，才執行以下指令。只在全 Test 推論選定的一種配方，候選與原 CSV 全數相同時只保存報告。重播不一致時可查 `diagnostic/v8_replay_mismatches.csv` 的 ID 與 margin，先確認原 batch、環境及來源。

```bash
.venv/bin/python -u classification_v9.py predict --allow-imagenet
```

候選輸出為 `runs/v9/candidate/submission_v9.csv`，含原模板順序的 10,791 筆、合法類別及 `id,character` 欄位。Kaggle 正式提交由你決定。各條件機率快取可續用，來源、設定、批次或程式版本不同時拒絕重用。GPU 記憶體不足時，Screen 可用新的 `--output runs/v9/batch8 --batch-size 8 --workers 2`；Test 診斷仍用封存批次大小（省共享記憶體可降 `--workers`）。已確認的配方與 Confirm 結果應保留供回溯。

## 預估與最少回報

以 batch 16、一張 RTX 5090 的 FP32 推論估算，實際受共享 I/O、CPU 及觸發比例影響。Screen 與 Test 診斷約需 4–8 GiB GPU 記憶體，並使用 DataLoader 的主機共享記憶體。資源數據待 Server 實測。

| 階段 | 時間規劃 | 只需貼回 |
| --- | ---: | --- |
| CPU `audit` | 1–3 分鐘 | 三個尺度的 Test 中位畫面保留率 |
| `screen` | 3–12 分鐘 | 終端最後 JSON 的 `decision`、`recipe`、`reference_hard`、`paired_hard` |
| `diagnostic` | 2–8 分鐘 | 樣本數、觸發數、`v8_replay_mismatches`，以及 `by_group.csv` 中 V8 與入選配方的 `upper_tail` 兩列 |
| `confirm` | 2–10 分鐘 | 終端最後 JSON 的 `decision`、`paired_hard` |
| `predict` | 4–25 分鐘，依入選視角數 | `recipe`、`changed_vs_v8`、`decision` |

若免訓練配方無明確增益，下一個值得投入的訓練實驗是以既有 ConvNeXt 短微調一次，固定加入全圖、偏心小角色與亮而有紋理的場景，並控制角色可見面積。使用競賽 Train 標籤與既有 Test 稽核統計，固定同樣步數和 checkpoint 對照 V8。先測場景偏移與容易被裁掉的條件有無配對增益，再看其他條件的退步。這個假設比換更大的骨幹或加 Epoch 更直接對應目前觀察到的分布偏移。
