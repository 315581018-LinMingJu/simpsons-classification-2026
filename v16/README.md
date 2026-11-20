# V16 機率融合

V16 使用 V14-A Epoch 3 和 V14-B Epoch 4 的 ConvNeXt-Tiny checkpoint。每個模型沿用 V14 的 RGB 解碼、選擇性去顆粒、短邊 240／288 的 FiveCrop 和原本的 ImageNet Normalize。十個視角各自算 FP32 Softmax，再取算術平均。最後用選定的權重融合兩個模型的 50 類機率。

## 輸入與執行

沿用研究室原有的 Python 3.10、PyTorch、torchvision 與 CUDA 環境。保留原專案路徑及原始資料時間戳，V14 來源指紋會檢查這些資訊。程式只讀取既有競賽資料和既有 checkpoint，沿用 V14 的 ImageNet-1K 使用確認。

| 模型 | Checkpoint | 原 V14 候選目錄 |
| --- | --- | --- |
| A | `runs/v14_continuation/a/epoch_03.pt` | `runs/v14_continuation/a/candidate_epoch_03/` |
| B | `runs/v14_continuation/b/epoch_04.pt` | `runs/v14_continuation/b/candidate_epoch_04/` |

候選目錄各需原始 `summary.json` 和 `submission.csv`。如果移動了候選目錄，在三個資料階段都指定同樣的 `--archive-a` 或 `--archive-b`。保留 V14 的 `prepare.json`、兩組的 `settings.json` 和 `history.json`，以及 V8 Prepare 與部署相關檔案。

在專案根目錄使用原有虛擬環境。

```bash
.venv/bin/python -u classification_v16.py prepare
nvidia-smi
# 把 0 換成已確認可用且有權使用的 GPU 編號。
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -u classification_v16.py probe
.venv/bin/python -u classification_v16.py validate --allow-imagenet
.venv/bin/python -u classification_v16.py predict --allow-imagenet
```

`prepare` 使用 CPU 核對 V14 設定、固定 Screen、Test 資料指紋、類別順序、checkpoint SHA-256 與兩份封存 CSV 的 SHA-256 和 ID 順序。它列出 A／B 的 Test 快取狀態、待算的視角數和從 V14 `summary.json` 估出的剩餘秒數。`probe` 只執行一張單卡的 FP32 ConvNeXt-Tiny 測試。Test 使用兩模型原提交各自記錄的 batch size，Screen 使用 V14 Prepare 的 batch size。若系統 RAM 或共享記憶體吃緊，可以先降低 `--workers`，維持原 batch size。

兩個模型逐一上卡。單張 RTX 5090 的規劃值是 Screen 約 2～10 分鐘，Test 兩組都缺快取約 10～40 分鐘，批次 4～32 的 VRAM 預留約 4～8 GiB。這些是部署前估計，優先參考 Prepare 顯示的 V14 歷史秒數及實際 ETA。兩組都缺快取時，Screen 有 24,000 個視角、Test 有 215,820 個視角。

## 選擇權重

原 V14 Screen 每類 4 張，共 200 張，沿用 Clean 及原 5 種 Hard。兩份 checkpoint 曾使用這個 Screen 評估。配對差距和 95% CI 是在此 Screen 上按類別重抽 2,000 次的結果，缺少獨立驗證及 Test 準確率資訊。Hard 彙總使用原 V8／V14 權重，救回及退步彙總計數以圖片 × 條件為單位。

固定比較 A、B 與 A 權重 90%、80%、70%、50% 的四種融合。預設 80/20。其他比例要同時滿足下列條件才會取代 80/20。

- 加權 Hard 配對提升至少 0.75 個百分點，95% CI 下界大於 0。
- Clean 降幅至多 0.50 個百分點。
- 每種 Hard 條件降幅至多 4.0 個百分點。

多個比例達標時選加權 Hard 提升最高的一組，完全同分時選 A 權重較高的一組。選擇結果和原始配對數值寫入 `runs/v16/validation.json`，Test 階段會從快取重算並檢查選擇結果。

## 中斷與產物

可在 `tmux new -s v16` 內執行。按 `Ctrl-b d` 離開視窗，再用 `tmux attach -t v16` 接回。進度條在批次的十視角全部完成後更新，也會顯示累計圖片數、速度推估的 ETA。每個 Screen 條件和每個 Test 模型各有獨立完整機率快取。重跑相同命令會核對來源、ID、機率和 SHA-256，從尚未完成的條件或模型開始。中斷時正在執行的條件或模型會從其第一個 batch 重跑。完整快取可用於無 CUDA 的融合，套件版本與其他來源設定需一致。

Test 必須讓 A、B 分別完全重現原提交的 10,791 筆 Top-1。若有差異，`replay_mismatch_a.csv` 或 `replay_mismatch_b.csv` 會保留差異的 ID 和標籤。候選產生後放在 `runs/v16/candidate/`。

- `submission_v16.csv` 是使用範本 ID 順序的 `id,character` 資料。
- `class_changes.csv` 和 `transitions.csv` 記錄相對 A3 的各類增減和角色轉移。
- `summary.json` 含兩模型重現差異數、權重、改判比例、checkpoint 與提交檔 SHA。候選由使用者手動提交。

先回傳 Validate 終端 JSON 的 `selection`、`weight_a`、各組 `hard_vs_a`。Test 階段回傳 `summary.json` 的 `replay_mismatches`、`changed_vs_a`、`changed_vs_a_pct` 和 `submission_sha256`。如果停止，只需貼出錯誤訊息和差異筆數。
