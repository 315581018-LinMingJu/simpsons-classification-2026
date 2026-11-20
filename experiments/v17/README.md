# V17 相鄰 checkpoint 權重平均

V17 從 V14-A 同一條訓練軌跡的 `epoch_03.pt`、`epoch_04.pt` 各取一半浮點模型張量，建立一個 ConvNeXt-Tiny。V16 的 A80/B20 是兩模型分別完成推論後融合機率。V17 的 Test 只需要一個模型執行十視角推論。

Prepare 核對兩個 checkpoint 的 SHA-256、同一組 V14-A 設定與訓練歷史、V14 固定切分、類別順序，以及實際 ConvNeXt-Tiny 的張量名稱、形狀和 dtype。浮點參數和 buffer 須有限值，平均後再查有限值。整數與布林 buffer 須逐元素一致，通過後逐一複製。獨立的 `averaged.pt` 只含平均模型、實驗來源和張量計數。所有階段都只讀取 V8、V14 與 V16 封存。

## 來源與執行

從原專案根目錄執行，沿用原本的 `.venv`、資料位置及時間戳。若用固定 SHA 的 GitHub 原始碼封存部署，可把整份原始碼解壓至 `runs/v17/code-<SHA>`，從原根目錄執行該路徑內的 `classification_v17.py`。

預設來源是 `runs/v14_continuation/a/epoch_03.pt`、`epoch_04.pt`、原 V14-A 的 `settings.json`、`history.json`、V14 的 `prepare.json`，以及 A3 的 `candidate_epoch_03/` 內原始 Summary 與 CSV。Test 與 Screen 來源沿用 V8／V14 的資料指紋與清單。若 A3 候選目錄已搬移，可指定 `--archive-a3`。

比較用的 V16 須為 A80/B20，保留 `candidate/` 內的 `submission_v16.csv` 與 `summary.json`、其父目錄中的 `prepare.json` 與 `validation.json`。程式會尋找 `runs/v16/candidate` 和 `runs/v16/*/candidate` 中唯一的相符版本。若封存另存或有多份，請在三個資料階段都加上同一個 `--v16-candidate <候選目錄>`。

```bash
.venv/bin/python -u classification_v17.py prepare
nvidia-smi
# 設成已確認有權使用的單張 GPU。
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -u classification_v17.py probe
.venv/bin/python -u classification_v17.py validate --allow-imagenet
.venv/bin/python -u classification_v17.py predict --allow-imagenet
```

Screen 使用 V14 固定的 200 張標記影像與原六個條件，沿用 Screen batch size；Test 依範本順序跑 10,791 張，沿用 A3 的原 Test batch size。預處理依 V14 正式流程選擇性去顆粒，在短邊 240 與 288 各取五個視角，各做 FP32 Softmax 後平均視角機率。CPU Prepare 預留 1～5 分鐘。單張 RTX 5090 的初估是完整 Screen 3～15 分鐘，Test 5～20 分鐘，GPU 預留 4～8 GiB。實際時間依圖片讀取、原 batch size 和 GPU 負載而異。

## 候選護欄與產物

Screen 計算 Clean、加權 Hard、各條件的配對救回與退步及 95% CI。加權 Hard 沿用 V14 條件權重。相對 A3，V17 的 Clean 下降超過 2 pp、加權 Hard 下降超過 3 pp，或任一 Hard 條件下降超過 10 pp，即停止產生候選。報告另含相對 A4 的配對結果。Screen 曾用來評估 V14 模型，結果應作為嚴重退步的篩查。

可用 `tmux new -s v17` 執行並用 `tmux attach -t v17` 接回。終端顯示實際完成的圖片數、百分比、Elapsed 和 ETA。每個模型的每個 Screen 條件及 V17 Test 各有獨立完整快取。再次執行同一階段時會先核對快取來源、ID、機率矩陣與 SHA-256。中斷時進行中的條件或 Test 會從該快取的第一批重做。

- `runs/v17/prepare.json` 記錄 checkpoint、切分與推論設定。
- `runs/v17/validation.json` 記錄三模型指標、救回、退步及護欄判定。
- `runs/v17/candidate/submission_v17.csv` 含按原範本排列的 `id,character`。
- 候選目錄含 `summary.json`、相對 A3 及 V16 的類別變動與轉移 CSV。由使用者手動提交。

請回傳 Validate 終端 JSON 的 `decision`、`scores`、`gaps_vs_a3`。若產生候選，再提供 `changed_vs_a3`、`changed_vs_v16`、兩個對應比例及 `submission_sha256`。
