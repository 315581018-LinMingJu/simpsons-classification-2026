# Simpsons 50 類分類 V2

競賽 `ml-2026-iaii-classification2`。從專案根目錄執行以下指令，V1 檔案、模型、紀錄與 Submission 都作為唯讀輸入。

## V1 原始碼審查

- V1 以 ImageNet 權重初始化 ResNet18，使用 AdamW（LR 與 weight decay 均為 1e-4）、CrossEntropyLoss、FP32。預設 10 epochs、batch size 64、seed 2026。V2 從 V1 的實際設定讀取 epochs、batch size 與 seed，並檢查其餘設定與附件一致。
- 訓練的 `RandomResizedCrop(224)` 預設面積範圍為 0.08–1.0，可能把角色裁得只剩局部。驗證與測試共用等比 `Resize(256)` 和 `CenterCrop(224)`，對長寬比偏大或角色偏離中心的測試圖可能裁掉有效內容。你提供的直接拉伸推論分數更低，這次正式對照保留 V1 的驗證與推論前處理。
- V1 在訓練和推論都使用當次 `ImageFolder.classes` 的排序，程式本身沒有顯示固定的類別映射錯誤。風險在於讀取舊切分只核對總筆數，沒有逐筆核對圖片路徑、資料夾類別和重複列，載入最佳權重後也沒有比對 checkpoint 的類別順序。V2 補上這些檢查。
- V1 先寫設定和類別，再自動續跑既有 `last.pt`，因此重新執行可能改寫歷程、權重與 Submission。V2 新跑次必須用新目錄，並把推論獨立成可重試的指令。
- 目前只取得 V1 Python 附件。你提供的 **99.4686%** 驗證準確率、**0.51961** Public Score、拉伸推論 **0.48285** 與 **101 組跨集合配對** 尚未以原始資料與紀錄重算。程式啟動時才會讀取並核對四份原檔。

## RTX 5090 Ubuntu 安裝

使用 Python 3.10–3.13（建議 3.12）與支援 CUDA 12.8 的 NVIDIA 驅動。PyTorch 使用預編譯 CUDA 12.8 wheel，可透過 TORCH_HOME 使用已下載的 ImageNet 權重快取。若執行機器不能連線到 `download.pytorch.org`，需先準備套件 wheel 和 ResNet18 官方權重快取。

```bash
nvidia-smi
python3 -m venv v2/.venv
source v2/.venv/bin/activate
python -m pip install -r v2/requirements-rtx5090.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_arch_list()); x=torch.ones(1,device='cuda'); print((x+x).item())"
```

把現有資料與 V1 原檔放在以下位置，或傳入 `--data`、`--v1`、`--template`。`v2/.venv` 和實驗輸出是本機檔案，請勿將競賽資料和權重加入 Git。

```text
data/extracted/train/train/train/<character>/*.jpg
data/extracted/test_images/test_images/*.jpg
data/extracted/sample_submission.csv
runs/v1_baseline/{run_config.json,history.json,classes.json,split.csv}
runs/v1_baseline/best.pt  # 選用，診斷時用於重算 V1 子集分數
```

## 執行順序

```bash
python classification_v2.py audit
python classification_v2.py smoke
python classification_v2.py train
python classification_v2.py predict
python classification_v2.py diagnostic --v1-best runs/v1_baseline/best.pt
```

1. **audit** 在 CPU 核對原始切分與訓練資料一一對應（77,547 / 19,382 張、50 類），驗證 10,791 個不重複的測試 ID 和測試圖片一致，然後逐張解碼訓練資料、對 RGB 像素與尺寸做 SHA-256 分組。報告位於 `runs/v2_audit/audit.json`。同一群組可形成多個配對，且不同雜湊定義可能與你先前的 101 組統計不同。
2. **smoke** 在 GPU 跑兩個訓練批次和一個驗證批次，不儲存模型、歷程或 Submission。正式跑次之前用這一步確認 CUDA、資料讀取、預訓練權重與張量運算。
3. **train** 使用和 V1 完全相同的切分檔與驗證流程。唯一的實驗因子是訓練增強配方。將裁切的最小面積提高至 55%，加上輕量旋轉、平移、縮小、灰階與局部遮擋，模擬測試圖中觀察到的幾何變化與材質缺失。這些增強以一組配方一起評估，無法從單一 V2 判定哪項增強貢獻最多。保持同架構、權重來源、優化器、epoch 數、batch size 與精度。保留原始 `split.csv` 位元組與 SHA-256、`classes.json`、`run_config.json`、`history.json`、`best.pt` 和 `summary.json`。摘要紀錄最佳準確率、相較 V1 的百分點差與耗時。
4. **predict** 讀取已完成的 V2 正式跑次與最佳 checkpoint。以範本的字串 ID 順序輸出 `runs/v2_primary/submission_v2.csv`，保留前導零。只讀範本的 ID，不採用範本提供的角色值。檔案包含 10,791 筆預測及 `id,character` 兩欄。
5. **diagnostic** 保持同一批驗證圖片，移除訓練集中所有與驗證圖 RGB 像素完全相同的圖片，再用同樣的 V2 設定重訓。以完整驗證集和受重複影響／無完全重複的驗證子集報分。若有 V1 `best.pt`，重算其相同子集分數。結果寫入 `runs/v2_dedup`，不產生 Submission。只檢測完全相同像素，無法排除近似重複。子集難度、訓練樣本量與最佳 epoch 選擇也會影響差值，因此把這組結果視為洩漏診斷，V1 與 V2 的正式比較只引用第 3 步。若無 V1 checkpoint，省略 `--v1-best`。

中斷後可保留原跑次並用 `--run runs/v2_primary_retry1` 開新跑次，接著對 `predict` 指定相同的 `--run`。若診斷要使用該跑次，傳入 `--reference-run runs/v2_primary_retry1`。若 GPU 記憶體不足，正式對照應先確認原 V1 的 batch size 與實際裝置。未完成的訓練不能產出 Submission。後續版本依序 V3、V4，另用 `classification_v3`、`submission_v3.csv` 等名稱，保留各版設定與紀錄。
