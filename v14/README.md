# V14｜V8 全量延續訓練與兩組候選

A、B 從同一 `runs/v8_full/models/convnext_all_layers/best.pt` 開始，使用 V8 full benchmark 的 77,547 張 Train，更新 ConvNeXt-Tiny 全層。V8 已提交的 `convnext_tta_denoise/submission_v8.csv` 仍是比較基準。V10 的 12,000 張、單 epoch、末段微調結果適用於該短跑，本次驗證全量多 epoch 全層延續訓練。

## CPU Prepare 與固定訓練設定

- 先驗證 V8 權重／封存 SHA、split、原 Train／Val manifest、類別、best epoch 和內嵌歷史。找到 V8 `last.pt` 時，核對其 epoch、歷史和 model tensors 是否逐一等於 best，再回報 optimizer 是否可用。也回報 `history.json` 是否吻合 best 前綴。
- 預設兩臂由 V8 的模型權重開啟新的 AdamW，避免較舊或不同權重的 moments 影響低 LR 試驗。若同一 checkpoint 的 optimizer 可驗證，可在 **prepare** 時選 `--reuse-v8-optimizer`。兩臂共用該政策。V14 後續恢復一定使用本臂最後完成 epoch 的 optimizer。
- 預設 4 epoch（prepare 可選 3–5），每 epoch 固定由 `3e-5` cosine 降到 `5e-6`，AdamW weight decay `1e-4`、梯度裁切 1.0，BF16 batch 16、累積 4。每 epoch 跑過完整 77,547 張，尾批按實際圖片數計算。所有非測試日誌和 checkpoint 分別存於 `runs/v14_continuation/a/`、`b/`。
- **A** 沿用 V8 checkpoint 記載的增強。V8 robust 在原圖增強外，以約 55% 路徑生成複合場景，部分加入 distractor、脈衝或 JPEG。
- **B** 第一到第四 epoch 分別以 70／65／60／55% 嘗試原 V8 配方，其餘拆成兩個明確分支。原尺寸換亮背景是 15／15／20／20%，只替換保守的連通外圍近黑遮罩。偏心小角色是 15／20／20／25%，重用 V10 的可見線索護欄，保留完整圖塊。第 5 epoch 保持第 4 的比例。新分支不再加第二角色、裁切或擦除；無可靠遮罩或前景太小就回退 V8 配方。每 epoch 保存實際各分支數量。V10 的固定 40% 單一新場景，在本試驗換成分支分開、固定逐 epoch 配比。兩個候選只用 Train 標籤和 V8 已有的分布稽核統計，避免引入新的外部資料。
- Kaggle Rules 在 Codex 仍遭代理 403。兩臂沿用 V8 的 ImageNet 衍生權重，確認競賽與作業均允許後才使用 `--allow-imagenet`。 [競賽 Rules](https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules)

## 選擇、恢復與用量

每 epoch 保存模型、optimizer、loss、實際更新數、Clean 及分條件 Hard、預測相較 V8 的改變、耗時與 VRAM。200 張固定 Screen（每類 4 張）使用正式 V8 FP32 二尺度十視角和選擇性去顆粒。V8 也在同一批圖及相同處理充分重跑一次作基準。候選只需通過最低安全線（Clean 落後不超過 3 pp，Hard 不超過 6 pp）。預設用 `0.4 Clean + 0.6 Hard` 選 checkpoint，同分用較低 loss、再用較早 epoch。亦可明確指定任一合格 `--epoch`。Test 改判只報告，供你手動決策。合成 Hard 的表現無法推出 Kaggle 提升。

Clean 低於 80% 或 Hard 落後 15 pp 以上直接停止。至少完成 3 epoch 後，連續 2 epoch 未提高選模分數 0.1 pp 即早停。每臂預設累計活躍時間 **4 小時**（含訓練與驗證，prepare 可調整到最多 8 小時）。預算在批次間檢查，時間約每 30 秒存檔。已完成 epoch 原子保存；中斷當下未完成的 epoch 需重跑。突然斷電可能漏記最後一次預算紀錄後的片段。OOM 後以 `--resume --batch-size 8 --workers 2` 續跑，有效 batch 維持 64；仍吃緊可降 4。每 epoch 保存實際 microbatch。準備時可用 `--eval-batch-size 4` 保守降低驗證占用。

RTX 5090 初估每 epoch（含驗證） **10–35 分鐘**，預設兩臂完整 4 epoch 合計約 **80–280 分鐘**，早停會縮短。不含可選的兩次 Test 推論。首次每臂另有 V8 參考驗證。訓練規劃 **8–20 GiB VRAM**，預留 **5 GiB 磁碟**。Test 每臂規劃約 **5–20 分鐘**，45 分鐘上限。實際以 Server 速度、共享 I/O 及報告為準。

## Server 指令

使用 pinned Commit 快照，工作目錄維持原專案根目錄。以本次 Commit SHA 取代下方變數，無需 Git checkout。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
V14_SHA='<本次 Commit SHA>'
V14_CODE="$PWD/runs/v14_continuation/code-$V14_SHA"
mkdir -p runs/v14_continuation
if [ ! -d "$V14_CODE" ]; then
  (set -eo pipefail
   v14_tmp="$(mktemp -d "$PWD/runs/v14_continuation/.sync.XXXXXX")"
   trap 'rm -rf -- "$v14_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/$V14_SHA" |
     tar -xz --strip-components=1 -C "$v14_tmp"
   mv -T -- "$v14_tmp" "$V14_CODE")
fi
.venv/bin/python -u "$V14_CODE/classification_v14.py" prepare --allow-imagenet
```

確認單張可用 RTX 5090，再於同一 shell 執行。透過 SSH 時可在 tmux 裡執行整個流程。

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' V14_GPU
export CUDA_VISIBLE_DEVICES="$V14_GPU"
.venv/bin/python -u "$V14_CODE/classification_v14.py" train --arm a --allow-imagenet &&
.venv/bin/python -u "$V14_CODE/classification_v14.py" train --arm b --allow-imagenet
```

中斷後對該臂加 `--resume`，例如 `train --arm a --resume --allow-imagenet`。同一路徑若已完成或停損，重跑報告原結果。

```bash
.venv/bin/python -u "$V14_CODE/classification_v14.py" predict --arm a --allow-imagenet
.venv/bin/python -u "$V14_CODE/classification_v14.py" predict --arm b --allow-imagenet
```

候選沿用已封存 V8 的批量、原 V8 十視角及選擇性去顆粒。若推論 OOM 可指定 `--predict-batch-size 8` 並記錄該差異。程式核對原 V8、指定 epoch checkpoint 的 SHA、10,791 唯一 ID 的原順序及合法角色，完成後才把候選資料夾原子發布至各臂的 `candidate_epoch_NN/submission.csv`。一起保存 `class_changes.csv`、`transitions.csv`、兩個 SHA 與改判數。Kaggle 提交由你操作。

最少回傳 prepare 的 `best_epoch`、`last_matches_best`、`optimizer_policy`，每臂 train 的 `decision` 與 selected 摘要，及兩個 predict 的最後一段 JSON。
