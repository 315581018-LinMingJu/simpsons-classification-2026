# V10｜偏心小角色與明亮紋理背景

V8 已提交的 ConvNeXt-Tiny 分數是 0.75563，原 Screen Hard 是 90.26%。V9 的推論配方在 Screen 全數停損。V10 將同一個推論配方固定，直接檢驗改變訓練場景能否改善有標籤的偏移條件。

## 起點與訓練設定

- 起始及對照都用 `runs/v8_full/models/convnext_all_layers/best.pt`。程式檢查 ConvNeXt-Tiny、V8 全層模式、訓練快照、原切分、類別順序及 benchmark manifest。
- 從 V8 benchmark 的 Train 固定抽樣每類 240 張，總計 12,000 張。維持原標籤。只訓練一次、只解凍最後一個 ConvNeXt stage 與分類頭。AdamW 的骨幹 LR 為 `3e-6`，分類頭為 `3e-5`，權重衰減 `1e-4`。BF16，預設 batch 16，梯度累積 4。
- 40% 樣本嘗試新場景。近黑且連通外圍的區域才可去除，遮罩邊緣另外保留一像素。以 224×224 場景為基準，主圖塊最長邊約佔 44%–66%，其可見線索框與場景中心距離至少 14%。至少 6.5% 畫面為前景中非近黑的可見像素代理，線索框兩邊都須達 36 像素。主圖塊完整放在畫面內。
- 明亮背景使用程序生成的多尺度紋理、光影及少量背景顆粒。參數參考全量稽核的亮度、對比與高頻細節中位數。合成分支保留前景 RGB 經縮放的色彩和完整取景。沒有去色、反色、再裁切、擦除或額外角色。其餘樣本及遮罩不可靠者沿用 V8 checkpoint 記載的訓練增強。
- 可見線索是光度及遮罩估計。角色細節可能被暗背景誤分，請先看 CPU Audit 的前後拼圖。程序背景無法涵蓋實際照片的語意與遮擋。

Codex 的 CPU smoke 使用先前提供的 200 張 Train 樣本，196 張通過可見範圍檢查。合成畫面的亮度中位數為 0.4916、細節中位數為 0.0275；同 ZIP 的 100 張隨機 Test 樣本為 0.4657 及 0.0275。這只是低階影像量測，沒有使用 Test 標籤。正式 CPU Audit 會改用 Server 上的全量稽核參數、固定抽樣及當地環境，結果另存。

## 同步程式，從任何 Server checkout 狀態執行

從本次 GitHub Commit 複製完整 SHA。用 curl 將 pinned 程式碼解到 V10 獨立目錄。執行時工作目錄仍是資料專案根目錄，因此既有 `.venv`、`data/` 和 `runs/` 可直接沿用。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
V10_SHA='<本次 GitHub Commit 的完整 SHA>'
V10_CODE="$PWD/runs/v10/code-${V10_SHA}"
mkdir -p runs/v10
if [ ! -d "$V10_CODE" ]; then
  (set -eo pipefail
   v10_tmp="$(mktemp -d "$PWD/runs/v10/.sync.XXXXXX")"
   trap 'rm -rf -- "$v10_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/${V10_SHA}" |
     tar -xz --strip-components=1 -C "$v10_tmp"
   mv -T -- "$v10_tmp" "$V10_CODE") || exit 1
fi
```

新功能檔案為 `classification_v10.py`、`v10/augment.py`、`v10/experiment.py`、`v10/README.md`、`tests/test_classification_v10.py`。下載同一 commit 的來源快照也提供 V8 等共用模組。產物記錄使用的來源檔 SHA-256、V8 checkpoint SHA-256、benchmark 與資料快照及固定種子。程式沿用 V8 已核可的 ImageNet 權重來源，`--allow-imagenet` 沿襲原有權重授權確認。V10 只載入本機 V8 權重。

## 執行順序

**CPU Audit。** 使用 `runs/v8_full/benchmark/` 與 `runs/distribution_audit/summary.json`。從 V10 Train 子集每類抽 5 張，強制嘗試增強，輸出 `runs/v10/audit/summary.json`、`preview.png` 和完整訓練 manifest。可用場景小於一半時停止 GPU 實驗。

```bash
.venv/bin/python -u "$V10_CODE/classification_v10.py" audit --allow-imagenet
```

**單卡訓練及 Screen。** 請先依實驗室規則確認一張可用卡。透過 SSH 執行建議用 tmux。只有一個 epoch，完成後 checkpoint 可續用。OOM 時可在同一個輸出目錄下用不同子目錄，例如 `--output runs/v10/batch8`，先跑 CPU Audit，再用 `--batch-size 8 --workers 2` 跑訓練。梯度累積自動調到 8，有效 batch 維持 64。若已完成訓練，只需完成對應的驗證階段。

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' V10_GPU
export CUDA_VISIBLE_DEVICES="$V10_GPU"
.venv/bin/python -u "$V10_CODE/classification_v10.py" train --allow-imagenet &&
.venv/bin/python -u "$V10_CODE/classification_v10.py" screen --allow-imagenet
```

V8 和 V10 在原 V8 去重 Screen／Confirm 的相同圖片、干擾與種子上評估。先固定 V8 的 FP32 短邊 240／288 十視角 Softmax 平均與選擇性去顆粒。保留 Clean、Context、Impulse、Clipped、Distractor、Negative。每個階段記錄各條件準確率、配對救回／退步與改判數。Hard 權重沿用 V8。信賴區間以原圖為單位在類別內重抽，同一張原圖的不同條件保持配對。

| 階段 | 全部通過才前進 |
| --- | --- |
| Screen | Hard 至少 +0.75 pp，Clean 降幅至多 0.5 pp，Context 不退步，其餘四個 Hard 條件各自降幅至多 4 pp |
| Confirm | Hard 至少 +1.0 pp，配對 95% CI 下界大於 0，其他保護條件同 Screen |

Screen 必須回報 `advance_to_confirm`，才能拿同一 checkpoint 跑一次 Confirm。任一階段 `stop` 時保留 V8。Confirm 若在 V8 研究中曾被使用，仍可能受先前模型選擇影響。合成有標籤驗證的通過僅支持候選實驗，真實 Kaggle 結果仍待提交。

```bash
.venv/bin/python -u "$V10_CODE/classification_v10.py" confirm --allow-imagenet
```

只有 Confirm 顯示 `eligible_for_candidate`，才可用相同 V8 推論方式輸出 `runs/v10/candidate/submission_v10.csv`。程式檢查與模板相同的 10,791 個唯一 ID、原順序及 50 類合法標籤。用哪一份候選正式提交由你決定。

```bash
.venv/bin/python -u "$V10_CODE/classification_v10.py" predict --allow-imagenet
```

## 估計資源與最少回報

以下以實驗室單張 RTX 5090、batch 16 和 4 workers 規劃，GPU 與 I/O 時間以 Server 紀錄為準。

| 階段 | 時間 | GPU 記憶體 | 只需貼回 |
| --- | --- | --- | --- |
| CPU Audit | 1–3 分鐘 | 0 | `decision`、`acceptance`、`geometry_p10_p50_p90` |
| 固定一個 epoch | 15–40 分鐘 | 8–16 GiB | `loss`、`seconds`、`peak_reserved_gib` |
| Screen | 5–18 分鐘 | 4–8 GiB | `decision`、`reasons`、`hard_effect`、`condition_gain_pp` |
| Confirm | 5–18 分鐘 | 4–8 GiB | 同 Screen |
| 達標後產出候選 | 12–35 分鐘 | 4–8 GiB | 終端的候選路徑與 ID 數 |

新產物集中於 `runs/v10/`。Screen／Confirm 保留對照和新模型各自的配對預測檔與報告。中斷後可重用同設定已完成的權重及報告。
