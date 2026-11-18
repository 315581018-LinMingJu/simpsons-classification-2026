# V7 前置診斷：篩出值得驗證的可見顆粒

V6 的 0.67068 仍是最佳提交。V5／V6 集成未通過選擇。這輪只驗證現有 V6 的一項前處理，不訓練、產生候選或提交 Kaggle。

## 真實影像與目標

附件包含 100 張隨機 Test 和 67 張 V3 早期改判圖。可見彩色顆粒的 6171、7126、9107、10399、5402 觸發所選濾波，2842 有顆粒但低於門檻。高頻但以背景紋理為主的 1369（書脊）、7182（瓦楞紋）、9816（磚縫）未觸發。候選像素比例不能當作真實噪聲密度，這個方法也可能漏掉成片噪聲或觸發細密紋理。隨機附件有 12／100 張觸發，特選改判附件有 15／67 張觸發。後者不能當作全體出現率，兩組也不代表 V6 的真實錯誤。

低頻背景與幾何差異已部分納入 V6。V7 優先測可局部處理的顆粒，原因是它在真實圖可觀察，而且只需推論改動。強反色、角色過小、偏心和多角色會需要不同機制。用拼圖保留這些線索，避免把同時出現的現象都歸到顆粒。小和偏心指可見圖塊，Test 沒有提供主角色選擇或真實框。裁切、遮擋、透視和疑似反色需人工確認。

## 只測一個濾波

先找 RGB 相對 3×3 中位數有明顯差異、四周有一致支持且原像素孤立的點。候選占比 ≥0.3% 時，才以局部中位數替換這些點。其餘像素完全相同，低於門檻整張圖不變。原與濾後影像沿用同一 V6 權重及 Multi-scale。對照保留原 batch 邊界；完全未觸發的 batch 可跳過第二次推論，未觸發圖片直接沿用原預測。

- **Review** 重用完整 Distribution Audit 的高頻細節、偏移帶及特徵支持分組，額外量測孤立點比例。拼圖包含高孤立點、低孤立點但高頻細節較多、隨機、人工看過的真實案例。自動排序的樣本標示需人工確認。若已有相符的 V6 機率快取，核對 checkpoint、Test、ID、class order 和原 Submission 後讀取，並列出信心與前兩名差值。
- **Validate** 從原去重驗證集選取每類最多 12 張，排除 V5／V6 使用過的局部驗證圖及其完全像素相同副本。比較 Clean、V6 紋理合成對照、在相同紋理與幾何上注入局部高振幅彩色顆粒。只有最後一組加入顆粒。合成只能量測機制，真實顆粒型態可能和注入結果不同。
- 有標籤比較報告救回、退步、啟用率及類別等權的配對 95% 區間。顆粒組提升區間下限大於 0，且 Clean 或紋理對照都沒有超過 0.5 pp 的退步，才支持繼續研究。即使通過，也要看真實拼圖的觸發是否集中在目標現象。若無證據或常誤觸紋理，保留原 V6。這個階段不產生 Submission。
- 驗證後只對選出的真實 Test 複核圖比較原與濾後 V6 的預測、信心、前兩名差值。沒有全量機率時只重播這些 ID。低信心或改判不能視作真實錯誤或救回。

CPU 預演用附件的 200 張 Train 圖加入相同配對條件。未注入額外顆粒的紋理對照 0 張觸發，注入組 200 張觸發。相對注入前的平均絕對像素誤差，中位數由 6.69 降至 1.84（0–255 尺度）。在此模擬中，約 99.75% 的被替換點確實是注入的點，約移除 69.7% 的注入點。這些數字只說明濾波能處理此種注入，不能推論真實 Test 或模型準確率。

## Server 指令

沿用原 .venv、資料、Distribution Audit 和 V6。全量 CPU Review 標準輸出在 `runs/v7_diagnostics/review/`。若路徑和預設不同，兩階段可指定 `--audit-run`、`--v6-run`、`--v6-submission` 或 `--v6-probabilities`。若要重跑，同樣指定新的 `--output runs/v7_diagnostics_...`。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
.venv/bin/python -u diagnose_v7.py review

nvidia-smi
read -r -p "已確認可使用的一張 GPU index：" gpu_index
CUDA_VISIBLE_DEVICES="$gpu_index" .venv/bin/python -u diagnose_v7.py validate
```

CPU Review 粗估 **3–12 分鐘**。GPU Validate 與少量真實 Test 複核粗估單張 RTX 5090 **5–20 分鐘、2–6 GiB VRAM**，尚未實測。預設 batch size 16、workers 4。若共享記憶體有限，兩階段可降低 `--workers`。

Review 產生 `review.json`、`test_categories.png`、`filter_examples.png`（有觸發例時）、`test_impulse_signals.csv`、`test_review.csv`。標註驗證產生 `validation/validation_report.json`、`paired_validation.csv` 及 `selected_test_v6_probe.csv`。先回傳兩份 JSON 即可。需要討論真實圖時，附上人工讀圖摘要或少量自選例子。權重、圖片與其他產物可以留在 Server。
