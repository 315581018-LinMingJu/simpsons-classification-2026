# 執行 log 檔

在研究室 Server（RTX 5090）實際執行時以 `tee` 保存的終端機輸出，2026-11-27 從 Server 的 `runs/` 收集。

**整理方式**：原始 log 中進度條每隔幾秒輸出一行，為避免檔案過大，只刪除進度條的中間更新行，保留每個階段 `100%` 的完成行，以及所有 JSON 結果、警告與錯誤訊息。除此之外內容未修改。

| 檔案 | 版本／用途 | 對應程式 |
|---|---|---|
| `v1_baseline/train.log` | V1 初始模型訓練 | `experiments/` |
| `v17/predict_log.txt` | V17（A3／A4 權重平均）測試推論 | `experiments/classification_v17.py` |
| `v18/smoke_log.txt`、`v18/run_log.txt` | V18 偽標籤：smoke 測試；訓練 2 epoch＋推論（規則疑慮，未採用） | `experiments/classification_v18.py` |
| `v20/smoke_log.txt` | V20 smoke 測試（30 batch，量測速度 197 img/s） | `final/classification_v20.py` |
| `v20/main_log.txt` | **V20 main**：ConvNeXt-B@224 訓練 10 epoch＋推論 | 同上 |
| `v20/b288_log.txt`、`v20/large_log.txt` | V21 成員：B@288、L@224（訓練＋推論＋驗證集機率） | `final/run_member.sh` |
| `v20/large288_log.txt`、`v20/base384_log.txt` | **V22 成員**：L@288、B@384 | 同上 |
| `report/a_log.txt` | 報告實驗 A：官方 baseline（ResNet-18）訓練＋四種干擾條件評估 | `baseline/baseline.py`、`report/evaluate_conditions.py` |
| `report/b_log.txt` | 報告實驗 B：對照組（`--p-testify 0`）訓練＋對照組、main、b288、large 四種條件評估 | 同上 |

V2–V16 的 log 當時沒有以 `tee` 保存，結果改以 `results/` 中的 `history.json`、`summary.json` 等記錄檔呈現。
