# V5／V6 機率集成對照

V6 的 Public Score 0.67068 高於 V5 的 0.65369，但 Heldout Composite 的 26.25 pp 大幅改善沒有等比例出現在 Kaggle。此實驗只使用原權重。Kaggle 分數不參與校準、選權重或確認。

## 實驗設計

1. **封存 CSV 的 CPU 稽核。** 比較完整 10,791 個相同 ID，列出改判比例、角色轉移、各角色預測流入流出，以及按視覺上尾、Train 組合支持、外圈黑邊和視覺組合分組的改判率。資料來自原 V5/V6 Submission 及既有 Distribution Audit。這些是預測分歧，沒有 Test 真實角色計數或正確率。
2. **新選取的標註驗證圖。** 沿用原 V1／V2 切分與跨 Train／Val 去重清單，排除舊 V5 局部驗證圖及其完全相同像素的副本。舊 V6 驗證取自該舊 V5 清單。剩餘 Val 按 RGB 像素去重，跨類衝突群剔除，避免完全相同影像跨新切分。每類最多 18 張，以 4／4／10 的預設配額分到校準、選擇、確認。小批次資料若每類不足 18，按約 1/4、1/4、其餘分配。同一原圖的所有變體同組。較早的 V1／V2 曾用整個原 Val 選 checkpoint，且仍可能有近似重複，確認集的獨立性限於本次集成選擇。
3. **重算機率。** V5 與 V6 都用原 V3 Multi-scale 的 10 視角，逐視角 Softmax 再平均。融合使用每張圖原始 50 維機率。驗證為 Clean、V5 Challenge 的 Geometry／Context／Clipped／Distractor／Negative，以及 V6 Heldout Composite。每條件的原機率以 NPZ 快取，續跑不重算已完成的條件。Test 另有快取。
4. **校準與挑選。** 兩個模型各以校準組擬合一個 `T`，用 `softmax(log(p)/T)` 控制 V3 Multi-scale 平均機率的尖銳程度。選擇組若整體 NLL 改善且 V5 Challenge NLL 未變差，才採用該溫度。報告 NLL、Brier、10 分箱 ECE 與平均信心，仍保留原 V5、原 V6 的對照。三個權重固定為 **V5／V6 0.25／0.75、0.50／0.50、0.75／0.25**。Clean、V5 Challenge、V6 Heldout 各占三分之一。V5 的五種 Challenge 取平均，類別等權。先從符合條件的權重選平均準確率最高者，持平時比較 NLL。
5. **一次確認。** 確認組只驗證選出的權重。在每個類別內重抽原圖，同一原圖的所有條件跟著重抽。需同時滿足相對原 V6 的三族平均提升之 95% CI 下限大於 0、V5 Challenge 的提升點估計大於 0、Clean 下降不超過 0.5 pp。只在 V6 合成條件表現好不足以通過。未通過就保留 V6，`predict` 會停止。通過只代表值得用一個候選驗證，無法估計 Kaggle 分數。

## Server 執行

CPU 階段用原 CSV。預設路徑是 `runs/v5_background/submission_v5.csv` 和 `runs/v6_composite/experiment/submission_v6.csv`，可用 `--v5-submission` 或 `--v6-submission` 指向真正提交的檔案。需要對應 V5 CSV 的 `runs/distribution_audit/`，路徑不同可指定 `--audit-run`。若模型跑次移到別處，也要指定 `--v5-run` 或 `--v6-run`。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
.venv/bin/python -u ensemble_v5_v6.py audit

nvidia-smi
read -r -p "已確認可使用的一張 GPU index：" gpu_index
CUDA_VISIBLE_DEVICES="$gpu_index" .venv/bin/python -u ensemble_v5_v6.py validate
```

預設驗證最多 900 張，每張 7 條件 × 2 模型 × 10 視角。單張 RTX 5090 預估驗證 **10–35 分鐘**，全量 Test 機率 **8–30 分鐘**，約 **2–6 GiB VRAM**，尚未實測。分批推論，batch size 16、4 workers。若要減少共享記憶體可用較小的 `--workers`。各階段沿用同一推論設定。需要改為不同做法時，使用新 `--output runs/v6_ensemble_...`，保留前次結果。

先看 `runs/v6_ensemble/validation_report.json`。只有 `supports_ensemble` 才執行下列 GPU 階段：

```bash
CUDA_VISIBLE_DEVICES="$gpu_index" .venv/bin/python -u ensemble_v5_v6.py predict
```

第一次會重算兩個模型完整 Test 機率，再以固定權重產生唯一 `runs/v6_ensemble/submission_v6_ensemble.csv`。若只需機率與封存類別重播差異，可執行相同指令的 `test-probs` 階段。它不輸出候選。重播差異在 `test/replay_check.json`，封存 CSV 保持原樣。集成始終使用實際重算的機率。程式不會訓練、覆蓋舊產物或提交 Kaggle。

最少只需回傳 `runs/v6_ensemble/v5_v6_test_audit.json` 和 `runs/v6_ensemble/validation_report.json`。完整角色轉移、分組改判與切分清單可留在 Server。
