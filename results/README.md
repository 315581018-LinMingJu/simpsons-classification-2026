# 設定檔與實驗結果

從研究室 Server 的 `runs/` 收集的小型設定與結果檔（不含模型權重 `.pt` 與機率快取 `.npz`）。路徑與 Server 上 `runs/` 底下相同。

## 最終方法（V20–V22）
| 路徑 | 內容 |
|---|---|
| `split_baseline_seed2026.csv` | baseline 切分（seed 2026、每類 20%）：每張訓練圖屬於 train 或 validation，所有方法共用 |
| `v20/<成員>/config.json` | 該次訓練的完整設定：模型、解析度、epochs、batch、lr、weight decay、label smoothing、drop path、p_testify、類別順序、背景張數 |
| `v20/<成員>/history.json` | 每個 epoch 的 train loss、Clean／Test-style 驗證正確率、訓練與總時間、峰值 VRAM |
| `v20/<成員>/confusion_val_clean.csv`、`confusion_val_testlike.csv` | 最佳 epoch 在兩組驗證集的 50×50 混淆矩陣（列＝真實、欄＝預測） |
| `v20/<成員>/candidate_epoch_NN/summary.json` | 測試集預測：submission SHA-256、使用的 epoch、驗證正確率、各類預測數、相對 V16／V18 改判數 |
| `v20/run_member.sh` | Server 上實際使用的成員啟動腳本（當時指向 `runs/v20/code-b539d567`；repo 版本見 `final/run_member.sh`） |
| `v21/<組合>/ensemble-N/summary.json` | V21 融合候選：各成員與融合後的驗證正確率、submission SHA-256 |
| `v22/<組合>/ensemble-N/summary.json` | V22 的 4 種融合組合比較（`new2` 為最終採用） |

成員：`main`（B@224）、`b288`（B@288）、`large`（L@224）、`large288`（L@288）、`base384`（B@384）、`ablation_no_testify`（對照組，關閉測試風格增強）。

## 報告實驗
| 路徑 | 內容 |
|---|---|
| `report/baseline_resnet18/` | 官方 baseline 的 `history.json`、`run_config.json`、`validation_by_class.json` |
| `report/*_conditions.json` | 四種干擾條件（clean／#1／#2／full）的正確率、最差類別、最常見混淆 |
| `report/*_confusion_full.csv` | full（Test-style）條件的 50×50 混淆矩陣 |
| `report/run_a.sh`、`run_b.sh` | 兩個報告實驗的執行腳本 |

## 開發歷程（V1–V18）
`v1_baseline/`、`v2_primary/`、`v5_background/`、`v6_*`、`v7_candidate/`、`v8_*`、`v9/`～`v14_continuation/`、`v16/`、`v17/`、`v18/`、`distribution_audit/` 為各版本當時產生的設定（`settings.json`、`run_config.json`）、訓練歷史（`history.json`）與驗證／診斷摘要（`summary.json`、`validation.json`），對應程式在 `experiments/`。
