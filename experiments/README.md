# 開發歷程 V1–V19（參考用）

這個資料夾保留最終方法之前的所有實驗程式，內容與當時在研究室 Server 執行的版本相同（只搬移位置）。每個版本的詳細設計與結果寫在對應子資料夾的 `README.md`。

最終繳交的方法在 [`../final/`](../final/README.md)。

## 版本總表

| 版本 | 主要內容 | Kaggle Public | 程式／說明 |
|---|---|---|---|
| V1 | 初始分類模型 | 0.51961 | — |
| V2 | ResNet-18 改進 | 0.58541 | `classification_v2.py`、[`v2/`](v2/README.md) |
| V3 | 多視角推論與測試圖診斷 | 0.59715 | `classification_v3.py`、[`v3/`](v3/README.md) |
| V4 | 幾何分布診斷與困難驗證 | — | `classification_v4.py`、[`v4/`](v4/README.md) |
| V5 | 外圍背景替換 | 0.65369 | `classification_v5.py`、[`v5/`](v5/README.md) |
| V6 | 合成場景短跑 | 0.67068 | `classification_v6.py`、[`v6/`](v6/README.md) |
| V7 | 可見顆粒去除 | 0.68211 | `predict_v7.py`、[`v7/`](v7/README.md) |
| V8 | ConvNeXt-Tiny（ImageNet-1K）＋十視角推論 | 0.75563 | `classification_v8.py`、[`v8/`](v8/README.md) |
| V9 | 免訓練推論配方 | 未改善 | `classification_v9.py`、[`v9/`](v9/README.md) |
| V10 | 偏心小角色與明亮背景微調 | 未改善 | `classification_v10.py`、[`v10/`](v10/README.md) |
| V11 | 角色區域檢索 | 未改善 | `classification_v11.py`、[`v11/`](v11/README.md) |
| V12 | ConvNeXt-Base 只訓練分類頭 | 未改善 | `classification_v12.py`、[`v12/`](v12/README.md) |
| V13 | Tiny／Base 部分解凍短跑 | 未改善 | `classification_v13.py`、[`v13/`](v13/README.md) |
| V14 | V8 全量延續訓練（A：原配方、B：分布適應） | A3 0.82082／B4 0.81742 | `classification_v14.py`、[`v14/`](v14/README.md) |
| V15 | V14-A Epoch 4 | 0.81958 | 同 V14 |
| V16 | V14-A3 80%＋V14-B4 20% 機率融合 | 0.82174 | `classification_v16.py`、[`v16/`](v16/README.md) |
| V17 | V14-A3／A4 checkpoint 權重平均 | 0.81958 | `classification_v17.py`、[`v17/`](v17/README.md) |
| V18 | 測試圖高信心偽標籤自我訓練 ⚠️ | 0.85943 | `classification_v18.py` |
| V19 | V18 第二輪偽標籤（未執行） | — | `classification_v19.py` |

輔助診斷：`distribution_audit/`（Train／Test 分布）、`v8_audit/`、`v8_controlled/`（V8 domain shift 與單因素診斷）、`ensemble/`（V5／V6 融合）。

## 為什麼 V1–V17 卡在 0.82

這段期間主要調整骨幹、推論視角、去顆粒與融合，並做過多次 Train／Test 影像統計（黑邊、亮度、細節）。但沒有直接重現測試圖的產生流程：測試圖的角色黑色輪廓線、眼白與旋轉黑角被換成照片背景（作業增強 #2），訓練圖則保有完整黑色輪廓與黑邊。V20 在訓練中重現這套流程後，Public 由 0.82 升到 0.985。

## ⚠️ V18／V19 與作業規則

作業規定「測試集僅用於產生最終預測」。V18／V19 以測試圖的模型預測當偽標籤再訓練，可能不符合此規定，因此不列入最終方法，也不作為最終提交。保留程式是為了如實記錄開發過程。

## 執行方式

從專案根目錄執行，例如：

```bash
python experiments/classification_v14.py --help
cd experiments && python -m pytest -q tests   # 多數測試需要 torch
```

各版本當時在 Server 上是以固定 Commit SHA 的原始碼執行（記錄在各版本的 README 與 `runs/vN/` 產物中）；搬移位置不影響那些固定版本。
