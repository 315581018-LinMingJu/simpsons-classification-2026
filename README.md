# The Simpsons 50 類角色分類（ML 2026 IAII Classification）

Kaggle 競賽：[ml-2026-iaii-classification2](https://www.kaggle.com/competitions/ml-2026-iaii-classification2)
姓名：林明儒　學號：315581018

## 成績摘要

| 方法 | 驗證正確率（Clean／Test-style） | Kaggle Public |
|---|---|---|
| 官方 Baseline：ResNet-18 | 99.54%／47.26% | 0.58418（主辦方公布） |
| 開發期最佳（V16，ConvNeXt-Tiny 機率融合） | — | 0.82174 |
| 對照組：ConvNeXt-Base，不用測試風格增強 | 99.87%／78.47% | — |
| **V20**：重現測試干擾增強＋ConvNeXt-Base（ImageNet-22K） | 99.917%／98.824% | 0.98517 |
| V21：ConvNeXt-Base@288＋ConvNeXt-Large@224 機率融合 | 99.943%／99.345% | 0.98887 |
| **V22（最終）**：ConvNeXt-Large@288＋ConvNeXt-Base@384 機率融合 | 99.943%／99.598% | **0.99073** |

- 驗證集完全重現 `baseline.py` 的切分（seed 2026、每類 20%，Train 77,547／Val 19,382）。
- **Test-style 驗證**：把與測試集相同的干擾流程（見下）以固定種子套在驗證圖上，用來估計模型在真實測試條件下的表現；模型與融合組合都只依驗證集挑選，**未用 Public 分數挑選**。
- 基準與對照組的驗證數字不含 TTA；V20／V21 含水平翻轉 TTA。四種干擾條件、全部不含 TTA 的完整比較見報告。
- 最終 Kaggle 選擇：V22＋V21。

## 給助教的閱讀順序

1. 本頁：結果與核心想法
2. [`report/Simpsons_分類實驗報告_315581018.pdf`](report/Simpsons_分類實驗報告_315581018.pdf)：**完整實驗報告 PDF（34 頁）**，含 Task 2 混淆矩陣與 Task 3 filter／feature map
3. [`report/README.md`](report/README.md)：實驗報告摘要（基準與改進的對照、各干擾條件分析、混淆角色、訓練成本）
4. [`final/README.md`](final/README.md)：最終方法的完整設定與重現指令
5. [`final/classification_v20.py`](final/classification_v20.py)：訓練、增強、驗證、預測（單一檔案）
6. [`final/classification_v21.py`](final/classification_v21.py)：驗證集挑選的機率融合（V21、V22 都用這支程式）
7. [`experiments/README.md`](experiments/README.md)：V1–V19 開發歷程與失敗原因（參考用）

## 目錄結構

```
README.md              ← 本頁（總覽）
report/                ← 實驗報告（PDF、README）與評估工具
baseline/              ← 官方 baseline.py（ResNet-18，原封不動）
final/                 ← 最終繳交方法（V20 訓練程式、V21／V22 融合）
  classification_v20.py    訓練／預測主程式
  classification_v21.py    驗證集機率與融合
  run_member.sh            一鍵訓練一個融合成員
  requirements.txt         實際使用的套件版本
  tests/                   CPU 單元測試
experiments/           ← 開發歷程 V1–V19（保留原始程式與各版 README，可重現）
```

## 核心發現：測試集是用作業附的增強程式產生的

觀察測試圖後確認：

- 訓練圖 224×224；測試圖 244–284 px，正好是 224 加上作業增強 #2 的 10–30 px 邊框。
- 增強 #2 把近黑（<20）與近白（>235）像素視為透明再貼到真實照片背景上，所以**角色的黑色輪廓線、眼白、旋轉後的黑色角落都變成背景紋理**。
- 多數測試圖另有作業增強 #1 的旋轉／仿射／透視、反相、灰階、色階壓縮、彈性變形、雜訊。

V1–V17 用一般增強與更好的推論，Public 卡在 0.52–0.82。V20 改成**在訓練時重現同一套 #1＋#2 流程**（背景使用 DTD 紋理照片），並換成 ImageNet-22K 預訓練的 ConvNeXt 完整微調，Public 由 0.82 升到 0.985。

## 規則遵循

- 只使用訓練集標籤；測試集只用於產生最終預測，沒有用於訓練、驗證或挑選模型（V20、V21）。
- V18 曾以測試圖的高信心預測做偽標籤自我訓練（Public 0.85943）。作業規定「測試集僅用於產生最終預測」，此作法可能不符合規定，因此**不列入最終方法、不作為最終提交**，程式與紀錄保留在 `experiments/` 以如實呈現開發過程。

## 引用與來源

- ConvNeXt：Liu et al., *A ConvNet for the 2020s*, CVPR 2022；權重來自 [timm](https://github.com/huggingface/pytorch-image-models)：`convnext_base.fb_in22k_ft_in1k`、`convnext_large.fb_in22k_ft_in1k`
- DTD 背景：Cimpoi et al., *Describing Textures in the Wild*, CVPR 2014（以 `torchvision.datasets.DTD` 下載，僅作背景，不含角色標籤）
- 增強流程：依本課程 Kaggle 頁面提供的「TRANSFORMING AND AUGMENTING IMAGES #1、#2」程式重新實作
- 切分：本課程 `baseline.py`
