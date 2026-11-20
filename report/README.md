# 實驗報告：The Simpsons 50 類角色分類

## 1. 問題與資料

- 訓練集 96,929 張（50 類，224×224），依 `baseline.py` 切分（seed 2026、每類 20%）：Train 77,547／Val 19,382。所有方法使用同一組驗證集。
- 測試集 10,791 張，244–284 px。實際觀察測試圖後確認，它是用作業頁面提供的兩段增強程式產生的：
  - **#1**：旋轉／仿射／透視、色彩、灰階、反相、色階壓縮、Solarize、彈性變形、模糊、多種雜訊
  - **#2**：近黑（<20）與近白（>235）像素視為透明，加 10–30 px 邊框後貼到真實照片背景。角色的**黑色輪廓線、眼白**與旋轉產生的**黑色角落**都因此變成背景紋理。
- 乾淨訓練圖的驗證分數不能代表測試表現。因此另建 **Test-style 驗證集**：把 #1＋#2 以每張固定的種子套在驗證圖上，不使用任何測試圖。

## 2. 方法

| 方法 | 與基準的差異 |
|---|---|
| **基準**：官方 `baseline.py` | ResNet-18（ImageNet-1K）、補邊成 224、輕度翻轉／旋轉／亮度對比、AdamW 1e-4（分類頭 1e-3）、10 epochs、patience 3 |
| **對照組**（`--p-testify 0`） | ConvNeXt-Base（ImageNet-22K）完整微調、label smoothing、cosine LR；**不使用**測試風格增強 |
| **V20（改進）** | 與對照組完全相同，**只改一項**：訓練時 85% 圖片套用重現的 #1＋#2 增強（背景用 DTD 紋理照片） |
| V20-b288／V20-large | V20 換成 288 解析度／ConvNeXt-Large |
| **V21（最終提交）** | V20-b288＋V20-large 的 softmax 機率平均；組合依 Test-style 驗證集挑選 |

完整設定見 [`../final/README.md`](../final/README.md)。

## 3. 結果

### 3.1 各干擾條件下的驗證正確率

同一組 19,382 張驗證圖，四種條件的干擾圖對所有模型完全相同（固定種子），不使用 TTA。工具：[`evaluate_conditions.py`](evaluate_conditions.py)。

| 方法 | Clean | 只有 #1 | 只有 #2 | #1＋#2（Test-style） |
|---|---|---|---|---|
| 基準 ResNet-18 | 99.54% | 59.89% | 92.97% | 47.26% |
| 對照組 ConvNeXt-B（無測試風格增強） | 99.87% | 87.82% | 98.63% | 78.47% |
| **V20 ConvNeXt-B（有測試風格增強）** | **99.92%** | **98.93%** | **99.89%** | **98.73%** |
| V20-b288 | 99.93% | 99.38% | 99.91% | 99.16% |
| V20-large | 99.93% | 99.34% | 99.92% | 99.11% |

V21 融合（水平翻轉 TTA、機率平均）：Clean 99.943%、Test-style 99.345%。

### 3.2 Kaggle 分數

| 方法 | Test-style 驗證 | Kaggle Public |
|---|---|---|
| 基準 ResNet-18 | 47.26% | 0.58418（主辦方公布的 baseline 實測） |
| 開發期最佳 V16（ConvNeXt-Tiny，無測試風格增強） | — | 0.82174 |
| V20 | 98.73% | 0.98517 |
| **V21** | **99.35%**（TTA） | **0.98887** |

Test-style 驗證和 Kaggle 分數的趨勢一致：V20 的 98.73% 對上 0.985。乾淨驗證集則所有方法都在 99.5% 以上，完全分不出差異。

### 3.3 訓練成本（RTX 5090 單卡，BF16）

| 方法 | Epochs | 時間 | VRAM |
|---|---|---|---|
| 基準 ResNet-18 | 10 | 14.6 分鐘（含每 epoch 驗證） | — |
| 對照組 ConvNeXt-B | 10 | 32.4 分鐘（訓練 26.1） | 10.7 GiB |
| V20 ConvNeXt-B@224 | 10 | 32.3 分鐘（訓練 25.9） | 10.7 GiB |
| V20-b288 | 12 | 每 epoch 訓練約 4.6 分鐘 | 15.6 GiB |
| V20-large | 10 | 每 epoch 訓練約 3.7 分鐘 | 16.7 GiB |

時間含每個 epoch 的 Clean 與 Test-style 驗證。測試風格增強在 CPU 上以 40 個 worker 平行產生，**對照組與 V20 的訓練時間幾乎相同**（26.1 vs. 25.9 分鐘），也就是這項改進幾乎不增加訓練成本。

## 4. 分析

### 4.1 改進來自哪裡
- **基準 → 對照組（換成較強的預訓練骨幹）**：Test-style 47.3% → 78.5%（+31.2 pp）。這一步同時改了架構、預訓練資料、最佳化與輸入處理，屬於多項因素一起變動，只能看成「較強基準」的整體效果。
- **對照組 → V20（只加入測試風格增強，單一因素）**：78.5% → 98.7%（**+20.3 pp**）；Clean 幾乎不變（99.87% → 99.92%）。這是最主要的改進，而且是公平的單因素對照。
- **V20 → V21（更高解析度、更大模型、融合）**：98.7% → 99.35%，Kaggle 0.985 → 0.989。

### 4.2 雜訊、frame、背景變化的影響
- **#1（幾何、色彩、雜訊）比 #2（外框、背景）更傷**：基準在只有 #1 時掉到 59.9%，只有 #2 時還有 93.0%；對照組分別是 87.8% 與 98.6%。
- **兩者疊加有交互作用**：基準 −39.6 pp（#1）與 −6.6 pp（#2）分開看，但一起套用掉了 52.3 pp；對照組也一樣（−12.0／−1.2 → 一起 −21.4 pp）。原因是 #1 的旋轉會產生黑色角落、反相會讓原本的亮色區域變暗，#2 再把這些近黑、近白的區域換成背景，角色的輪廓和大面積區域一起被破壞。
- **訓練時見過同樣的干擾，影響幾乎消失**：V20 在四種條件下都在 98.7% 以上，#2 的影響小於 0.1 pp。

### 4.3 容易混淆的角色（Test-style 條件）
- **基準**：干擾下大量預測成 `comic_book_guy`（snake_jailbird、barney_gumble、gary_chalmers、lunchlady_doris、fat_tony 被判成 comic_book_guy 各 50 張以上），代表模型在看不清楚時退化成固定答案。最差的類別是 timothy_lovejoy（22%）、patty_bouvier（30%）、gary_chalmers（31%）。
- **對照組**：錯誤集中在外型或髮色相近的角色：maggie → lisa（36 張）、otto_mann → moe_szyslak／krusty、sideshow_mel → krusty（綠色爆炸頭 vs. 綠髮小丑）。最差的是 otto_mann（45%）、timothy_lovejoy（51%）。
- **V20／V21**：剩下的錯誤都在合理範圍內：**patty_bouvier ↔ selma_bouvier（雙胞胎姊妹）**、lisa ↔ maggie（同色髮型的姊妹）、fat_tony → moe_szyslak、bart → marge。每類正確率都在 97% 以上。

50×50 混淆矩陣：`runs/report/<方法>_confusion_full.csv`，以及各訓練 run 的 `confusion_val_clean.csv`、`confusion_val_testlike.csv`（列為真實、欄為預測）。

## 5. 失敗的嘗試與誠實說明

- V1–V17 嘗試過背景替換、合成場景、去顆粒、多視角推論、檢索、ConvNeXt-Tiny／Base 部分微調、checkpoint 權重平均等方法，Public 一直停在 0.52–0.82。這些方法沒有重現 #2 造成的輪廓線消失。詳見 [`../experiments/README.md`](../experiments/README.md)。
- **V18（測試集偽標籤）**：以測試圖的高信心預測當標籤再訓練，Public 0.85943。作業規定「測試集僅用於產生最終預測」，此作法可能不符合規定，因此不列入最終方法與最終提交。
- Test-style 驗證集是依照作業增強程式**重新實作**的，背景照片（DTD）也和主辦方使用的背景不同，所以它是測試條件的近似，不是同一個分布。
- 基準的 Kaggle 分數 0.58418 是主辦方公布的實測值；本報告的基準驗證數字是在研究室 Server 用同一份 `baseline.py` 重新訓練得到，沒有另外提交 Kaggle。

## 6. 重現

```bash
# 基準
python baseline/baseline.py train --train-dir data/extracted/train --output runs/report/baseline_resnet18 --workers 16
# 對照組（只關掉測試風格增強）
python final/classification_v20.py train --backgrounds "$BG" --workers 40 --epochs 10 --p-testify 0 --output runs/v20/ablation_no_testify
# 四種條件評估
python report/evaluate_conditions.py baseline runs/report/baseline_resnet18/best.pt --name baseline --backgrounds "$BG"
python report/evaluate_conditions.py v20 runs/v20/main/best.pt --name v20_main --backgrounds "$BG"
```
V20／V21 的完整指令見 [`../final/README.md`](../final/README.md)。
