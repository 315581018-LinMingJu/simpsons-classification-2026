# V1／V2 本地抗干擾比較

比較兩份 ResNet18 最佳權重在同一批驗證圖上的表現。程式只做推論，讀取本機訓練圖片與跑次紀錄。結果存於新跑次，實驗過程不需要 Kaggle 連線。

## 在 RTX 5090 執行

使用實驗室現有 Python 3.10、PyTorch 2.14.1+cu130 與 torchvision。圖表需要 Pillow 和 Matplotlib。若缺少 Matplotlib，可在原虛擬環境只補裝 `matplotlib`，無須安裝 V2 的 CUDA 12.8 套件清單。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
source .venv/bin/activate
python -c 'import torch, torchvision, PIL, matplotlib; print(torch.__version__, torch.cuda.get_device_name())'
python compare_v1_v2.py
```

程式讀取 `runs/v1_baseline/`、`runs/v2_primary/` 各自的 `split.csv`、`classes.json` 和 `best.pt`，以及 `data/extracted/train/train/train`。兩份切分逐位元比較，並核對實驗指定的 SHA-256。兩版權重的類別順序須與原始映射相同。V1 checkpoint 本身沒有切分雜湊，使用跑次的原始切分檔核對。V2 checkpoint 內嵌的雜湊也會核對。

預設輸出 `runs/v2_robustness/`。再次比較時指定新的目錄，例如 `python compare_v1_v2.py --output runs/v2_robustness_repeat1`。可使用 `--batch-size`、`--workers` 調整運算與 I/O。前置去重會逐張解碼全部訓練與驗證圖片，耗時受磁碟速度影響。

合成測試只需 Python、Pillow 和 Matplotlib，可在無 GPU 的環境執行。

```bash
python -m unittest discover -s tests -p 'test_compare_v1_v2.py' -v
```

## 比較條件與輸出

完整驗證集只跑 Clean，另計算排除跨集合像素完全重複後的 Clean。主要比較使用後者，同一批圖片各跑四種獨立干擾。

| 條件 | 參數 |
| --- | --- |
| 旋轉及平移 | 隨機方向旋轉 6–12°，水平與垂直各移動最多 6% |
| 灰階 | 混入 65–85% 灰階，保留部分原始色彩 |
| 縮小加背景 | 居中縮至原寬高的 72–86%，外圍補中性色 |
| 局部遮擋 | 貼上占影像寬高約 27–35% 的矩形，約遮住總面積 7–12% |

預設種子為 2026。每張圖的參數由種子、相對路徑及條件決定。先套干擾，再使用原始 `Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize`。同批處理後的張量送入兩個模型。圖例顯示 resize 和 crop 後模型會看到的畫面。跨環境若 Pillow、torchvision、GPU 或 CUDA 版本不同，像素與推論數字可能略有差異。

- `comparison.csv` 列出樣本數、兩版準確率、V2 減 V1 的百分點、預測差異比例、相對 Clean 的下降與兩版推論時間。
- `comparison.json` 保存相同數值，以及切分、類別順序、checkpoint 資訊、干擾設定、去重統計與執行環境。
- `comparison.png` 顯示各條件準確率與下降幅度。 `examples.png` 顯示固定種子選出的同一組驗證圖及四項干擾。

## 如何讀結果

- `v2_minus_v1_pp` 為正表示 V2 在該條件下答對的比例較高。下降幅度以**去重後的 Clean** 減去同一子集的干擾準確率。正值表示下降，負值表示干擾下的結果較高。兩列 Clean 都沒有下降值。
- 模型預測不同只代表分類結果改變。搭配同列的兩版準確率判斷改變是否提高答對比例。
- 去重比對解碼後的 RGB 像素及尺寸，無法偵測裁切、重新壓縮或相鄰影格等近似重複。模型保留原始訓練歷史。完整 Clean 與去重後 Clean 的差值也可能反映兩個樣本集合本身的難度。
- 這些干擾使用原有驗證圖片與標籤，無法重現競賽測試集的全部分布差異。不要把結果當成 Kaggle Public Score 的估計。
- 時間以 GPU 預熱後同步計時，累加各批次的模型前向運算。圖片讀取、前處理、傳輸與預測後處理不計入。兩個模型固定依 V1、V2 順序執行，細微速度差可能受 GPU 負載與執行順序影響。
