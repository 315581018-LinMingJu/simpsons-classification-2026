# V5 外圍背景替換實驗

## 真實圖樣本觀察

附件共有 367 張可解碼的圖片，ZIP 約 24.66 MiB。訓練組按角色等量抽樣，固定種子並依 ZIP 容量縮減配額。測試隨機組與 V3 Multi-scale 改判組的 ID 不重疊。

| 組別 | 張數 | 尺寸範圍（寬／高） | 影像格式 | 最外圈至少一半接近黑色 |
| --- | ---: | --- | --- | ---: |
| 訓練，每類 4 張 | 200 | 224／224 | 200 JPEG | 199 |
| 隨機測試 | 100 | 244–284／244–284 | 100 PNG | 4 |
| V3 Multi-scale 改判 | 67 | 244–282／244–284 | 67 PNG | 3 |

近黑定義為 RGB 各通道均 ≤24。逐圖先計算外圈近黑比例，再取中位數，三組分別為 81.05%、0.34%、0.48%。隨機測試樣本有夾在木紋、牆面、室內照片等背景中的小動畫區塊，也可看到偏移、旋轉、裁切、整塊反色或灰階、局部雜訊，以及可見多個角色的畫面。樣本寬高比約 0.87–1.13，因此僅靠寬高比不易區分這種變化。角色大小與位置未做精確框選，改判組用來找案例，沒有用於估計全體測試圖出現比例。測試無真實標籤，格式差異和視覺差異也不能直接解釋 Kaggle 失分。

## 單一改動與驗證

V5 從現有 V2 ResNet18 最佳 checkpoint 起跑。原 V2 增強之前，約 60% 的訓練輸入嘗試將與邊界相連的外圍近黑像素換成程序紋理。只處理估計覆蓋面積介於 7% 到 70% 的遮罩，保留遮罩外的原始像素。在附件的 200 張訓練圖上，196 張有合格遮罩，中位覆蓋面積為 34.25%。替換只在記憶體中進行。此實驗不使用測試圖當素材，不調整既有 V3 Multi-scale 推論，也不加入反色或其他資料增強。程序紋理只能模擬部分背景變動，連通遮罩也不是語義分割。

用原切分的去重驗證圖，每類至多 24 張，在相同的干擾圖上比較 V2 與 V5。兩者均使用原 V3 Multi-scale。驗證另用訓練未使用的接縫紋理，並測 Clean。報告保留每種條件的救回、退步、micro 準確率與按類等權的配對 bootstrap 95% 區間。只有干擾組改善的區間下限高於 0，且 Clean 的 macro 準確率沒有下降超過 0.5 個百分點，才判為支持。模型固定用最後一輪。V5 多做了兩輪更新，缺少同等步數的無替換微調對照，因此成績差不能完全歸因於背景替換。驗證收益也無法推算 Kaggle 分數。

## RTX 5090

沿用現有 .venv、PyTorch、CUDA 和資料路徑。先確認一張 GPU 可用，確定可使用後再輸入 index。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
source .venv/bin/activate
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
read -r -p "已確認可使用的 GPU index：" gpu_index
CUDA_VISIBLE_DEVICES="$gpu_index" python -u classification_v5.py train
```

預設 AdamW、LR 3e−5、兩輪、batch size 32、workers 4、FP32、seed 2026。推論每次只送一個視角的 batch 到 GPU。估計單張 RTX 5090 微調加驗證需 25–90 分鐘、4–10 GiB VRAM，未在伺服器實測。若 V3 去重清單不存在或資料已變，另預留約 3–15 分鐘重新掃描。資料或 V3 結果不在預設目錄時，可傳入 `--data` 或 `--v3-run`。重跑使用新的 `--output runs/v5_...` 以保留舊跑次。共享記憶體受限可降低 `--workers`，VRAM 受限可降低 `--batch-size`。

先回傳 `runs/v5_background/validation.json` 即可。若結果支持假設，再使用同一個模型產生候選。程式只輸出已完成跑次的模型預測，驗證 10,791 個唯一 ID 與原模板順序。要不要提交 Kaggle 由使用者決定。

```bash
CUDA_VISIBLE_DEVICES="$gpu_index" python -u classification_v5.py predict
```

所有輸出在 `runs/v5_background/`，包含設定、來源 checkpoint 雜湊、每輪權重與耗時、配對預測與候選檔 `submission_v5.csv`。V1–V4 產物及原訓練／測試檔案保持原樣。
