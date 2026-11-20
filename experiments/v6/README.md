# V6：短跑次合成場景對照

## 為何優先補這個組合

完整稽核中，少黑邊、彩色、偏亮、高頻細節較多、低或中對比的兩組合計 5,847／10,791 張 Test（54.2%），完整 Train 只有 60＋37 張。Train／Test 的差異集中在共現的視覺訊號。將 PNG 轉成 JPEG，或只增加灰階、旋轉的比例，無法補到這個組合。

V5 相對 V3 的 Public Score 上升 5.654 個百分點，現有最好的實際證據支持沿這條方向測試。舊 Heldout Background 雖然替換了黑框，仍維持動畫區塊的大小、位置與既有色彩。V2 在該條件已達 99.6667%，很少有可救回的錯誤。真實測試圖中可看到小動畫塊、照片背景、輕重不一的透視與顆粒共同出現。V5 也多做了訓練更新，不能把成績增幅全部歸因於背景替換。

最偏移的 Test 上尾改判率 46.76%，高於參考範圍內的 23.24%。這足以把該圖群列為診斷優先，沒有 Test 標籤就無法知道哪一版答對。細節分數同時混合材質、顆粒及邊緣，所以先模擬具紋理的背景，保留角色區塊的內容。局部裁切、反色與多角色放在既有 V5 Challenge 分開研究。本輪沒有加入這些因素，以避免裁掉辨識線索或混入第二個標籤。

## 單一配對實驗

從 V5 最後一輪 checkpoint 分成兩個各一輪的短跑次，固定每類 300 張 Train，共 15,000 張。兩臂都用相同的資料順序、步數、AdamW、LR 3e-5、FP32、batch size 32。控制臂繼續用原 V5 配方。V6 臂約 45% 的輸入改成完整來源圖塊合成，約 15% 沿用外框替換，其餘沿用原 V2 增強。這是一個配方比較，不能分離其中每項變化各自的貢獻。

合成先使用 V5 外框遮罩，將完整來源圖塊以縮小、偏移、旋轉及溫和透視放進程序紋理。亮度、對比與細節程度以完整稽核 Test 的中位數為校準目標，沒有擷取任何 Test 圖像像素，也不讀 Test 標籤。顆粒加在背景，合成路徑跳過二次隨機裁切，仍使用 ToTensor、Normalize 與 Random Erasing。面板遮罩是外圍近黑估計，不是角色分割。程序紋理近似實際背景，照片層次及真實標註規則仍無法完整重現。

CPU 預演可讀 `preview.json` 和拼圖。用附件的 200 張訓練圖試做，合成訓練圖的亮度、對比、高頻細節中位數約 0.461、0.207、0.0275。完整稽核提供的亮度、高頻細節目標是 0.45852、0.03137，這次 CPU 預演的對比使用附件的隨機 Test 中位數 0.210。Server 將從真正的 `summary.json` 讀入完整 Test 的對比中位數。

用原 V5 去重驗證清單，每類至多 16 張，在 Clean 和 Heldout Composite 比較原 V5、V5 控制臂和 V6。Heldout 採訓練未使用的接縫紋理、不同種子、較小圖塊與更大的透視變化，三套權重看相同驗證圖與原 V3 Multi-scale 視角。報告列出逐圖救回、退步、共同錯誤及按類等權的配對 bootstrap 95% 區間。

只有 Heldout 相對原 V5、控制臂的改善區間下限皆大於 0，且 Clean 比原 V5 的 macro 準確率下降不超過 0.5 個百分點，才支持這個配方。若原 V5 或控制臂在 Heldout 的錯誤率低於 5%，標記為接近天花板，避免把小差異當強證據。資料只支援本地機制比較，Kaggle Private 或 Public Score 仍需各自驗證。

## Server 執行

現有 .venv 與資料直接沿用。CPU 預演輸出約需 1–3 分鐘。微調和配對驗證預估單張 RTX 5090 **20–60 分鐘、4–10 GiB VRAM**，尚未實測。確認 GPU 可用後再執行。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
.venv/bin/python -u classification_v6.py preview

nvidia-smi
read -r -p "已確認可使用的一張 GPU index：" gpu_index
CUDA_VISIBLE_DEVICES="$gpu_index" .venv/bin/python -u classification_v6.py train
```

若完整稽核不在預設的 `runs/distribution_audit/summary.json`，傳入 `--audit`。原 V5 跑次在其他位置時傳入 `--v5-run`。記憶體不足可降低 `--batch-size`，共享記憶體不足可調低 `--workers`。重跑請指定新的 `--output runs/v6_...`。

新產物集中在 `runs/v6_composite/`，包括 `preview/` 和 `experiment/`。先回傳 `experiment/validation.json` 即可。若支持 V6，才值得產生一份候選，指令如下。候選使用原模板 ID 順序與 50 類映射。是否提交 Kaggle 由使用者決定。

```bash
CUDA_VISIBLE_DEVICES="$gpu_index" .venv/bin/python -u classification_v6.py predict
```

不會改寫 V1～V5 權重、資料、既有 Submission 或報表。
