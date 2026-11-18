# Simpsons Train／Test Distribution Audit

## 目前證據

從 367 張附件圖片實際重跑特徵。Train 是每類等量抽樣的 200 張，不能用這組抽樣推斷真實類別頻率。Test 的 100 張隨機圖與 67 張 V3 改判圖分開解讀。

| 樣本 | 實際格式 | 外圈未達一半近黑 | 縮圖亮度中位數 | 高頻細節中位數 |
| --- | --- | ---: | ---: | ---: |
| Train 200 張 | 200 JPEG | 1／200 | 0.292 | 0.00392 |
| 隨機 Test 100 張 | 100 PNG，原副檔名全為 .jpg | 96／100 | 0.466 | 0.02745 |
| V3 改判 67 張 | 67 PNG，原副檔名全為 .jpg | 64／67 | 0.503 | 0.03922 |

用這 200 張 Train 擬合分箱，隨機 Test 有 97／100 張落在 Train 樣本未出現的視覺組合。最常見的兩組都少黑邊、呈彩色、偏亮且高頻細節偏高，對比分別偏低（35 張）與中等（23 張）。這是小樣本的組合支持率。完整 Train 的覆蓋必須由伺服器稽核確認。隨機 Test 高偏移圖中，6171、7126、9107 和 2842 可見顆粒狀干擾，有些其他高分圖則主要是木紋或細密背景。代理值同時會對這兩類訊號升高。V3 改判組的選取依據是 V3 Multi-scale 相對 Baseline，與此工具比較的 V3／V5 改判不同。

V5 相對 V3 的 Public Score 上升 5.654 個百分點，與優先處理背景的方向相容，但不能單獨歸因於背景，也無法從無標籤 Test 推算哪些改判是救回。

## 伺服器執行

沿用現有 .venv，需 Pillow、NumPy、Matplotlib。全程 CPU，預設 4 個 worker。先檢查 96,929 張 Train 是否和原 V2 split 一致，並核對 V2、V5 的 50 類順序。用 Submission 範本的 10,791 個唯一字串 ID 接合兩份實際預測 CSV。範本的角色值不參與分析。Train 的目錄與 classes.json 是合法候選角色的依據，兩份 CSV 也會逐行檢查角色名稱。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
git pull --ff-only origin main
.venv/bin/python -u audit_distribution.py
```

預設 V3 路徑為 `runs/v3_tta_155/submission_v3_multiscale.csv`，V5 為 `runs/v5_background/submission_v5.csv`。若正式提交使用其他路徑，傳入 `--v3-submission`、`--v5-submission`；V5 的 `settings.json` 在別處時一併指定 `--v5-run`。已存在的輸出不會被覆蓋，可用 `--output runs/distribution_audit/recheck` 另存。粗估需約 5–20 分鐘，取決於 CPU 與磁碟，未在 Server 實測。可用 `--workers 0` 改成單進程。

## 指標與解讀

- 檔頭格式、尺寸和副檔名吻合率以可辨識檔頭為分母。影像視覺指標以可解碼圖片為分母。CRC 或像素讀取出錯的檔案另外列出，未納入視覺排名；V3／V5 的整體改判率仍涵蓋完整的 10,791 個 Test ID。
- 近黑定義為 RGB 各通道皆 ≤24。原始最外一圈近黑比例 ≥50% 稱為多數近黑外圈，另量測外側 5% 的區帶。這是可重現的黑邊量測，沒有分割完整背景或角色。
- 亮度與對比為縮圖灰階的平均與標準差。低色度像素是通道差 ≤5，另報中央區域。高頻細節是縮圖的絕對拉普拉斯中位數，混合反映雜訊、木紋、銳利邊緣和壓縮痕跡。照片背景、角色大小、偏心、透視、多角色、反色和裁切由人工複核。`manual_review.csv` 有對應拼圖的空白欄位。
- **技術支持**分別核對實際格式、精確寬高與兩者組合。**視覺支持**由黑邊比例、低色度、亮度、對比與高頻細節的粗組合判斷。亮度／對比／高頻門檻用 Train 第 20、80 百分位；組合頻率為零記為未見，非零且 ≤0.1% 記為少見。這些是可解釋的群組，不能自動判定做過哪種增強。
- 視覺排序針對超出 Train p01–p99 的特徵，將超出量除以 Train IQR（下限 0.01），各分量上限 8，再加總。格式和尺寸沒有進入視覺分數。`upper_tail` 使用 Test 分數的第 90 百分位作為報告分組，同分會保留在一起。擬合門檻不使用 Test，未訓練域分類器或使用 Test 標籤。
- 改判報告提供組內改判率、占全部改判比例、相對全體的比率。組別可重疊，也可能受樣本數與群組關聯影響。模型預測僅用於差異對照，不能當作真實角色出現次數。沒有 Test 真實標籤，無法判定是否有新角色或哪版答對。

## 產物與下一步

`runs/distribution_audit/summary.json` 是最小回傳資料，包含類別不平衡、Train／Test 量測、主要罕見組合、視覺上尾的改判率及優先核對項。詳細資料在 `train_class_counts.csv`、`format_size_counts.csv`、`feature_quantiles.csv`、`feature_combinations.csv`、`v3_v5_by_group.csv`、`test_novelty.csv`、`test_visual_outliers.png` 和 `manual_review.csv`。必要時用 `unreadable_images.csv` 查缺漏。

先看完整資料是否仍以少黑邊及高頻細節偏移為主，再用高分和隨機拼圖分辨背景紋理與真正雜訊。若 V3／V5 改判在其中一組聚集，下一個有標籤對照只測該機制，例如原圖與溫和去噪或保留動畫區塊。留意改判率在其他組是否同樣高。只有取得對照證據後才決定下一版，這個工具不重訓模型、不產生或提交 Kaggle Submission。
