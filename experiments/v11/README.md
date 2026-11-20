# V11｜角色區域檢索的 Train／Validation 小試

V11 凍結 `runs/v8_full/models/convnext_all_layers/best.pt`，並核對 `runs/v8_full/candidates/convnext_tta_denoise/` 內的 V8 Submission 及其 `settings.json` 雜湊。V8 CSV 只有標籤，只用來確認產物身分。原多尺度及選擇性去顆粒的 V8 分類輸出在本次診斷圖上重新取得。現有程式有 RGB 完全重複檢查及 V9 的分類機率快取，未有可直接使用的局部特徵或幾何匹配。V11 共用原切分、合成場景、Normalizing 與推論碼。

競賽 Rules 頁的存取仍遭代理拒絕，作業條文無法從 repo 確認。本版入口只有 `audit` 和 `pilot`，索引及查詢範圍限定競賽 Train／Validation。原 V8 Test CSV 只計算檔案雜湊。本版能產出的只有診斷；若要擴到任何 Test 操作，需另核對競賽及作業規則。使用本機 V8 的 ImageNet 衍生權重仍沿用 `--allow-imagenet` 的授權確認。[競賽 Rules](https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules)

## 資料與匹配

- CPU Audit 讀取 V8 原分割的 77,547 張 Train，計算解碼 RGB SHA-256 及裁除近黑邊後的雙向 128-bit 差分雜湊。原圖及水平翻轉任一雜湊距離 ≤12 時保守排除。比對涵蓋全部 V8 Train，也排除兩個 Val 階段所選圖片間的近似重複。Train 檔名未提供可靠的原始影格群組，因此來源隔離屬於近似檢查。大幅裁切或變形的同源圖仍可能漏過。
- 索引取每類至多 40 張合法 Train（約 2,000 張），從黑框遮罩保留的內容取得 ConvNeXt 第三 stage 向量及第二 stage 的局部描述。查詢使用固定全幅與四個重疊局部區域找近鄰，分類對照保持原 V8 十視角。只有分散的雙向局部近鄰能通過縮放、旋轉、平移一致性，才可接受；不同類別幾何票數接近時保留 V8。
- Screen 和 Confirm 各有約 100 張**同源正對照**及 150–200 張 **Val 有標籤查詢**。每個原圖只生成一種固定干擾，交替測偏心背景與多角色干擾。同源正對照的原圖在索引內。對同一正對照另做**目標來源缺席的負對照**，查詢前移除整個正確角色類別，故任何接受都屬於誤接受。Val 保留原角色類別在索引內，檢驗近似來源隔離時的實際分類救回與退步。
- 每條覆蓋、來源找回與誤接受各以對應組別為分母。同源找回要求相同來源 ID，匹配到其他同角色圖算未找回。合成結果只能用於判定是否值得擴大診斷。

## 停損

CPU Audit 要求每類每個 fold 至少 3 張隔離 Val 與 2 張不同的同源正對照。Screen 比較固定的三組門檻：

| 門檻 | 最少內點 | 內點比例 | 兩圖中內點分布的最短邊 | 向量 cosine |
| --- | ---: | ---: | ---: | ---: |
| balanced | 5 | 0.48 | 0.16 | 0.58 |
| conservative | 7 | 0.55 | 0.20 | 0.64 |
| strict | 9 | 0.60 | 0.23 | 0.70 |

入選門檻必須同時達到 **來源找回 ≥15%、負對照誤接受 ≤2%、Wilson 95% 上界 ≤6%、隔離 Val 接受率 ≥5%、配對提升 ≥1 pp 且至少救回 3 張**。依 Screen 配對提升選出一組，再依誤接受數及嚴格程度決定同分。將門檻鎖在 `screen.json` 後才執行 Confirm。Confirm 沿用全部限制，另要求配對 95% CI 下界 >0。未通過即停止。全數通過只支持擴大有標籤診斷；擴大前應抽查改判的圖塊對應，確認分割與不同來源判斷。

## Server 執行

在專案根目錄執行。使用這次 GitHub Commit 的完整 SHA，curl 的來源快照解到獨立 V11 目錄。既有 V8 資料與 `.venv` 照原路徑讀取。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
V11_SHA='<本次 GitHub Commit 完整 SHA>'
V11_CODE="$PWD/runs/v11/code-$V11_SHA"
mkdir -p runs/v11
if [ ! -d "$V11_CODE" ]; then
  (set -eo pipefail
   v11_tmp="$(mktemp -d "$PWD/runs/v11/.sync.XXXXXX")"
   trap 'rm -rf -- "$v11_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/$V11_SHA" |
     tar -xz --strip-components=1 -C "$v11_tmp"
   mv -T -- "$v11_tmp" "$V11_CODE")
fi
.venv/bin/python -u "$V11_CODE/classification_v11.py" audit
```

Audit 顯示 `ready` 才使用一張經授權的 RTX 5090。CPU 做小矩陣相似度時限制 BLAS threads，減少執行緒成本。

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' V11_GPU
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$V11_GPU" \
  .venv/bin/python -u "$V11_CODE/classification_v11.py" pilot --allow-imagenet
```

CPU Audit 用 4 workers 約 **3–15 分鐘**。小試索引加 Screen 約 **5–20 分鐘**，若進 Confirm 再約 **3–12 分鐘**；峰值 VRAM 規劃 **4–8 GiB**（沿用 V8 封存 batch 16 時）。估計受共享 I/O、CPU 與實際批量影響。

回傳 Audit 的 `decision`、`counts`、`exclusions`。Pilot 回傳 `decision`、Screen 各門檻與 Confirm（如有）的 `exact_source_recall`、`absent_label_false_acceptance`、`absent_label_wilson95_upper`、`isolated_validation_coverage`、`wins`、`losses`、`gain`，以及峰值 VRAM。程式將雜湊、種子、配對清單及決策寫到 `runs/v11/audit.json`、`screen.json`、`pilot.json` 與對應 evidence。
