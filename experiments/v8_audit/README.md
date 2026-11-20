# V8 Test Domain Shift 診斷

此工具在 CPU 上核對已封存的 V8 `convnext_all_layers/best.pt`、`convnext_tta_denoise/submission_v8.csv` 及原始 V8 benchmark。產物集中在 `runs/v8_domain_audit/`。Kaggle Rules 頁在 Codex 的存取回傳 403，repo 也沒有可核實的作業條文。本工具把 Test 分析限制於一般影像統計與雜湊可核對的既有預測。只使用 Train／Val 原標註，Test 欄位保留預測角色與測量值。程式完全以 CPU 執行。 [Kaggle Rules](https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules)

## 統計及來源

- **Train** 是原 V8 Train manifest 中固定抽樣的 1,000 張。**Validation** 使用原 V8 Screen 樣本。Clean 用原圖，另以固定種子還原可查到預測的 Hard／Heldout 場景。如 V12／V13 的 Heldout 使用 V11 Audit，只有檔案雜湊及影像清單相符才重用。這些都是有標籤的資料。
- **Test** 優先重用全量 `test_novelty.csv`，先核對資料來源、合法類別與全部 Test ID，再抽 16 張核對目前檔案的尺寸與影像量測。舊稽核不相容時改採固定均勻樣本。新增的空間代理用固定抽樣 768 張，另可從有可靠 V9 預測的 ID 中抽最多 768 張計算重疊群組。報告分開列出整體、均勻樣本及預測重疊群的分母。
- **影像特徵** 包含尺寸、長寬比、近黑外圈、亮度、細節與外圍 20% 的亮度梯度活動量。後兩者可能同時反映紋理、邊界及噪音。幾何裁切保留率只描述整幅畫面。在縮圖上以邊緣能量估計偏心、集中程度、中心裁切以外的活動，以及分隔的活動峰數。這些數值只能做人工檢視排序，角色位置、大小、遮擋與數量仍未知。
- **預測統計** 一律以完整 V8 CSV 計數，Train 真實標籤和 Test 預測各自標示。只有 V9 `diagnostic/views.npz` 的模型、原提交、範本、圖片、角色順序與視角定義均核對後，才計算平均十視角 Softmax 的最大機率、Top-1／Top-2 差、自然對數熵、視角分歧、全一致比例與兩兩視角一致率。與 V8 原 CSV 改判的 ID 單獨列出，主要機率分布只用重跑 Top-1 與原 CSV 一致的列。視角矩陣採原 V9 重跑之樣本。
- **快取不足** 時，若 V9 的逐圖 CSV 可由封存及分組摘要核對，讀取其中的信心、差值及原分歧指標。熵和兩兩視角矩陣由逐視角機率快取提供。缺少可用預測時，CSV 欄位留空，報告列覆蓋數為 0。全量 V8 重跑是 107,910 次圖片視角前向。成本從 V9 或近似的 V8 Screen 既有計時線性外推，缺計時則列不同吞吐率下的情境，硬體與 I/O 細節會影響實際時間。
- **Validation 風險** 用 V9 的同模型 Screen 快取或 V13／V12／V10／V8 的既有 V8 逐圖預測。每個條件內比較有風險旗標的一組與其他組的實際錯誤、分母和 Wilson 95% 區間。兩組都至少 20 張、總錯誤至少 5 張且風險組區間下界大於其他組上界時標作探索訊號。群組可重疊，條件重複使用同一原圖，角色組成也可能影響差異。這些資料適合排定核查優先序。

Codex 用已提供的 200 張 Train 與 100 張均勻 Test 樣本做 CPU 量測。Test 的亮度中位數為 0.4657，Train 為 0.2918。288 尺度中心裁切外活動量的中位數分別為 35.4% 與 21.5%。分隔活動峰旗標的比例兩邊都是 74%，顯示它可能只是反映共同的邊緣結構。這批附件沒有對應的 V8 機率或可用的原 Screen 標籤配對，實際的錯誤群組須從 Server 封存產物取得。

## 執行

工作目錄為現有專案根目錄。使用這次 pinned Commit，將程式快照解到獨立路徑，適用於無 Git checkout 的 Server。沿用 .venv 的 Pillow 和 NumPy。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
AUDIT_SHA='<本次 Commit 完整 SHA>'
AUDIT_CODE="$PWD/runs/v8_domain_audit/code-$AUDIT_SHA"
mkdir -p runs/v8_domain_audit
if [ ! -d "$AUDIT_CODE" ]; then
  (set -eo pipefail
   audit_tmp="$(mktemp -d "$PWD/runs/v8_domain_audit/.sync.XXXXXX")"
   trap 'rm -rf -- "$audit_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/$AUDIT_SHA" |
     tar -xz --strip-components=1 -C "$audit_tmp"
   mv -T -- "$audit_tmp" "$AUDIT_CODE")
fi
.venv/bin/python -u "$AUDIT_CODE/audit_v8_domain.py"
```

CPU 規劃 **1–5 分鐘、4 workers、0 GPU**，共享 I/O 較慢時耗時可能增加。若 Server 上的 V9、V11 Audit 或近期評估目錄不同，可用 `--v9-run`、`--v11-audit`、`--v12-run` 或 `--v13-run` 指向原產物。新的輸出取 `--output runs/v8_domain_audit/recheck` 等子目錄。

## 產物及最少回傳

| 檔案 | 內容 |
| --- | --- |
| `summary.json` | 影像分布、V8 預測涵蓋率、兩兩視角矩陣、視覺群組的分數分布、Validation 錯誤與成本外推 |
| `prediction_classes.csv` | 50 類 Train 真實數量及 Test 預測數量 |
| `risk_groups.csv` | 各群組可量測數、比例及 Validation 錯誤與區間 |
| `test_risk.csv` | 全部 Test ID、原 V8 預測、可量測欄位及風險旗標 |
| `review.csv` | 少量 Test 隨機與風險案例，及有標籤的 Val 錯誤清單。合成條件列出原圖路徑及條件 |

請貼終端最後一段 JSON。優先看 `test_scores`、`score_coverage`、影像中位數、`validation_errors` 與探索訊號。Test 信心變低或視角有分歧只能反映穩定性；與有標籤群組對照後再選下一步。
