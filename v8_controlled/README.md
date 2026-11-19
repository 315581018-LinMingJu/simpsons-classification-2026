# V8 單因素受控診斷

V8 Domain Shift Audit 的 Train／Test 中位數為黑邊 81.50%／0.39%、亮度 0.2724／0.4634、細節 0.00392／0.03137、中心裁切外活動代理 18.66%／36.97%。這些是影像差異。此次只用有標籤的 V8 Screen 量測模型對特定處理的反應。

V9 已比較全圖、裁切、縮放與去顆粒等固定配方並在 Screen 停止；部分程式的預設 checkpoint 路徑和目前 V8 部署不同。V10 曾一起變動角色大小、位置與合成背景並微調，整體 Hard 較 V8 低 0.4875 pp。此次在 CPU `prepare` 時讀取 Server 上的 V9／V10 本機對照報告，核對其 V8 checkpoint 與 benchmark 雜湊，缺檔或不同 checkpoint 會另標。這套診斷凍結目前 V8 checkpoint，使用配對的單因素變動。 [V9 方法](https://github.com/315581018-LinMingJu/simpsons-classification-2026/blob/013f6d13387b04983a778cc3fc18c3174c549bc9/v9/README.md) · [V10 方法](https://github.com/315581018-LinMingJu/simpsons-classification-2026/blob/d3cf0b1d88c521dd5ca51071fe49efa352c43624/v10/README.md)

## 固定樣本與處理

從原 V8 去重 Screen 固定抽每類 4 張，最多 200 張，沿用原標籤。每張原圖完整等比縮至長邊 176，置於 280×280 畫布。三種外圍處理共用同一中性灰參考及同一中心原圖塊。各臂套用原 V8 選擇性去顆粒，CPU 會逐像素確認處理及去顆粒的差異都留在指定外圍區域。任何條件破壞隔離便於推論前排除這張的全部臂，且保留排除原因。

| 配對 | 唯一變動 |
| --- | --- |
| 黑邊 | 外側 12 px 的人工邊框，灰 112 改成黑。內嵌原圖自帶的黑邊及原圖像素共同保留 |
| 亮背景 | 背景區灰階 112 改成 174，外框及原圖塊附近保護帶固定 |
| 外圍細節 | 相同背景區內加上 ±22 的小尺度紋理，正負像素數配平，平均強度保持相等 |
| 內容保留 | 同一張偏心畫布、相同 240／288 十視角機率。比較原 V8 全部平均，與只平均原圖塊保留至少 98% 的視角 |
| 等視角數對照 | 與內容保留法使用每尺度相同的視角數，但以固定種子選取、不讀原圖塊位置 |

最後兩項共用一批前向結果。原圖塊邊界由合成的位置已知，另報其中非近黑像素的保留率作角色線索代理。原圖塊可能包含自身背景。這個取景實驗估計「已知原圖塊位置」時的效果，真實 Test 的角色位置仍未知。此次沿用 V8 的視角，專注比較視角中保留的已知內容。

Codex 用先前提供的 200 張 Train 附件做 CPU 幾何煙霧測試。200 張通過隔離；在這套刻意偏心的合成構圖上，平均原圖塊保留率約為 83.6%（十視角）、87.6%（等數量對照）、99.5%（內容保留）。這是測試設計的幾何量測。分類反應須以實驗室的有標籤 Screen 實測。

## 指標與停損

- CPU 每類至少 3 張通過隔離才進推論，其餘情況停在 `stop_insufficient_protected_samples`。中性參考準確率低於 75% 時，所有配對標為 `stop_weak_control`。
- 準確率差以處理減參考的百分點表示，負值代表處理增加錯誤。改判率包含兩邊均答錯的改判。兩者分別以類內配對重抽給 95% CI，並報告救回與退步。
- 準確率至少差 2 pp、名義配對 95% CI 排除 0、五個預先指定對照的 Holm 校正 McNemar p ≤0.05，才標明改善或退步。改判率 ≥5% 且其 CI 下界 >2 pp、準確率仍不確定時另報敏感。其餘停止。區間較寬時只表示本次未取得明確證據。
- 內容保留法相對各自對照的平均原圖塊保留量要增加至少 5 pp。若只比原 V8 好而未通過等視角數對照，保留「選視角數量的效應未拆開」的結果。即使兩項都通過，仍只支持在有標籤資料上追查角色位置及裁切。

## 同步並執行

工作目錄為已有 .venv、V8 和資料的專案根目錄。以本次 Commit 完整 SHA 展開來源快照，適用於無 Git checkout 的 Server。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
FACTORS_SHA='<本次 Commit 的完整 SHA>'
FACTORS_CODE="$PWD/runs/v8_controlled/code-$FACTORS_SHA"
mkdir -p runs/v8_controlled
if [ ! -d "$FACTORS_CODE" ]; then
  (set -eo pipefail
   factors_tmp="$(mktemp -d "$PWD/runs/v8_controlled/.sync.XXXXXX")"
   trap 'rm -rf -- "$factors_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/$FACTORS_SHA" |
     tar -xz --strip-components=1 -C "$factors_tmp"
   mv -T -- "$factors_tmp" "$FACTORS_CODE")
fi
.venv/bin/python -u "$FACTORS_CODE/compare_v8_factors.py" prepare
```

`prepare` 完全用 CPU，規劃約 1–3 分鐘。輸出 ready 後，依實驗室規則選一張可用 GPU，再自行執行：

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' FACTORS_GPU
CUDA_VISIBLE_DEVICES="$FACTORS_GPU" .venv/bin/python -u "$FACTORS_CODE/compare_v8_factors.py" validate --allow-imagenet
```

V8 已有的 ImageNet 權重授權確認沿用 `--allow-imagenet`。最多 **10,000 次圖片視角前向**，暖機另計。三種外圍處理各追加 10×樣本數，一張共用參考只算一次。兩種視角聚合皆使用相同偏心臂的快取，額外前向 **0**。單卡規劃約 **2–8 分鐘、4–8 GiB VRAM**，以 Server 實測為準。程式由原封存器記錄的 eval batch 開始，可用 `--batch-size 8 --workers 2` 降低用量，將新結果另存於 `--output runs/v8_controlled/batch8`。Server 的 V9／V10 路徑不同可傳 `--v9-run`／`--v10-run`。

`runs/v8_controlled/` 保留固定 `cohort.csv`、`prepare.json`、`comparison.csv`、`predictions.csv` 及 `summary.json`。只需回傳 prepare 的 `decision`、`eligible`、`earlier_screens`，和 validate 最後一段 JSON。合成結果與真實 Test 準確率的關係仍待測量。
