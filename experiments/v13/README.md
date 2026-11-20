# V13｜同預算 Tiny／Base 部分微調

V12 的官方 Base 權重、凍結骨幹與分類頭更新已在 Server 確認。V12 Head 的停損紀錄保留。V13 比較官方 ImageNet-1K ConvNeXt-Tiny 與 ConvNeXt-Base，兩者從第一步都更新最後一個 stage 及 50 類分類頭。V8 `convnext_all_layers/best.pt` 與已封存的十視角、選擇性去顆粒 Submission 一併核對，作為部署效益對照。

## 固定設定

- Train 固定樣本、原 V8 增強方式及每階段種子相同。BF16、預設 batch 8、累積 8、有效 batch 64。AdamW、weight decay `1e-4`，分類頭 LR `1e-3`、最後 stage LR `1e-4`。兩臂關閉 stochastic depth，減少模型原本不同的正規化強度對短跑的干擾。這個實驗比較兩套官方預訓練模型在同預算下的表現，預訓練差異也會影響結果。
- 最小 smoke 每臂 **2 次實際 optimizer update／128 張**。檢查最後 stage 與分類頭選定張量有非零有限梯度、權重更新，並確認凍結 stem 的張量不變。Smoke 計時上限每臂 **120 秒**，使用獨立初始化。
- Pilot 每臂 **100 次實際 update／6,400 張**（每類 128 張），兩臂從同一來源規格的官方權重各自重新初始化。獲准延長後，每臂用另一批互斥的 6,400 張再跑 **100 次**，續用各自 pilot 權重及 optimizer moments。每臂、每個訓練階段上限 **12 分鐘**，每次 optimizer 更新檢查是否超時。若任一臂超時即停，避免用不同完成度比成績。下載、模型初始化、驗證及寫檔時間分開。
- 兩臂各階段用同一份 V8 Screen／Confirm，加上固定 Heldout Composite。如果有匹配的 V11 Audit，Heldout 使用其近似去重子集。報告會顯示使用的子集。三組評估都用同樣的 FP32 十視角與選擇性去顆粒。報告包括 Clean、原 Hard、Heldout、每條件準確率、配對效果及區間、實際更新數、訓練秒數與峰值 VRAM。Heldout 仍是合成影像，改善需透過實際 Test 驗證。

官方權重載入前驗證完整 SHA-256。會優先查找本機 V8 Tiny 及 V12 Base 的已下載權重，並驗證雜湊。也可指定 `--tiny-weights`／`--base-weights`。使用之前先確認競賽與作業均允許官方 ImageNet-1K；雲端仍無法核實 Kaggle Rules。「確認允許」以 `--allow-imagenet` 表示。 [官方權重與訓練來源](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/convnext.py) · [Kaggle Rules](https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules)

## 停損與 Confirm

- **Pilot 或 Extend 入選。** 相對 V8 的 Hard ≥ **+0.75 pp**、Heldout ≥ **+1 pp**，兩者配對 95% CI 下界均 >0，Clean 降幅 ≤ **0.5 pp**，任一原 Hard 條件降幅 ≤ **4 pp**。若要選 Base，還須相對同預算 Tiny 在 Hard 或 Heldout 任一項 ≥ **+1 pp** 且該項 CI 下界 >0，另一項及 Clean 降幅各 ≤ **0.5 pp**。Base 未證明額外優勢時，保留達標的 Tiny 作為較省算力的候選。
- **只允許一次加跑。** Pilot 無入選候選時，兩臂 Clean 須 ≥ **70%**。Base 與 V8 的 Hard 差距最多 **15 pp**、Heldout 最多 **18 pp**；相對同預算 Tiny 的 Hard 或 Heldout 須至少一項 ≥ **+1 pp**，另一項降幅 ≤ **3 pp**、Clean 降幅 ≤ **2 pp**。全部通過才輸出 `advance_to_extend`。
- **Confirm 固定一個候選。** 相對 V8 的 Hard 與 Heldout 都須 ≥ **+1 pp** 且 CI 下界 >0，Clean 與其他條件的護欄相同。若入選 Base，同時用未再調參的同預算 Tiny 複核 Base 優勢。任何失敗即停止。輸出 `confirmed_advantage` 表示值得考慮下一步；此實驗止於驗證報告。

## Server 指令

同步 pinned 程式碼到新目錄。工作目錄維持專案根目錄，直接沿用既有 `.venv`、資料和實驗產物。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
V13_SHA='<本次 Commit 完整 SHA>'
V13_CODE="$PWD/runs/v13/code-$V13_SHA"
mkdir -p runs/v13
if [ ! -d "$V13_CODE" ]; then
  (set -eo pipefail
   v13_tmp="$(mktemp -d "$PWD/runs/v13/.sync.XXXXXX")"
   trap 'rm -rf -- "$v13_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/$V13_SHA" |
     tar -xz --strip-components=1 -C "$v13_tmp"
   mv -T -- "$v13_tmp" "$V13_CODE")
fi
.venv/bin/python -u "$V13_CODE/classification_v13.py" prepare
```

先確認可用 GPU，待 `probe` 顯示 ready、規則確認後，執行 smoke。

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' V13_GPU
export CUDA_VISIBLE_DEVICES="$V13_GPU"
.venv/bin/python -u "$V13_CODE/classification_v13.py" probe &&
.venv/bin/python -u "$V13_CODE/classification_v13.py" smoke --allow-imagenet
```

Smoke 回報 `smoke_pass` 後執行 `pilot --allow-imagenet`。只有 `advance_to_extend` 才執行 `extend --allow-imagenet`，只有 `advance_to_confirm` 才執行 `confirm --allow-imagenet`，其餘旗標沿用。相同設定重跑路徑可重用已完成的階段。出現 OOM 可在 `--output runs/v13/batch4` 另存，從 prepare 以 `--batch-size 4 --eval-batch-size 8` 預備，梯度累積自動增至 16。比較兩臂維持一致。

| 階段 | 規劃時間 | 規劃峰值 VRAM |
| --- | --- | --- |
| CPU prepare／probe | 1–3 分鐘／數秒 | 卷積 probe 用少量 VRAM |
| 雙臂 smoke | 約 1–4 分鐘，另計初次下載 | 6–16 GiB |
| Pilot 或 Extend | 雙臂訓練合計最多約 24 分鐘，Screen 另估 5–18 分鐘 | 6–16 GiB |
| 獲准後 Confirm | 約 5–18 分鐘 | 5–12 GiB |

時間與顯存為單張 RTX 5090 的保守規劃，實際以 Server 紀錄為準，建議預留約 4 GiB 磁碟。所有新報告及 checkpoint 放 `runs/v13/`，不需要重新安裝 PyTorch。最少回報終端的 `smoke_pass` 與兩臂梯度／權重 delta、Pilot 最後一段 JSON 的 `decision`、`comparisons` 及 `training`。若只有短時間 GPU，先做 smoke 即可保留診斷結果。
