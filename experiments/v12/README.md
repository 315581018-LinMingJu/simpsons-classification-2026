# V12｜ConvNeXt-Base 單卡短跑

現有最佳 V8 Public 0.75563。V9 的視角篩選、V10 的場景短微調和 V11 的小索引檢索均已停損。V11 同源找回 2%、Val 覆蓋最多 0.5%、配對增益 −0.5 pp。V12 固定現有部署方式及短跑預算，測試容量較大的 ConvNeXt 能否改善 V8 在亮背景與幾何偏移下的辨識。

只做 **ConvNeXt-Base ImageNet-1K V1**。同系列保留可控的 224 輸入、原 V8 Normalization、增強及推論方式。官方 Base 約 8,859 萬參數，224 輸入 15.355 G ops，V8 Tiny 約 2,859 萬和 4.456 G ops。較大的模型有更多容量用於小角色局部與場景上下文，但兩次短 epoch 可能仍無法充分適應。ConvNeXt-Base 來源及下載檔案已核對，載入任何本機拷貝前也會比對**完整 SHA-256**。 [torchvision ConvNeXt 官方來源、訓練資料及權重](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/convnext.py)

Kaggle Rules 的存取仍遭代理 403，repo 裡也沒有可核實的作業條文。torchvision 程式碼為 BSD-3-Clause；ImageNet 權重使用還須遵守競賽與作業條款。確認兩者允許後，使用 `--allow-imagenet`。規則不明或權重無法取得時，停在 CPU／GPU 預檢，保留 V8。 [Kaggle Rules](https://www.kaggle.com/competitions/ml-2026-iaii-classification2/rules) · [torchvision LICENSE](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE)

## 同步與環境預檢

以下工作目錄應是已備妥 `.venv`、V8 checkpoint、資料及報告的專案根目錄。用回覆內的完整 Commit SHA，將 pinned 程式碼解到獨立目錄。此方式可在無 Git checkout 的 Server 上使用。

```bash
cd /usr1/Aa315581018/simpsons-classification-2026
V12_SHA='<本次 Commit 完整 SHA>'
V12_CODE="$PWD/runs/v12/code-$V12_SHA"
mkdir -p runs/v12
if [ ! -d "$V12_CODE" ]; then
  (set -eo pipefail
   v12_tmp="$(mktemp -d "$PWD/runs/v12/.sync.XXXXXX")"
   trap 'rm -rf -- "$v12_tmp"' EXIT
   curl -fLsS "https://codeload.github.com/315581018-LinMingJu/simpsons-classification-2026/tar.gz/$V12_SHA" |
     tar -xz --strip-components=1 -C "$v12_tmp"
   mv -T -- "$v12_tmp" "$V12_CODE")
fi
.venv/bin/python -u "$V12_CODE/classification_v12.py" prepare
```

CPU `prepare` 鎖定原 V8 full benchmark、`convnext_all_layers/best.pt` 和 `convnext_tta_denoise/submission_v8.csv`，核對其雜湊及設定。從 Train 固定抽每類 240 張，共 12,000 張。保留同一份 V8 Screen／Confirm。若有匹配的 `runs/v11/audit.json`，額外的 Heldout Composite 會使用它篩過近似重複的子集，不重做 V11 檢索。若 V11 Audit 位於其他位置，所有階段加相同的 `--v11-audit`；若找不到 Audit，Heldout 使用原 V8 驗證圖並在報告標明。 `--weights-file /path/to/official.pth` 可使用已取得的官方副本，CPU prepare 可先核對其 SHA。

先依實驗室規則確認一張可用的卡，再測既有 PyTorch／CUDA 與硬體：

```bash
nvidia-smi
read -r -p '已確認可用的 GPU 編號: ' V12_GPU
export CUDA_VISIBLE_DEVICES="$V12_GPU"
.venv/bin/python -u "$V12_CODE/classification_v12.py" probe
```

`probe` 回報 PyTorch／torchvision、CUDA build、可見 GPU、顯示卡、能力版本和可用 VRAM，並執行一次 BF16 卷積正反向。Codex 預檢實際是 Python 3.12、PyTorch 2.9.1+cpu、torchvision 0.24.1+cpu、CUDA unavailable；RTX 5090 相容性需以 Server 的 `ready` 為準。

## 只跑一次的 GPU 訓練

在規則確認、CPU prepare 及 GPU probe 都完成後執行：

```bash
.venv/bin/python -u "$V12_CODE/classification_v12.py" train --allow-imagenet
```

預設 BF16、batch 8、梯度累積 8、有效 batch 64。各階段沿用 V8 checkpoint 記錄的訓練增強，保持背景、位置與顆粒增強的原設定，只改模型容量與短跑訓練範圍。新模型從官方 Base 初始化，與已完整訓練的 V8 比較時也包含訓練預算差異，因此這個結果量測的是期限內可取得的候選表現。

1. 先凍結 Backbone，分類頭 LR `1e-3` 跑 **1 epoch**。V8 與 V12 都用原十視角和去顆粒，比較 Clean 與 Heldout Composite。若 Clean 低於 **85%**、落後 V8 超過 **8 pp**，或 Heldout 落後超過 **15 pp**，回報 `stop_head`。
2. 通過才從該頭權重解凍最後兩個解析度 stage，再跑 **1 epoch**。分類頭 LR `2e-4`、骨幹 LR `2e-5`。Screen 同時比較原 Hard 五條件、Clean、Heldout Composite。Heldout 使用 V6 的 held-out 合成配方與固定新種子，包含偏心、縮小、亮紋理和透視。它有同一已知角色標籤，仍是合成代理，與真實 Kaggle Test 的效果需另驗證。
3. Screen 要求 **原 Hard +0.75 pp、Heldout +1 pp**、Clean 降幅至多 0.5 pp、每個原 Hard 條件降幅至多 4 pp。通過後用同一 checkpoint 做一次 Confirm。Confirm 把 Hard 要求提高至 **+1 pp**，並要求 Hard、Heldout **各自配對 95% CI 下界 >0**。任一階段失敗即保留 V8。

```bash
.venv/bin/python -u "$V12_CODE/classification_v12.py" confirm --allow-imagenet
```

只有 Confirm 通過，才可選擇執行 `predict --allow-imagenet`。候選用 V8 原推論配方，核對範本的 10,791 個 ID、順序與合法角色。自動產出候選前也會比較 V8 CSV，結果若完全一樣只存統計。實際 Kaggle 提交由你決定。

## 預估資源及最少回報

| 階段 | 規劃時間 | 規劃 VRAM | 只需回傳 |
| --- | --- | --- | --- |
| CPU prepare／GPU probe | 約 1–3 分鐘／數秒 | 卷積 probe 只需少量 | `train_images`、`shift_cohort`、`status`、`gpu`、`torch` |
| Head 與快速比較 | 約 12–40 分鐘 | 5–12 GiB | `decision`、`heldout_effect`、峰值 VRAM |
| 入選後微調與 Screen | 約 25–75 分鐘 | 8–20 GiB | `decision`、`hard_effect`、`heldout_effect`、峰值 VRAM |
| Confirm | 約 8–25 分鐘 | 5–12 GiB | 同 Screen |
| 達標後 Test 推論 | 約 12–40 分鐘 | 5–12 GiB | `images`、`changed_vs_v8` |

時間與 VRAM 皆為估計值，受實驗室共享 GPU、I/O 和原 V8 eval batch 大小影響。預留約 3 GiB 磁碟。OOM 時可在新的 `--output runs/v12/batch4` 用 `--batch-size 4 --eval-batch-size 8`，先跑 prepare；訓練時累積自動增至 16。比較兩臂的 eval batch 始終相同。等設定重跑可重用已完成的一個 epoch 及每條件推論，檢查點保存在 `runs/v12/checkpoints`。
