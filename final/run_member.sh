#!/bin/bash
# 訓練一個 V20 成員 → 預測測試集 → 計算驗證集機率（供 V21 融合）。
# 從專案根目錄執行：bash final/run_member.sh <GPU> <名稱> <timm 模型> <解析度> <epochs> <batch>
# 例：bash final/run_member.sh 2 b288 convnext_base.fb_in22k_ft_in1k 288 12 64
set -u
CODE=${CODE:-final}
BG=${BG:-data/backgrounds/dtd/dtd/dtd/dtd/images}
export CUDA_VISIBLE_DEVICES=$1
( .venv/bin/python -u $CODE/classification_v20.py train --backgrounds $BG --workers 40 \
    --model $3 --size $4 --epochs $5 --batch $6 --output runs/v20/$2 &&
  .venv/bin/python -u $CODE/classification_v20.py predict --workers 40 --output runs/v20/$2 &&
  .venv/bin/python -u $CODE/classification_v21.py valprobs runs/v20/$2 --workers 40 --backgrounds $BG ) \
  2>&1 | tee runs/v20/$2_log.txt
echo "EXIT=${PIPESTATUS[0]}" | tee -a runs/v20/$2_log.txt
