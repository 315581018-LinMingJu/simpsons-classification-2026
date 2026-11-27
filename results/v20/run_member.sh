#!/bin/bash
# 用法：run_member.sh <GPU> <名稱> <模型> <解析度> <epochs> <batch>
cd /usr1/Aa315581018/simpsons-classification-2026
CODE=runs/v20/code-b539d567
BG=data/backgrounds/dtd/dtd/dtd/dtd/images
export CUDA_VISIBLE_DEVICES=$1
( .venv/bin/python -u $CODE/classification_v20.py train --backgrounds $BG --workers 40 \
    --model $3 --size $4 --epochs $5 --batch $6 --output runs/v20/$2 &&
  .venv/bin/python -u $CODE/classification_v20.py predict --workers 40 --output runs/v20/$2 &&
  .venv/bin/python -u $CODE/classification_v21.py valprobs runs/v20/$2 --workers 40 ) 2>&1 | tee runs/v20/$2_log.txt
echo "EXIT=${PIPESTATUS[0]}" | tee -a runs/v20/$2_log.txt
