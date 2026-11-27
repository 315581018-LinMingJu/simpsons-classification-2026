cd /usr1/Aa315581018/simpsons-classification-2026
export CUDA_VISIBLE_DEVICES=2
( .venv/bin/python -u runs/report/code-8a108e7910e75534f61c0c9171622d9fa9812615/baseline/baseline.py train --train-dir data/extracted/train --output runs/report/baseline_resnet18 --workers 16 &&
  .venv/bin/python -u runs/report/code-8a108e7910e75534f61c0c9171622d9fa9812615/report/evaluate_conditions.py baseline runs/report/baseline_resnet18/best.pt --name baseline --backgrounds data/backgrounds/dtd/dtd/dtd/dtd/images ) 2>&1 | tee runs/report/a_log.txt
echo "EXIT=${PIPESTATUS[0]}" | tee -a runs/report/a_log.txt
