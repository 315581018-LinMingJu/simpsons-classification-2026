cd /usr1/Aa315581018/simpsons-classification-2026
export CUDA_VISIBLE_DEVICES=4
E=".venv/bin/python -u runs/report/code-8a108e7910e75534f61c0c9171622d9fa9812615/report/evaluate_conditions.py v20"
( .venv/bin/python -u runs/report/code-8a108e7910e75534f61c0c9171622d9fa9812615/final/classification_v20.py train --backgrounds data/backgrounds/dtd/dtd/dtd/dtd/images --workers 40 --epochs 10 --p-testify 0 --output runs/v20/ablation_no_testify &&
  $E runs/v20/ablation_no_testify/best.pt --name ablation_no_testify --backgrounds data/backgrounds/dtd/dtd/dtd/dtd/images &&
  $E runs/v20/main/best.pt --name v20_main --backgrounds data/backgrounds/dtd/dtd/dtd/dtd/images &&
  $E runs/v20/b288/best.pt --name v20_b288 --backgrounds data/backgrounds/dtd/dtd/dtd/dtd/images &&
  $E runs/v20/large/best.pt --name v20_large --backgrounds data/backgrounds/dtd/dtd/dtd/dtd/images ) 2>&1 | tee runs/report/b_log.txt
echo "EXIT=${PIPESTATUS[0]}" | tee -a runs/report/b_log.txt
