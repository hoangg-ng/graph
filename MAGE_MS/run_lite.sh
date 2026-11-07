#!/bin/bash
# 2026-11-07: slim MAGE (arm 'lite': graph experts -> c_b * sensor mean), same recipe/GPU/precision as the MAGE baselines
cd "$(dirname "$0")"
for job in "PEMS04 0" "PEMS04 1" "PEMS08 0"; do
  set -- $job
  n=${1}_T96_lite_s$2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python train.py --dataset $1 --seq_len 96 --arm lite \
     --seed $2 --tf32 1 --data_dir data_w288 --out results/lite/$n.json > logs/lite/$n.log 2>&1 &
done
wait; echo ALL DONE
