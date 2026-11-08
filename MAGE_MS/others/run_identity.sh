#!/bin/bash
# 2026-11-08: identity test (average + sensor embedding) and random-graph test, Graph WaveNet and ASTGCN, seed 0
cd "$(dirname "$0")"; mkdir -p logs results
L(){ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 ../.venv/bin/python ${@:2} > logs/$LOG.log 2>&1 & }
for ds in PEMS04 PEMS08; do
  LOG=astgcn_${ds}_s0_mean_emb; L 0 train_other2.py --model astgcn --dataset $ds --seed 0 --mixing mean --sensor_emb 1
  LOG=astgcn_${ds}_s0_random;   L 4 train_other2.py --model astgcn --dataset $ds --seed 0 --mixing random
  LOG=gwnet_${ds}_s0_mean_emb;  L 0 train_other.py  --model gwnet  --dataset $ds --seed 0 --mixing mean --sensor_emb 1
  LOG=gwnet_${ds}_s0_random;    L 4 train_other.py  --model gwnet  --dataset $ds --seed 0 --mixing random
done
wait; echo ALL DONE
