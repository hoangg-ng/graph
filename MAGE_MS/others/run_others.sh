#!/bin/bash
# 2026-11-07: collapse check on Graph WaveNet and AGCRN (reimpl.py), PEMS04/08, seed 0, official recipes
cd "$(dirname "$0")"
for m in gwnet agcrn; do for d in PEMS04 PEMS08; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} ../.venv/bin/python train_other.py --model $m --dataset $d --seed 0 > logs/${m}_${d}_s0.log 2>&1 &
done; done
wait; echo ALL DONE
