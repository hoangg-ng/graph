#!/bin/bash
# 2026-11-07: collapse check part 2 (MTGNN, STAEformer, ASTGCN). PEMS08 now; PEMS04 after run_others.sh finishes.
# STAEformer capped at 80 epochs: its recipe cuts the LR by 1000x after epoch 65 (milestones 25/45/65, x0.1).
cd "$(dirname "$0")"
launch() { CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} ../.venv/bin/python train_other2.py --model $1 --dataset $2 --seed 0 $3 > logs/${1}_${2}_s0.log 2>&1 & }
launch mtgnn PEMS08; launch astgcn PEMS08; launch staeformer PEMS08 "--epochs 80"
until grep -q "ALL DONE" logs/run_others.out 2>/dev/null; do sleep 60; done
launch mtgnn PEMS04; launch astgcn PEMS04; launch staeformer PEMS04 "--epochs 80"
wait; echo ALL DONE
