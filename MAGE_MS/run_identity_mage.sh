#!/bin/bash
# 2026-11-08: sensor identity in slim MAGE (96 -> 12, seed 0, TF32): computed profile vs none, and the unseen-sensor
# test (20 % of sensors held out of training) with learned embedding vs computed profile. GPUs 0 and 3, 4 jobs each.
cd "$(dirname "$0")"; mkdir -p logs/idty results/idty
L(){ n=$2_T96_lite_$3_s0; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 .venv/bin/python train.py --dataset $2 --seq_len 96 \
     --arm lite --seed 0 --tf32 1 --data_dir data_w288 ${@:4} --out results/idty/$n.json > logs/idty/$n.log 2>&1 & }
for ds in PEMS04 PEMS08; do
  g=$([ $ds = PEMS04 ] && echo 0 || echo 3)
  L $g $ds profile --identity profile
  L $g $ds noid --identity none
  L $g $ds hold_learned --holdout 0.2
  L $g $ds hold_profile --holdout 0.2 --identity profile
done
wait; echo ALL DONE
