#!/bin/bash

python3 -u train.py \
  --data_root ./traindata/LA \
  --batch_size 8 \
  --accum_steps 4 \
  --grad_checkpoint \
  --val_every 2 > train.log 2>&1 &

tail -f train.log
