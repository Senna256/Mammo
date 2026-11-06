#!/bin/bash
# Retrain the six models with mammo_prep 0.4.0 (adaptive breast-crop threshold).
# Same GPU queues and tmux session names as the 2026-10-20 runs (training_status.py).
ROOT="/home/enric/Mammo/Lesion Detection/Yolo v8/Without _dataleak"
PY=/home/enric/Mammo/mammo_env/bin/python
launch() {  # session gpu script [session_to_wait_for]
  local session=$1 gpu=$2 script=$3 wait_for=$4
  local dir="$ROOT/$(dirname "$script")" name=$(basename "$script" .py)
  local cmd="cd \"$dir\""
  [ -n "$wait_for" ] && cmd="$cmd; while tmux has-session -t $wait_for 2>/dev/null; do sleep 60; done"
  cmd="$cmd; CUDA_VISIBLE_DEVICES=$gpu $PY $name.py > $name.log 2>&1"
  tmux new-session -d -s "$session" "$cmd"
}
launch vit_full_gpu0        0 "vit/Train/full_dataset/train_full_vit.py"
launch swin_full_gpu0       0 "swin/Train/full_dataset/train_full_swin.py"     vit_full_gpu0
launch vit_4k_gpu1          1 "vit/Train/4k images/train_4k_vit.py"
launch swin_4k_gpu1         1 "swin/Train/4k images/train_4k_swin.py"          vit_4k_gpu1
launch swin_positives_gpu1  1 "swin/Train/positives/train_swin_positives.py"   swin_4k_gpu1
launch vit_positives_gpu1   1 "vit/Train/positives/train_vit_positives.py"     swin_positives_gpu1
tmux ls
