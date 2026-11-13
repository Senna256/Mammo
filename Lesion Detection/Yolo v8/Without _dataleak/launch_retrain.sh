#!/bin/bash
# Retrain with equalised Swin / ViT settings: Swin-Tiny vs ViT-Small, both ImageNet-21k + 1k fine-tune,
# drop_path 0.1, timm input normalisation, same Detect head, TRAIN data augmentation (mammo_prep 0.5.0),
# plus the ViT-Base ablation on the positives regime. The previous runs are in backup_before_equalization/.
# One job per GPU; each session waits for the previous one in its queue (training_status.py uses these names).
ROOT="/home/enric/Mammo/Lesion Detection/Yolo v8/Without _dataleak"
PY=/home/enric/Mammo/mammo_env/bin/python
launch() {  # session gpu script [session_to_wait_for]
  local session=$1 gpu=$2 script=$3 wait_for=$4
  local dir="$ROOT/$(dirname "$script")" name=$(basename "$script" .py)
  local cmd="cd \"$dir\""
  [ -n "$wait_for" ] && cmd="$cmd; while tmux has-session -t $wait_for 2>/dev/null; do sleep 60; done"
  cmd="$cmd; CUDA_VISIBLE_DEVICES=$gpu $PY $name.py --train > $name.log 2>&1"
  tmux new-session -d -s "$session" "$cmd"
}
# GPU 0 (A40): ~33 h
launch swin_full_gpu0           0 "swin/Train/full_dataset/train_full_swin.py"
launch swin_4k_gpu0             0 "swin/Train/4k images/train_4k_swin.py"                         swin_full_gpu0
launch vit_4k_gpu0              0 "vit/Train/4k images/train_4k_vit.py"                           swin_4k_gpu0
launch vit_base_positives_gpu0  0 "vit_base_ablation/Train/positives/train_vit_base_positives.py" vit_4k_gpu0
# GPU 1 (A30): ~30 h
launch vit_full_gpu1            1 "vit/Train/full_dataset/train_full_vit.py"
launch swin_positives_gpu1      1 "swin/Train/positives/train_swin_positives.py"                  vit_full_gpu1
launch vit_positives_gpu1       1 "vit/Train/positives/train_vit_positives.py"                    swin_positives_gpu1
tmux ls
