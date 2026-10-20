#!/usr/bin/env python3
"""ViT-Base + YOLOv8 — TRAIN regime: full.

All data, preprocessing, model, loss and training code lives in
Without _dataleak/common.py; this script only selects the backbone and the
TRAIN regime. Usage: python train_full_vit.py --test | --train | --resume CHECKPOINT
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # Without _dataleak/

from common import *  # noqa: E402,F401,F403  (Test/evaluate_all.py reads config and classes from this module)
from common import ViTYOLO, main  # noqa: E402

MODEL_CLASS = ViTYOLO
REGIME = "full"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

if __name__ == "__main__":
    main(MODEL_CLASS, REGIME, OUTPUT_DIR)
