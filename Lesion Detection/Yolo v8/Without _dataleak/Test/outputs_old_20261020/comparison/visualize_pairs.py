#!/usr/bin/env python3
"""Draw Swin vs ViT bounding boxes on the same TEST image, one figure per regime.

For every selected image three figures are written:
    pairs/full/<image>.png       Swin · full       vs ViT · full
    pairs/4k/<image>.png         Swin · 4k         vs ViT · 4k
    pairs/positives/<image>.png  Swin · positives  vs ViT · positives

Each panel shows the image exactly as that model saw it (its own breast crop
and 1024x1024 resize/pad), with the cached predictions from evaluate_all.py
(outputs/<experiment>/predictions.csv, already after NMS) and the ground truth
mapped into the same space (outputs/<experiment>/ground_truth.csv).

Which boxes are drawn: by default each model is shown at its own 1 FPPI
operating point (the confidence threshold that gives 1 false positive per TEST
image at IoU 0.30), the same point used in tpr_vs_iou_1fppi.png. Use --conf for
a fixed threshold instead.

Box colors: ground truth = yellow dashed, TP (IoU >= --iou with a GT) = green,
FP = red. The number on each box is the confidence.

Run evaluate_all.py first so the cached predictions exist.

Examples:
    python visualize_pairs.py --n 10
    python visualize_pairs.py --images <study_id>/<image_id> --conf 0.1
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

COMPARISON_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = COMPARISON_DIR.parent
sys.path.insert(0, str(OUTPUTS_DIR.parent))  # Test/, for evaluate_all.py

from evaluate_all import (EXPERIMENTS, image_key, iou_matrix,  # noqa: E402
                          load_training_module, match_detections)

# ============================================================
# CONFIGURATION
# ============================================================

REGIMES = ["full", "4k", "positives"]
OPERATING_IOU = 0.30
OPERATING_FPPI = 1.0

GT_COLOR, TP_COLOR, FP_COLOR = "#f5c518", "#1baf7a", "#e5484d"
INK_PRIMARY, INK_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# ============================================================
# DATA
# ============================================================

def experiment_key(backbone, regime):
    return f"{backbone.lower()}_{regime}"


def load_cached(key):
    exp_dir = OUTPUTS_DIR / key
    meta = json.loads((exp_dir / "meta.json").read_text())
    predictions = pd.read_csv(exp_dir / "predictions.csv")
    ground_truth = pd.read_csv(exp_dir / "ground_truth.csv")
    return meta, predictions, ground_truth


def operating_threshold(predictions, ground_truth, n_images):
    """Confidence threshold giving OPERATING_FPPI false positives per image at OPERATING_IOU."""
    tp = match_detections(predictions, ground_truth, OPERATING_IOU)
    cum_fp = np.cumsum(~tp)
    confidence = np.sort(predictions["confidence"].to_numpy())[::-1]
    kept = np.searchsorted(cum_fp, OPERATING_FPPI * n_images, side="right")
    if kept == 0:
        return float(confidence[0]) + 1e-9
    return float(confidence[kept - 1])


class ModelImages:
    """Loads TEST images with the exact preprocessing of one experiment."""

    def __init__(self, key):
        module = load_training_module(key)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, test_records = module.build_dataset_records(module.ANNOTATIONS_CSV, module.NETWORK_IMAGES)
        self.dataset = module.VindrDataset(test_records, img_size=module.IMG_SIZE)
        self.index = {image_key(r["dicom_path"]): i for i, r in enumerate(test_records)}

    def get(self, image_id):
        image, _, _ = self.dataset[self.index[image_id]]
        return np.transpose(image.numpy(), (1, 2, 0))

# ============================================================
# PLOTTING
# ============================================================

def draw_panel(ax, image, predictions, gt_boxes, threshold, iou_threshold, title):
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for x1, y1, x2, y2 in gt_boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=GT_COLOR,
                               linewidth=2, linestyle="--"))

    kept = predictions[predictions["confidence"] >= threshold].sort_values("confidence", ascending=False)
    boxes = kept[["x1", "y1", "x2", "y2"]].to_numpy()
    matched = np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else np.zeros((len(boxes), 0))
    n_tp = 0
    for row, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes, kept["confidence"])):
        # Same greedy matching as evaluate_all: each GT is matched at most once.
        candidates = np.where(matched, -1.0, ious[row]) if len(gt_boxes) else np.array([-1.0])
        best = int(np.argmax(candidates))
        is_tp = candidates[best] >= iou_threshold
        if is_tp:
            matched[best] = True
            n_tp += 1
        color = TP_COLOR if is_tp else FP_COLOR
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.6))
        ax.text(x1, y1 - 4, f"{conf:.2f}", color="white", fontsize=7,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2})

    subtitle = (f"conf ≥ {threshold:.3f} · {len(kept)} boxes · TP {n_tp} · FP {len(kept) - n_tp} · "
                f"GT found {int(matched.sum())}/{len(gt_boxes)}")
    ax.set_title(f"{title}\n{subtitle}", color=INK_PRIMARY, fontsize=10, loc="left")


def save_pair(image_id, regime, panels, iou_threshold, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, panel in zip(axes, panels):
        draw_panel(ax, panel["image"], panel["predictions"], panel["gt"], panel["threshold"],
                   iou_threshold, panel["title"])
    legend = [Line2D([], [], color=GT_COLOR, linestyle="--", linewidth=2, label="Ground truth"),
              Line2D([], [], color=TP_COLOR, linewidth=2, label=f"TP (IoU ≥ {iou_threshold:.2f})"),
              Line2D([], [], color=FP_COLOR, linewidth=2, label="FP")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{regime} · {image_id}", color=INK_SECONDARY, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.patch.set_facecolor(SURFACE)

    path = output_dir / regime / f"{image_id.replace('/', '__')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[SAVE] {path}", flush=True)

# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Swin vs ViT bounding boxes on TEST images, one figure per regime.")
    parser.add_argument("--images", nargs="+", default=None, help="Image ids '<study>/<image>' (see predictions.csv).")
    parser.add_argument("--n", type=int, default=5, help="Random TEST images to draw when --images is not given.")
    parser.add_argument("--negatives", action="store_true", help="Sample images without lesions instead of positives.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--conf", type=float, default=None,
                        help=f"Fixed confidence threshold. Default: each model's {OPERATING_FPPI:g} FPPI point.")
    parser.add_argument("--iou", type=float, default=OPERATING_IOU, help="IoU to colour a box as TP.")
    parser.add_argument("--output-dir", type=Path, default=COMPARISON_DIR / "pairs")
    return parser.parse_args()


def select_images(args, ground_truth_by_key):
    if args.images:
        return args.images
    # Images present in every experiment's TEST split; positives = GT in all of them.
    keys = [experiment_key(b, r) for r in args.regimes for b in ("Swin", "ViT")]
    all_ids = set.intersection(*(set(pd.read_csv(OUTPUTS_DIR / k / "predictions.csv",
                                                 usecols=["image_id"])["image_id"]) for k in keys))
    positive_ids = set.intersection(*(set(ground_truth_by_key[k]["image_id"]) for k in keys))
    pool = sorted(all_ids - positive_ids if args.negatives else positive_ids)
    rng = np.random.default_rng(args.seed)
    return list(rng.choice(pool, size=min(args.n, len(pool)), replace=False))


def main():
    args = parse_args()

    cached = {}
    for regime in args.regimes:
        for backbone in ("Swin", "ViT"):
            key = experiment_key(backbone, regime)
            meta, predictions, ground_truth = load_cached(key)
            threshold = args.conf if args.conf is not None else operating_threshold(
                predictions, ground_truth, meta["test_images"])
            print(f"[{key}] confidence threshold: {threshold:.4f}", flush=True)
            cached[key] = {"predictions": predictions, "ground_truth": ground_truth, "threshold": threshold}

    image_ids = select_images(args, {k: v["ground_truth"] for k, v in cached.items()})
    print(f"Images: {len(image_ids)}", flush=True)

    for regime in args.regimes:
        loaders = {b: ModelImages(experiment_key(b, regime)) for b in ("Swin", "ViT")}
        for image_id in image_ids:
            panels = []
            for backbone in ("Swin", "ViT"):
                key = experiment_key(backbone, regime)
                data = cached[key]
                predictions = data["predictions"][data["predictions"]["image_id"] == image_id]
                gt = data["ground_truth"][data["ground_truth"]["image_id"] == image_id]
                panels.append({
                    "image": loaders[backbone].get(image_id),
                    "predictions": predictions,
                    "gt": gt[["x1", "y1", "x2", "y2"]].to_numpy(),
                    "threshold": data["threshold"],
                    "title": EXPERIMENTS[key]["label"],
                })
            save_pair(image_id, regime, panels, args.iou, args.output_dir)


if __name__ == "__main__":
    main()
