#!/usr/bin/env python3
"""Compare the six Without _dataleak experiments on the official CSV TEST split.

Experiments: {Swin, ViT} x {full dataset, 4k images, positives}.

Each experiment is evaluated by its own script (e.g. vit/Test/full_dataset/
eval_full_vit.py), which writes Test/outputs/<experiment>/. This script only
reads those outputs; it runs no model:
    Test/outputs/<experiment>/metrics_by_iou.csv, metrics.json
and writes Test/outputs/comparison/:
    metrics_by_iou_all.csv, summary_comparison.csv
    tpr_vs_iou_1fppi.png, tpr_vs_iou_operating_points.png, ap_vs_iou.png
    comparison_report.txt

It also holds what the analysis scripts in outputs/comparison/ share
(EXPERIMENTS, image_key, the matching and metric functions, plot colours).
The metric functions are the same as in the six evaluation scripts.

Usage:
    python compare_all.py
    python compare_all.py --experiments swin_full vit_full
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

EXPERIMENTS = {
    "swin_full": {"label": "Swin · full", "backbone": "Swin", "regime": "full",
                  "script": "swin/Train/full_dataset/train_full_swin.py",
                  "eval_script": "swin/Test/full_dataset/eval_full_swin.py"},
    "swin_4k": {"label": "Swin · 4k", "backbone": "Swin", "regime": "4k",
                "script": "swin/Train/4k images/train_4k_swin.py",
                "eval_script": "swin/Test/4k images/eval_4k_swin.py"},
    "swin_positives": {"label": "Swin · positives", "backbone": "Swin", "regime": "positives",
                       "script": "swin/Train/positives/train_swin_positives.py",
                       "eval_script": "swin/Test/positives/eval_swin_positives.py"},
    "vit_full": {"label": "ViT · full", "backbone": "ViT", "regime": "full",
                 "script": "vit/Train/full_dataset/train_full_vit.py",
                 "eval_script": "vit/Test/full_dataset/eval_full_vit.py"},
    "vit_4k": {"label": "ViT · 4k", "backbone": "ViT", "regime": "4k",
               "script": "vit/Train/4k images/train_4k_vit.py",
               "eval_script": "vit/Test/4k images/eval_4k_vit.py"},
    "vit_positives": {"label": "ViT · positives", "backbone": "ViT", "regime": "positives",
                      "script": "vit/Train/positives/train_vit_positives.py",
                      "eval_script": "vit/Test/positives/eval_vit_positives.py"},
}

# Same values as the evaluation scripts (used for the plots and the report).
CONF_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300
IOU_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55,
                  0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FPPI_POINTS = [0.25, 0.5, 1.0, 2.0, 4.0]
MAIN_FPPI = 1.0

# Plot styling: color = training regime, line style = backbone.
REGIME_COLORS = {"full": "#2a78d6", "4k": "#eb6834", "positives": "#1baf7a"}
BACKBONE_STYLES = {"Swin": {"linestyle": "-", "marker": "o"},
                   "ViT": {"linestyle": "--", "marker": "s"}}
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# ============================================================
# SHARED HELPERS (used by the analysis scripts in outputs/comparison/)
# ============================================================

def load_training_module(key):
    """Import an experiment's training script (its preprocessing, splits and model)."""
    path = EXPERIMENTS_ROOT / EXPERIMENTS[key]["script"]
    spec = importlib.util.spec_from_file_location(f"train_{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_key(dicom_path):
    """Image identifier used in the output CSVs: "study_id/image_id"."""
    path = Path(dicom_path)
    return f"{path.parent.name}/{path.stem}"


def iou_matrix(boxes_a, boxes_b):
    """IoU of every box in boxes_a (N, 4) with every box in boxes_b (M, 4); boxes are (x1, y1, x2, y2)."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def match_detections(predictions, ground_truth, iou_threshold):
    """Greedy matching by descending confidence; each GT can be matched once.

    Returns the TP flag of every detection, sorted by descending confidence
    over the whole TEST set.
    """
    confidence = predictions["confidence"].to_numpy()
    is_tp = np.zeros(len(predictions), dtype=bool)
    gt_by_image = {img: g[["x1", "y1", "x2", "y2"]].to_numpy() for img, g in ground_truth.groupby("image_id")}
    for image_id, group in predictions.groupby("image_id"):
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None:
            continue  # Images without GT: every detection is a FP.
        order = group.index.to_numpy()[np.argsort(-group["confidence"].to_numpy(), kind="stable")]
        ious = iou_matrix(predictions.loc[order, ["x1", "y1", "x2", "y2"]].to_numpy(), gt_boxes)
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for row, det_index in enumerate(order):
            candidates = np.where(matched, -1.0, ious[row])
            best = int(np.argmax(candidates))
            if candidates[best] >= iou_threshold:
                matched[best] = True
                is_tp[det_index] = True
    global_order = np.argsort(-confidence, kind="stable")
    return is_tp[global_order]


def average_precision(cum_tp, cum_fp, total_gt):
    """All-point interpolated AP."""
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall_points = np.concatenate([[0.0], recall, [1.0]])
    precision_points = np.concatenate([[1.0], precision, [0.0]])
    precision_points = np.maximum.accumulate(precision_points[::-1])[::-1]
    return float(np.sum(np.diff(recall_points) * precision_points[1:]))


def tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, fppi):
    """TPR when detections are kept down to the confidence giving `fppi` FPs per image."""
    kept = np.searchsorted(cum_fp, fppi * n_images, side="right")
    return float(cum_tp[kept - 1] / total_gt) if kept > 0 else 0.0

# ============================================================
# LOADING
# ============================================================

def load_experiment(key, output_dir):
    """metrics_by_iou.csv, metrics.json and meta.json written by the experiment's evaluation script."""
    exp_dir = output_dir / key
    missing = [name for name in ("metrics_by_iou.csv", "metrics.json", "meta.json") if not (exp_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{exp_dir}: missing {', '.join(missing)}. "
                                f"Run {EXPERIMENTS[key]['eval_script']} first.")
    per_iou = pd.read_csv(exp_dir / "metrics_by_iou.csv")
    summary = json.loads((exp_dir / "metrics.json").read_text())
    meta = json.loads((exp_dir / "meta.json").read_text())
    summary["checkpoint"] = Path(meta["checkpoint"]).name
    return per_iou, summary

# ============================================================
# PLOTS
# ============================================================

def style_axis(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("IoU threshold", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min(IOU_THRESHOLDS) - 0.02, max(IOU_THRESHOLDS) + 0.02)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)


def plot_lines(ax, per_iou, column):
    for key, exp in EXPERIMENTS.items():
        data = per_iou[per_iou["experiment"] == key]
        if data.empty:
            continue
        ax.plot(data["iou"], data[column], label=exp["label"], color=REGIME_COLORS[exp["regime"]],
                linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                **BACKBONE_STYLES[exp["backbone"]])


def save_figure(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[SAVE] {path}", flush=True)


def make_plots(per_iou, comparison_dir):
    legend_kwargs = {"frameon": False, "fontsize": 8, "labelcolor": INK_SECONDARY}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_lines(ax, per_iou, f"TPR@{MAIN_FPPI:g}FPPI")
    style_axis(ax, f"TPR vs IoU at {MAIN_FPPI:g} false positive per image (TEST)", "TPR (sensitivity)")
    ax.legend(loc="upper right", **legend_kwargs)
    save_figure(fig, comparison_dir / f"tpr_vs_iou_{MAIN_FPPI:g}fppi.png")

    panels = [(f"TPR@{f:g}FPPI", f"{f:g} FP / image") for f in FPPI_POINTS]
    panels.append(("TPR_all_detections", f"All detections (≤{MAX_DETECTIONS} / image)"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, (column, title) in zip(axes.flat, panels):
        plot_lines(ax, per_iou, column)
        style_axis(ax, title, "TPR (sensitivity)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), **legend_kwargs)
    fig.suptitle("TPR vs IoU at different operating points (TEST)", color=INK_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, comparison_dir / "tpr_vs_iou_operating_points.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_lines(ax, per_iou, "AP")
    style_axis(ax, "AP vs IoU (TEST)", "Average precision")
    ax.set_ylim(0, max(0.05, float(per_iou["AP"].max()) * 1.15))
    ax.legend(loc="upper right", **legend_kwargs)
    save_figure(fig, comparison_dir / "ap_vs_iou.png")

# ============================================================
# REPORT
# ============================================================

def write_report(summary, comparison_dir):
    lines = [
        "Without _dataleak — TEST evaluation of the six experiments",
        "=" * 70,
        f"Checkpoint: {', '.join(sorted(summary['checkpoint'].unique()))}",
        f"NMS: conf {CONF_THRESHOLD}, IoU {NMS_IOU_THRESHOLD}, max {MAX_DETECTIONS} detections/image",
        "Why NMS: the YOLOv8 head scores all 21,504 grid positions, so each lesion yields many",
        "overlapping boxes; without NMS the duplicates are counted as false positives,",
        "making FPPI, precision and AP meaningless. Same settings for all experiments.",
        "TPR denominator: CSV TEST boxes (identical for all experiments); boxes removed",
        "by an experiment's breast crop count as missed lesions.",
        "FPPI = false positives per TEST image (all images, positive and negative).",
        "",
        summary.drop(columns="checkpoint").to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "Caveats",
        "-" * 70,
        "- The six training scripts are identical (same detect_breast_roi crop, splits,",
        "  validate(), batch size 8 x 4 accumulation) except for the backbone and",
        "  select_train_records(). VAL is never filtered.",
        "- The six evaluation scripts are identical except for the backbone and the",
        "  experiment paths; each uses the preprocessing of its training script.",
    ]
    path = comparison_dir / "comparison_report.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"[SAVE] {path}", flush=True)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare the Without _dataleak experiments on the CSV TEST split.")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Folder with one subfolder per experiment (as written by the evaluation scripts).")
    args = parser.parse_args()

    loaded = [load_experiment(key, args.output_dir) for key in args.experiments]
    per_iou = pd.concat([per_iou for per_iou, _ in loaded], ignore_index=True)
    summary = pd.DataFrame([summary for _, summary in loaded])

    comparison_dir = args.output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    per_iou.to_csv(comparison_dir / "metrics_by_iou_all.csv", index=False)
    summary.drop(columns="checkpoint").to_csv(comparison_dir / "summary_comparison.csv", index=False)
    make_plots(per_iou, comparison_dir)
    write_report(summary, comparison_dir)

    print()
    print(summary[["label", "checkpoint_epoch", "gt_lost_by_preprocessing", "AP@0.50", "mAP@0.50:0.95",
                   "TPR@IoU0.30_1FPPI", "TPR@IoU0.50_1FPPI", "TPR@IoU0.50_all_detections"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
