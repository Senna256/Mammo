#!/usr/bin/env python3
"""FROC curves of the six experiments on the TEST split.

A FROC curve plots sensitivity (TPR = matched GT boxes / all GT boxes) against
the mean number of false positives per image (FPPI) while the confidence
threshold goes from high to low. Every point of the curve is one threshold.

How it is computed
------------------
- Uses Test/outputs/<experiment>/predictions.csv and ground_truth.csv written
  by the evaluation scripts (so the curve has at most MAX_DETECTIONS boxes per
  image, as in the metrics).
- Detections are matched with the same greedy function as the evaluation
  scripts (match_detections, from compare_all.py) at each IoU threshold.
- TPR denominator = all CSV GT boxes (csv_gt_boxes in meta.json), FPPI
  denominator = all TEST images, as in tpr_at_fppi(). At FPPI 0.5/1/2 the
  curve gives exactly the TPR@FPPI values of summary_comparison.csv (checked
  at start-up).
- A curve ends where the model has no more detections: models that predict
  little ("full") never reach high FPPI, so their curve stops early. That end
  point is the TPR "all detections".

Outputs (in comparison/froc/):
    froc_iou0.30.png, froc_iou0.50.png   the six curves on one plot
    froc_per_model.png                   one panel per model, IoU 0.30 and 0.50
    froc_points.csv                      the curves (threshold, FPPI, TPR)
    froc_summary.csv                     TPR at fixed FPPI + mean sensitivity

Run the six evaluation scripts (*/Test/*/eval_*.py) and compare_all.py first.

Usage:
    python froc.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARISON_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = COMPARISON_DIR.parent
sys.path.insert(0, str(OUTPUTS_DIR.parent))  # Test/, for compare_all.py

from compare_all import (AXIS_COLOR, BACKBONE_STYLES, EXPERIMENTS, GRID_COLOR,  # noqa: E402
                         INK_MUTED, INK_PRIMARY, INK_SECONDARY, MAX_DETECTIONS,
                         REGIME_COLORS, SURFACE, match_detections, tpr_at_fppi)

# ============================================================
# CONFIGURATION
# ============================================================

FROC_DIR = COMPARISON_DIR / "froc"
IOUS = [0.30, 0.50]
# Usual FROC operating points; their mean TPR is the "mean sensitivity" score.
FPPI_POINTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
X_LIMITS = (0.01, 10.0)
X_TICKS = [0.01, 0.03, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# ============================================================
# DATA
# ============================================================


def froc_curve(predictions, ground_truth, iou, total_gt, n_images):
    """Cumulative FPPI and TPR after each detection (sorted by descending confidence)."""
    is_tp = match_detections(predictions, ground_truth, iou)
    confidence = np.sort(predictions["confidence"].to_numpy())[::-1]
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(~is_tp)
    return pd.DataFrame({"confidence": confidence,
                         "fppi": cum_fp / n_images,
                         "tpr": cum_tp / total_gt,
                         "cum_tp": cum_tp, "cum_fp": cum_fp})


def load_curves():
    curves, rows = [], []
    for key, exp in EXPERIMENTS.items():
        exp_dir = OUTPUTS_DIR / key
        meta = json.loads((exp_dir / "meta.json").read_text())
        summary = json.loads((exp_dir / "metrics.json").read_text())
        predictions = pd.read_csv(exp_dir / "predictions.csv")
        ground_truth = pd.read_csv(exp_dir / "ground_truth.csv")
        total_gt, n_images = meta["csv_gt_boxes"], meta["test_images"]

        for iou in IOUS:
            curve = froc_curve(predictions, ground_truth, iou, total_gt, n_images)
            cum_tp, cum_fp = curve["cum_tp"].to_numpy(), curve["cum_fp"].to_numpy()

            # Check: same numbers as the evaluation script.
            for fppi in (0.5, 1.0, 2.0):
                name = f"TPR@IoU{iou:.2f}_{fppi:g}FPPI"
                ours = tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, fppi)
                if name in summary and abs(ours - summary[name]) > 1e-9:
                    raise RuntimeError(f"{key} {name}: {ours} != {summary[name]} (metrics.json)")

            tprs = {f"TPR@{f:g}FPPI": tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, f) for f in FPPI_POINTS}
            rows.append({"experiment": key, "label": exp["label"], "iou": iou, **tprs,
                         "mean_sensitivity": float(np.mean(list(tprs.values()))),
                         "max_fppi": float(curve["fppi"].iloc[-1]) if len(curve) else 0.0,
                         "max_tpr": float(curve["tpr"].iloc[-1]) if len(curve) else 0.0})
            curves.append(curve.assign(experiment=key, iou=iou))
    return pd.concat(curves, ignore_index=True), pd.DataFrame(rows)

# ============================================================
# PLOTS
# ============================================================


def style_axis(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xscale("log")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([f"{f:g}" for f in X_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("False positives per image (log scale)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Sensitivity (TPR)", color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)


def draw_curve(ax, curve, color, linestyle, label):
    """Step curve; a dot marks where the model runs out of detections."""
    if curve.empty:
        return
    x = np.concatenate([[X_LIMITS[0]], curve["fppi"].to_numpy()])
    y = np.concatenate([[0.0], curve["tpr"].to_numpy()])
    x = np.maximum(x, X_LIMITS[0])
    ax.step(x, y, where="post", color=color, linestyle=linestyle, linewidth=2, label=label)
    ax.plot(x[-1], y[-1], "o", color=color, markersize=5, markeredgecolor=SURFACE)


def save_figure(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[SAVE] {path}", flush=True)


def make_plots(curves):
    legend_kwargs = {"frameon": False, "fontsize": 8, "labelcolor": INK_SECONDARY}

    # All six models on one plot, one figure per IoU.
    for iou in IOUS:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for key, exp in EXPERIMENTS.items():
            curve = curves[(curves["experiment"] == key) & (curves["iou"] == iou)]
            draw_curve(ax, curve, REGIME_COLORS[exp["regime"]],
                       BACKBONE_STYLES[exp["backbone"]]["linestyle"], exp["label"])
        style_axis(ax, f"FROC · IoU {iou:.2f} · TEST (≤{MAX_DETECTIONS} det/image)")
        ax.legend(loc="upper left", **legend_kwargs)
        save_figure(fig, FROC_DIR / f"froc_iou{iou:.2f}.png")

    # One panel per model with both IoU thresholds.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharey=True)
    iou_styles = {0.30: "-", 0.50: ":"}
    for ax, (key, exp) in zip(axes.flat, EXPERIMENTS.items()):
        for iou in IOUS:
            curve = curves[(curves["experiment"] == key) & (curves["iou"] == iou)]
            draw_curve(ax, curve, REGIME_COLORS[exp["regime"]], iou_styles[iou], f"IoU {iou:.2f}")
        style_axis(ax, exp["label"])
        ax.legend(loc="upper left", **legend_kwargs)
    fig.suptitle(f"FROC per model · TEST (≤{MAX_DETECTIONS} det/image)",
                 color=INK_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, FROC_DIR / "froc_per_model.png")

# ============================================================
# MAIN
# ============================================================


def main():
    FROC_DIR.mkdir(parents=True, exist_ok=True)
    curves, summary = load_curves()
    print("[CHECK] TPR@0.5/1/2 FPPI equal to metrics.json for every experiment", flush=True)

    curves.drop(columns=["cum_tp", "cum_fp"]).to_csv(FROC_DIR / "froc_points.csv", index=False)
    summary.to_csv(FROC_DIR / "froc_summary.csv", index=False)
    print(f"[SAVE] {FROC_DIR / 'froc_points.csv'}")
    print(f"[SAVE] {FROC_DIR / 'froc_summary.csv'}")
    make_plots(curves)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns=["experiment"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
