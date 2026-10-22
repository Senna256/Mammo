#!/usr/bin/env python3
"""Paired study-level bootstrap of the TEST metrics of the six experiments.

What the intervals mean
-----------------------
They measure TEST-set sampling uncertainty only: how much each metric would move
with a different TEST set of the same size from the same population. They do
NOT include training variability (seeds); that needs retraining.

How it works
------------
- Resampling unit = study (1,000 studies x 4 views). Views of the same breast /
  patient are correlated, so resampling single images would give intervals that
  are too narrow.
- Paired: every bootstrap replicate uses the SAME resampled studies for all six
  experiments, so differences between models are computed on identical data.
- Detection matching (greedy, IoU threshold) is done per image and does not
  depend on which studies are drawn, so it is computed once. A replicate only
  re-weights each detection, GT box and image by the number of times its study
  was drawn, then recomputes TPR@FPPI and AP with the same functions as
  evaluate_all.py. With all weights = 1 the point estimates are exactly the ones
  in summary_comparison.csv (checked at start-up).

Outputs (in comparison/bootstrap/):
    bootstrap_ci.csv                 estimate + 95% percentile CI per model/metric
    bootstrap_paired_diff.csv        model - baseline: delta, 95% CI, two-sided p
    bootstrap_distributions.png      boxplots of each model's bootstrap distribution
    bootstrap_paired_differences.png boxplots of model - baseline differences
    bootstrap_samples.npz            raw replicates

Run evaluate_all.py first so the cached predictions exist.
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
from tqdm import tqdm

COMPARISON_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = COMPARISON_DIR.parent
sys.path.insert(0, str(OUTPUTS_DIR.parent))  # Test/, for evaluate_all.py

from evaluate_all import (AXIS_COLOR, EXPERIMENTS, GRID_COLOR, INK_MUTED,  # noqa: E402
                          INK_PRIMARY, INK_SECONDARY, REGIME_COLORS, SURFACE,
                          average_precision, image_key, load_training_module,
                          match_detections, tpr_at_fppi)

# ============================================================
# CONFIGURATION
# ============================================================

# (column name, IoU, kind, FPPI) — TPR@1FPPI IoU 0.30 is the main metric.
METRICS = [
    ("TPR@IoU0.30_1FPPI", 0.30, "tpr", 1.0),
    ("TPR@IoU0.30_0.5FPPI", 0.30, "tpr", 0.5),
    ("TPR@IoU0.30_2FPPI", 0.30, "tpr", 2.0),
    ("TPR@IoU0.50_1FPPI", 0.50, "tpr", 1.0),
    ("AP@0.30", 0.30, "ap", None),
    ("AP@0.50", 0.50, "ap", None),
]
METRIC_TITLES = {
    "TPR@IoU0.30_1FPPI": "TPR · 1 FP/image · IoU 0.30  (main)",
    "TPR@IoU0.30_0.5FPPI": "TPR · 0.5 FP/image · IoU 0.30",
    "TPR@IoU0.30_2FPPI": "TPR · 2 FP/image · IoU 0.30",
    "TPR@IoU0.50_1FPPI": "TPR · 1 FP/image · IoU 0.50",
    "AP@0.30": "AP · IoU 0.30",
    "AP@0.50": "AP · IoU 0.50",
}
CI_LEVEL = 0.95

# ============================================================
# DATA
# ============================================================

def load_test_studies():
    """Per-image study code, CSV GT count and the study list (TEST is identical for all experiments)."""
    module = load_training_module("swin_full")
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test_records = module.build_dataset_records(module.ANNOTATIONS_CSV, module.NETWORK_IMAGES)
    images = pd.DataFrame({
        "image_id": [image_key(r["dicom_path"]) for r in test_records],
        "study": [image_key(r["dicom_path"]).split("/")[0] for r in test_records],
        "gt": [len(r["boxes"]) for r in test_records],
    })
    studies = np.array(sorted(images["study"].unique()))
    images["study_code"] = np.searchsorted(studies, images["study"])
    gt_per_study = np.bincount(images["study_code"], weights=images["gt"], minlength=len(studies))
    images_per_study = np.bincount(images["study_code"], minlength=len(studies)).astype(float)
    return images, studies, gt_per_study, images_per_study


def detection_tables(key, images, iou_thresholds):
    """For each IoU: TP flag and study code of every detection, sorted by descending confidence."""
    exp_dir = OUTPUTS_DIR / key
    meta = json.loads((exp_dir / "meta.json").read_text())
    if meta["test_images"] != len(images) or meta["csv_gt_boxes"] != images["gt"].sum():
        raise RuntimeError(f"{key}: cached predictions do not match the current TEST split")
    predictions = pd.read_csv(exp_dir / "predictions.csv")
    ground_truth = pd.read_csv(exp_dir / "ground_truth.csv")

    study_of_image = images.set_index("image_id")["study_code"]
    if not predictions["image_id"].isin(study_of_image.index).all():
        raise RuntimeError(f"{key}: predictions contain images outside TEST")
    # Same global order as match_detections().
    order = np.argsort(-predictions["confidence"].to_numpy(), kind="stable")
    study_codes = study_of_image.loc[predictions["image_id"].to_numpy()[order]].to_numpy()
    return {iou: (match_detections(predictions, ground_truth, iou), study_codes) for iou in iou_thresholds}


def metric_values(tables, study_weights, gt_per_study, images_per_study):
    """All METRICS for one experiment and one set of study weights."""
    total_gt = float(study_weights @ gt_per_study)
    n_images = float(study_weights @ images_per_study)
    curves = {}
    for iou, (tp, study_codes) in tables.items():
        weights = study_weights[study_codes]
        curves[iou] = (np.cumsum(weights * tp), np.cumsum(weights * ~tp))
    values = {}
    for name, iou, kind, fppi in METRICS:
        cum_tp, cum_fp = curves[iou]
        if len(cum_tp) == 0:
            values[name] = 0.0
        elif kind == "tpr":
            values[name] = tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, fppi)
        else:
            values[name] = average_precision(cum_tp, cum_fp, total_gt)
    return values

# ============================================================
# BOOTSTRAP
# ============================================================

def run_bootstrap(keys, n_boot, seed):
    images, studies, gt_per_study, images_per_study = load_test_studies()
    n_studies = len(studies)
    print(f"TEST: {len(images):,} images · {n_studies:,} studies · {int(gt_per_study.sum())} GT boxes", flush=True)

    iou_thresholds = sorted({iou for _, iou, _, _ in METRICS})
    tables = {key: detection_tables(key, images, iou_thresholds) for key in keys}

    # Sanity check: unit weights reproduce evaluate_all.py.
    summary = pd.read_csv(COMPARISON_DIR / "summary_comparison.csv").set_index("experiment")
    ones = np.ones(n_studies)
    estimates = {key: metric_values(tables[key], ones, gt_per_study, images_per_study) for key in keys}
    for key in keys:
        for name, *_ in METRICS:
            if not np.isclose(estimates[key][name], summary.loc[key, name], atol=1e-9):
                raise RuntimeError(f"{key} {name}: {estimates[key][name]} != summary {summary.loc[key, name]}")

    # One set of resampled studies per replicate, shared by all experiments (paired).
    rng = np.random.default_rng(seed)
    study_weights = np.stack([np.bincount(rng.integers(0, n_studies, n_studies), minlength=n_studies)
                              for _ in range(n_boot)]).astype(float)

    samples = {key: np.zeros((n_boot, len(METRICS))) for key in keys}
    for b in tqdm(range(n_boot), desc="Bootstrap", unit="rep", dynamic_ncols=True):
        for key in keys:
            values = metric_values(tables[key], study_weights[b], gt_per_study, images_per_study)
            samples[key][b] = [values[name] for name, *_ in METRICS]
    return estimates, samples


def summarize(estimates, samples, baseline):
    alpha = (1 - CI_LEVEL) / 2
    ci_rows, diff_rows = [], []
    for key, values in samples.items():
        for m, (name, *_) in enumerate(METRICS):
            ci_rows.append({
                "experiment": key, "label": EXPERIMENTS[key]["label"], "metric": name,
                "estimate": estimates[key][name],
                "ci_low": np.quantile(values[:, m], alpha), "ci_high": np.quantile(values[:, m], 1 - alpha),
                "bootstrap_sd": values[:, m].std(ddof=1),
            })
            if key == baseline:
                continue
            delta = values[:, m] - samples[baseline][:, m]
            diff_rows.append({
                "experiment": key, "label": EXPERIMENTS[key]["label"], "baseline": baseline, "metric": name,
                "delta": estimates[key][name] - estimates[baseline][name],
                "ci_low": np.quantile(delta, alpha), "ci_high": np.quantile(delta, 1 - alpha),
                "p_two_sided": min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean())),
                "ci_excludes_0": bool(np.quantile(delta, alpha) > 0 or np.quantile(delta, 1 - alpha) < 0),
            })
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)

# ============================================================
# PLOTS
# ============================================================

def style_axis(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=8)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)


def draw_boxes(ax, data, keys, estimates):
    """Horizontal boxes: box = IQR, whiskers = 95% percentile CI, dot = point estimate."""
    lo, hi = 100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2
    positions = np.arange(len(keys))[::-1]
    boxes = ax.boxplot(data, positions=positions, vert=False, whis=(lo, hi), showfliers=False,
                       widths=0.55, patch_artist=True,
                       medianprops={"color": INK_PRIMARY, "linewidth": 1.2},
                       whiskerprops={"color": INK_SECONDARY, "linewidth": 1.2},
                       capprops={"color": INK_SECONDARY, "linewidth": 1.2})
    for patch, key in zip(boxes["boxes"], keys):
        color = REGIME_COLORS[EXPERIMENTS[key]["regime"]]
        if EXPERIMENTS[key]["backbone"] == "Swin":
            patch.set(facecolor=color, edgecolor=color, alpha=0.85)
        else:  # ViT: hatched outline, matching the dashed lines of the IoU plots
            patch.set(facecolor=SURFACE, edgecolor=color, hatch="////", linewidth=1.4)
    ax.scatter(estimates, positions, s=36, color=INK_PRIMARY, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_yticks(positions, [EXPERIMENTS[k]["label"] for k in keys])
    ax.set_ylim(-0.7, len(keys) - 0.3)
    return positions


def annotate(ax, positions, texts):
    for y, text in zip(positions, texts):
        ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_SECONDARY)


def plot_distributions(keys, estimates, samples, ci, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for m, ((name, *_), ax) in enumerate(zip(METRICS, axes.flat)):
        positions = draw_boxes(ax, [samples[k][:, m] for k in keys], keys, [estimates[k][name] for k in keys])
        rows = ci[ci["metric"] == name].set_index("experiment")
        annotate(ax, positions, [f"{rows.loc[k, 'estimate']:.3f} [{rows.loc[k, 'ci_low']:.3f}, "
                                 f"{rows.loc[k, 'ci_high']:.3f}]" for k in keys])
        style_axis(ax, METRIC_TITLES[name], "value")
        if m % 3:
            ax.set_yticklabels([])
    fig.suptitle(f"Bootstrap distribution per model (TEST, {len(next(iter(samples.values())))} paired study "
                 f"resamples) — box: IQR · whiskers: {CI_LEVEL:.0%} CI · dot: estimate",
                 color=INK_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=9)
    save(fig, path)


def format_p(p, n_boot):
    # A bootstrap p-value cannot resolve below ~2/n_boot.
    return f"p<{2 / n_boot:.3f}" if p < 2 / n_boot else f"p={p:.3f}"


def plot_differences(keys, baseline, estimates, samples, diff, path):
    others = [k for k in keys if k != baseline]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for m, ((name, *_), ax) in enumerate(zip(METRICS, axes.flat)):
        deltas = [samples[k][:, m] - samples[baseline][:, m] for k in others]
        ax.axvline(0, color=INK_PRIMARY, linewidth=1, linestyle="--", zorder=1)
        positions = draw_boxes(ax, deltas, others, [estimates[k][name] - estimates[baseline][name] for k in others])
        rows = diff[diff["metric"] == name].set_index("experiment")
        annotate(ax, positions, [f"{rows.loc[k, 'delta']:+.3f} [{rows.loc[k, 'ci_low']:+.3f}, "
                                 f"{rows.loc[k, 'ci_high']:+.3f}]  {format_p(rows.loc[k, 'p_two_sided'], len(deltas[0]))}"
                                 for k in others])
        style_axis(ax, METRIC_TITLES[name], f"model − {EXPERIMENTS[baseline]['label']}")
        if m % 3:
            ax.set_yticklabels([])
    fig.suptitle(f"Paired difference vs {EXPERIMENTS[baseline]['label']} (TEST) — left of the dashed line = "
                 f"worse than baseline · whisker crossing 0 = no significant difference",
                 color=INK_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=13)
    save(fig, path)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[SAVE] {path}", flush=True)

# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Paired study-level bootstrap CIs for the TEST metrics.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", choices=list(EXPERIMENTS), default="swin_positives")
    parser.add_argument("--output-dir", type=Path, default=COMPARISON_DIR / "bootstrap")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(EXPERIMENTS)

    estimates, samples = run_bootstrap(keys, args.n_boot, args.seed)
    ci, diff = summarize(estimates, samples, args.baseline)
    ci.to_csv(args.output_dir / "bootstrap_ci.csv", index=False)
    diff.to_csv(args.output_dir / "bootstrap_paired_diff.csv", index=False)
    np.savez_compressed(args.output_dir / "bootstrap_samples.npz", metrics=[m[0] for m in METRICS],
                        **{k: v for k, v in samples.items()})

    plot_distributions(keys, estimates, samples, ci, args.output_dir / "bootstrap_distributions.png")
    plot_differences(keys, args.baseline, estimates, samples, diff,
                     args.output_dir / "bootstrap_paired_differences.png")

    print()
    print(diff[["label", "metric", "delta", "ci_low", "ci_high", "p_two_sided"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
