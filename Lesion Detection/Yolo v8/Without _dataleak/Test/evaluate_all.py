#!/usr/bin/env python3
"""Evaluate the six Without _dataleak experiments on the official CSV TEST split.

Experiments: {Swin, ViT} x {full dataset, 4k images, positives}.

Each experiment is evaluated with its OWN training script: the script's
build_dataset_records() gives the TEST records, its Dataset class applies the
exact preprocessing the model was trained with, and its model class loads the
checkpoint. Only the metric code below is shared, so differences between
experiments come from the models (and their preprocessing), not the evaluator.

Pipeline per experiment:
    TEST DICOM -> script preprocessing -> model.eval() -> NMS -> predictions.csv
Metrics (from cached predictions, recomputed with --metrics-only):
    TPR vs IoU at fixed false positives per image (FPPI), TPR with all
    detections, AP vs IoU, mAP@0.50:0.95.

Why NMS is applied
------------------
The YOLOv8 Detect head scores every anchor (21,504 per 1024x1024 image), so a
single lesion produces many overlapping boxes. Without NMS every duplicate of a
correct box that cannot be matched to a ground truth counts as a false
positive, which makes FPPI, precision and AP meaningless. NMS keeps one box per
object, as YOLOv8 is designed to be used. Settings follow Ultralytics
validation defaults and the previous evaluation scripts: conf 0.001, NMS IoU
0.70, max 300 detections per image. All six experiments use identical settings.

Ground-truth denominator
------------------------
TPR uses the number of CSV boxes in TEST (identical for all experiments). Boxes
removed by an experiment's breast crop are therefore counted as missed
lesions, so a crop that cuts lesions is penalised instead of shrinking the
denominator. Matching is done in each model's 1024x1024 input space.
"""
import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.nms import non_max_suppression

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

EXPERIMENTS = {
    "swin_full": {"label": "Swin · full", "backbone": "Swin", "regime": "full",
                  "script": "swin/Train/full_dataset/train_full_swin.py"},
    "swin_4k": {"label": "Swin · 4k", "backbone": "Swin", "regime": "4k",
                "script": "swin/Train/4k images/train_4k_swin.py"},
    "swin_positives": {"label": "Swin · positives", "backbone": "Swin", "regime": "positives",
                       "script": "swin/Train/positives/train_swin_positives.py"},
    "vit_full": {"label": "ViT · full", "backbone": "ViT", "regime": "full",
                 "script": "vit/Train/full_dataset/train_full_vit.py"},
    "vit_4k": {"label": "ViT · 4k", "backbone": "ViT", "regime": "4k",
               "script": "vit/Train/4k images/train_4k_vit.py"},
    "vit_positives": {"label": "ViT · positives", "backbone": "ViT", "regime": "positives",
                      "script": "vit/Train/positives/train_vit_positives.py"},
}

CONF_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300

IOU_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55,
                  0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MAP_THRESHOLDS = [t for t in IOU_THRESHOLDS if t >= 0.50]  # 0.50:0.05:0.95
FPPI_POINTS = [0.25, 0.5, 1.0, 2.0, 4.0]
MAIN_FPPI = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot styling: color = training regime, line style = backbone.
REGIME_COLORS = {"full": "#2a78d6", "4k": "#eb6834", "positives": "#1baf7a"}
BACKBONE_STYLES = {"Swin": {"linestyle": "-", "marker": "o"},
                   "ViT": {"linestyle": "--", "marker": "s"}}
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# ============================================================
# TRAINING SCRIPT LOADING
# ============================================================

def load_training_module(key):
    path = EXPERIMENTS_ROOT / EXPERIMENTS[key]["script"]
    spec = importlib.util.spec_from_file_location(f"train_{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_class(module, predicate):
    matches = [obj for name, obj in vars(module).items()
               if inspect.isclass(obj) and obj.__module__ == module.__name__ and predicate(name)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching class in {module.__name__}, found {len(matches)}")
    return matches[0]


def image_key(dicom_path):
    path = Path(dicom_path)
    return f"{path.parent.name}/{path.stem}"

# ============================================================
# INFERENCE
# ============================================================

def run_inference(key, checkpoint_name, batch_size, workers, limit, output_dir):
    module = load_training_module(key)
    model_cls = find_class(module, lambda n: n.endswith("YOLO"))
    dataset_cls = find_class(module, lambda n: n.startswith("Vindr") and n.endswith("Dataset"))
    img_size = module.IMG_SIZE

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test_records = module.build_dataset_records(module.ANNOTATIONS_CSV, module.NETWORK_IMAGES)
    if limit:
        # Smoke-test subset with both positives and negatives.
        positives = [r for r in test_records if r["boxes"]]
        negatives = [r for r in test_records if not r["boxes"]]
        test_records = positives[: limit // 2] + negatives[: limit - limit // 2]
    csv_boxes = sum(len(r["boxes"]) for r in test_records)

    checkpoint_path = module.OUTPUT_DIR / checkpoint_name
    print(f"[{key}] Checkpoint: {checkpoint_path}", flush=True)
    print(f"[{key}] TEST images: {len(test_records):,} | CSV boxes: {csv_boxes:,}", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        model = model_cls(img_size=img_size, num_classes=1, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(DEVICE).eval()
    model.detect.stride = torch.tensor(module.DETECT_STRIDES, dtype=torch.float32, device=DEVICE)

    loader = DataLoader(
        dataset_cls(test_records, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=module.collate_fn,
    )

    prediction_rows, gt_rows = [], []
    start = time.time()
    for images, targets, paths in tqdm(loader, desc=f"[{key}] TEST", unit="batch", dynamic_ncols=True):
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            output = model(images.to(DEVICE, non_blocking=True))
        # Eval-mode Detect returns (decoded predictions, raw head outputs).
        decoded = output[0] if isinstance(output, (tuple, list)) else output
        detections = non_max_suppression(
            decoded.float(),
            conf_thres=CONF_THRESHOLD,
            iou_thres=NMS_IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
        )
        for i, det in enumerate(detections):
            image_id = image_key(paths[i])
            for x1, y1, x2, y2, conf, _ in det.cpu().numpy().tolist():
                prediction_rows.append((image_id, conf, x1, y1, x2, y2))

        boxes = targets["bboxes"].numpy() * img_size
        for batch_index, (cx, cy, w, h) in zip(targets["batch_idx"].numpy(), boxes):
            gt_rows.append((image_key(paths[int(batch_index)]), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    elapsed = time.time() - start

    exp_dir = output_dir / key
    exp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows, columns=["image_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(
        exp_dir / "predictions.csv", index=False)
    pd.DataFrame(gt_rows, columns=["image_id", "x1", "y1", "x2", "y2"]).to_csv(
        exp_dir / "ground_truth.csv", index=False)
    meta = {
        "experiment": key,
        "training_script": str(EXPERIMENTS_ROOT / EXPERIMENTS[key]["script"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        "img_size": img_size,
        "test_images": len(test_records),
        "csv_gt_boxes": csv_boxes,
        "gt_boxes_after_preprocessing": len(gt_rows),
        "predictions_after_nms": len(prediction_rows),
        "inference_seconds": round(elapsed, 1),
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{key}] Done in {elapsed / 60:.1f} min | predictions after NMS: {len(prediction_rows):,} | "
          f"GT boxes kept by preprocessing: {len(gt_rows)}/{csv_boxes}", flush=True)

    del model, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# METRICS
# ============================================================

def iou_matrix(boxes_a, boxes_b):
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
    """All-point interpolated AP (same definition as the previous evaluation scripts)."""
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


def compute_metrics(key, output_dir):
    exp_dir = output_dir / key
    meta = json.loads((exp_dir / "meta.json").read_text())
    predictions = pd.read_csv(exp_dir / "predictions.csv")
    ground_truth = pd.read_csv(exp_dir / "ground_truth.csv")
    total_gt, n_images = meta["csv_gt_boxes"], meta["test_images"]

    rows = []
    for iou_threshold in IOU_THRESHOLDS:
        tp = match_detections(predictions, ground_truth, iou_threshold)
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(~tp)
        row = {
            "experiment": key,
            "iou": iou_threshold,
            "AP": average_precision(cum_tp, cum_fp, total_gt),
            "TPR_all_detections": float(cum_tp[-1] / total_gt) if len(tp) else 0.0,
            "TP_all_detections": int(cum_tp[-1]) if len(tp) else 0,
            "FP_all_detections": int(cum_fp[-1]) if len(tp) else 0,
        }
        for fppi in FPPI_POINTS:
            row[f"TPR@{fppi:g}FPPI"] = tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, fppi) if len(tp) else 0.0
        rows.append(row)
    per_iou = pd.DataFrame(rows)
    per_iou.to_csv(exp_dir / "metrics_by_iou.csv", index=False)

    at = per_iou.set_index("iou")
    summary = {
        "experiment": key,
        "label": EXPERIMENTS[key]["label"],
        "checkpoint_epoch": meta["checkpoint_epoch"],
        "test_images": n_images,
        "csv_gt_boxes": total_gt,
        "gt_lost_by_preprocessing": total_gt - meta["gt_boxes_after_preprocessing"],
        "predictions_after_nms": meta["predictions_after_nms"],
        "AP@0.10": at.loc[0.10, "AP"],
        "AP@0.30": at.loc[0.30, "AP"],
        "AP@0.50": at.loc[0.50, "AP"],
        "mAP@0.50:0.95": float(at.loc[MAP_THRESHOLDS, "AP"].mean()),
    }
    for iou_threshold in (0.30, 0.50):
        for fppi in (0.5, 1.0, 2.0):
            summary[f"TPR@IoU{iou_threshold:.2f}_{fppi:g}FPPI"] = at.loc[iou_threshold, f"TPR@{fppi:g}FPPI"]
        summary[f"TPR@IoU{iou_threshold:.2f}_all_detections"] = at.loc[iou_threshold, "TPR_all_detections"]
    (exp_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
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

def write_report(summary, comparison_dir, checkpoint_name):
    lines = [
        "Without _dataleak — TEST evaluation of the six experiments",
        "=" * 70,
        f"Checkpoint: {checkpoint_name}",
        f"NMS: conf {CONF_THRESHOLD}, IoU {NMS_IOU_THRESHOLD}, max {MAX_DETECTIONS} detections/image",
        "Why NMS: the YOLOv8 head scores all 21,504 anchors, so each lesion yields many",
        "overlapping boxes; without NMS the duplicates are counted as false positives,",
        "making FPPI, precision and AP meaningless. Same settings for all experiments.",
        "TPR denominator: CSV TEST boxes (identical for all experiments); boxes removed",
        "by an experiment's breast crop count as missed lesions.",
        "FPPI = false positives per TEST image (all images, positive and negative).",
        "",
        summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "Caveats",
        "-" * 70,
        "- ViT full / ViT 4k crop with mammo_prep get_breast_mask; the other four use",
        "  detect_breast_roi. Each model is evaluated with its own preprocessing.",
        "- ViT positives selected best.pt on a positives-only validation set; Swin",
        "  positives on the full validation set.",
    ]
    path = comparison_dir / "comparison_report.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"[SAVE] {path}", flush=True)

# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all Without _dataleak experiments on the CSV TEST split.")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--checkpoint", choices=["best.pt", "last.pt"], default="best.pt")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only N TEST images (smoke test).")
    parser.add_argument("--metrics-only", action="store_true", help="Recompute metrics/plots from cached predictions.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)

    if not args.metrics_only:
        for key in args.experiments:
            run_inference(key, args.checkpoint, args.batch, args.workers, args.limit, args.output_dir)

    per_iou_tables, summaries = [], []
    for key in args.experiments:
        per_iou, summary = compute_metrics(key, args.output_dir)
        per_iou_tables.append(per_iou)
        summaries.append(summary)
    per_iou = pd.concat(per_iou_tables, ignore_index=True)
    summary = pd.DataFrame(summaries)

    comparison_dir = args.output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    per_iou.to_csv(comparison_dir / "metrics_by_iou_all.csv", index=False)
    summary.to_csv(comparison_dir / "summary_comparison.csv", index=False)
    make_plots(per_iou, comparison_dir)
    write_report(summary, comparison_dir, args.checkpoint)

    print()
    print(summary[["label", "checkpoint_epoch", "gt_lost_by_preprocessing", "AP@0.50", "mAP@0.50:0.95",
                   "TPR@IoU0.30_1FPPI", "TPR@IoU0.50_1FPPI", "TPR@IoU0.50_all_detections"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
