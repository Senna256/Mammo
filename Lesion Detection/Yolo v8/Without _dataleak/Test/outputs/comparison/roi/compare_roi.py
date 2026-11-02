#!/usr/bin/env python3
"""Compare the breast crop of the training scripts with the one of mammo_prep.

    training:   detect_breast_roi()  threshold > 5, closing + opening (31 px ellipse),
                bounding box of the largest contour
    mammo_prep: crop_breast()        get_breast_mask(): Otsu + largest connected
                                     component, then bounding box of the mask

Both run on the same image the training pipeline crops: the windowed DICOM
resized to 1024 px on its long side.

Outputs (in this folder):
    roi_stats.csv            one row per image: ROI of each method, IoU between
                             them, area and how much of each lesion stays inside
    roi_summary.png          distribution of the differences
    examples/<image>.png     the most different images plus a few random ones

Example:
    python compare_roi.py --n-negatives 200 --n-examples 12
"""
import argparse
import contextlib
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROI_DIR.parents[2]))  # Test/, for compare_all.py

from compare_all import image_key, load_training_module  # noqa: E402
from mammo_prep import crop_breast, get_breast_mask  # noqa: E402

# Preprocessing code is identical in the six scripts, so any of them will do.
TRAIN = load_training_module("swin_positives")

TRAIN_COLOR, PREP_COLOR, GT_COLOR = "#2a78d6", "#eb6834", "#f5c518"
INK_SECONDARY = "#52514e"

# ============================================================
# THE TWO METHODS
# ============================================================

def training_mask(image):
    """The mask detect_breast_roi() builds internally (largest contour, filled)."""
    _, mask = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def mammo_prep_roi(image):
    """Box (x1, y1, x2, y2) that crop_breast() crops, plus its mask."""
    mask = get_breast_mask(image)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        box = (0, 0, image.shape[1], image.shape[0])
    else:
        (y0, x0), (y1, x1) = coords.min(axis=0), coords.max(axis=0)
        box = (int(x0), int(y0), int(x1) + 1, int(y1) + 1)
    # Same crop as the package function itself.
    assert np.array_equal(crop_breast(image, mask)[0], image[box[1]:box[3], box[0]:box[2]])
    return box, mask

# ============================================================
# PER IMAGE
# ============================================================

def box_iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def fraction_inside(gt, roi):
    """Fraction of the lesion box area that stays inside the crop."""
    w = max(0, min(gt[2], roi[2]) - max(gt[0], roi[0]))
    h = max(0, min(gt[3], roi[3]) - max(gt[1], roi[1]))
    return w * h / max((gt[2] - gt[0]) * (gt[3] - gt[1]), 1e-6)


def load_image(record):
    """Windowed image resized to 1024 (exactly what detect_breast_roi sees) and its GT boxes."""
    image = TRAIN.preprocess_dicom(record["dicom_path"])
    h, w = image.shape
    image, scale = TRAIN.resize_keep_aspect(image, TRAIN.IMG_SIZE)
    gt = [(np.clip(x1, 0, w) * scale, np.clip(y1, 0, h) * scale, np.clip(x2, 0, w) * scale, np.clip(y2, 0, h) * scale)
          for x1, y1, x2, y2 in record["boxes"]]
    return image, [b for b in gt if b[2] > b[0] and b[3] > b[1]]


def analyse(record):
    image, gt = load_image(record)
    area = image.shape[0] * image.shape[1]
    train_box = TRAIN.detect_breast_roi(image)
    prep_box, prep_mask = mammo_prep_roi(image)
    inside_train = [fraction_inside(b, train_box) for b in gt]
    inside_prep = [fraction_inside(b, prep_box) for b in gt]
    mask_train = training_mask(image) > 0
    mask_prep = prep_mask > 0
    return {
        "image": image_key(record["dicom_path"]),
        "n_lesions": len(gt),
        "train_x1": train_box[0], "train_y1": train_box[1], "train_x2": train_box[2], "train_y2": train_box[3],
        "prep_x1": prep_box[0], "prep_y1": prep_box[1], "prep_x2": prep_box[2], "prep_y2": prep_box[3],
        "roi_iou": box_iou(train_box, prep_box),
        "train_area_frac": (train_box[2] - train_box[0]) * (train_box[3] - train_box[1]) / area,
        "prep_area_frac": (prep_box[2] - prep_box[0]) * (prep_box[3] - prep_box[1]) / area,
        "mask_dice": 2 * (mask_train & mask_prep).sum() / max(mask_train.sum() + mask_prep.sum(), 1),
        "train_min_lesion_inside": min(inside_train, default=np.nan),
        "prep_min_lesion_inside": min(inside_prep, default=np.nan),
    }

# ============================================================
# FIGURES
# ============================================================

def draw_box(ax, box, color, style="-", width=1.8):
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor=color, linestyle=style, linewidth=width))


def plot_example(record, row, path):
    image, gt = load_image(record)
    train_box = tuple(int(row[f"train_{c}"]) for c in ("x1", "y1", "x2", "y2"))
    prep_box = tuple(int(row[f"prep_{c}"]) for c in ("x1", "y1", "x2", "y2"))
    mask_train, mask_prep = training_mask(image) > 0, get_breast_mask(image) > 0

    # Pixels in one mask but not in the other, painted on top of the image.
    overlay = np.dstack([image] * 3).astype(np.float32) / 255
    for only, color in ((mask_train & ~mask_prep, TRAIN_COLOR), (mask_prep & ~mask_train, PREP_COLOR)):
        overlay[only] = 0.45 * overlay[only] + 0.55 * np.array(matplotlib.colors.to_rgb(color))

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    axes[0].imshow(overlay)
    draw_box(axes[0], train_box, TRAIN_COLOR, width=2.2)
    draw_box(axes[0], prep_box, PREP_COLOR, "--", width=2.2)
    axes[0].set_title(f"Imatge redimensionada (1024)\nIoU entre ROI = {row['roi_iou']:.3f}, "
                      f"Dice màscares = {row['mask_dice']:.3f}", fontsize=10, loc="left")
    for ax, box, color, name, inside in ((axes[1], train_box, TRAIN_COLOR, "Entrenament (detect_breast_roi)",
                                          row["train_min_lesion_inside"]),
                                         (axes[2], prep_box, PREP_COLOR, "mammo_prep (crop_breast)",
                                          row["prep_min_lesion_inside"])):
        ax.imshow(image[box[1]:box[3], box[0]:box[2]], cmap="gray", vmin=0, vmax=255)
        crop_area = (box[2] - box[0]) * (box[3] - box[1]) / (image.shape[0] * image.shape[1])
        title = f"{name}\nretall {box[2] - box[0]}×{box[3] - box[1]} px ({crop_area:.0%} de la imatge)"
        if gt:
            title += f"\nlesió menys inclosa: {inside:.0%} dins del retall"
        ax.set_title(title, fontsize=10, loc="left", color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    for ax, offset in ((axes[0], (0, 0)), (axes[1], train_box[:2]), (axes[2], prep_box[:2])):
        for b in gt:
            draw_box(ax, (b[0] - offset[0], b[1] - offset[1], b[2] - offset[0], b[3] - offset[1]), GT_COLOR, ":")
        ax.set_xlim(ax.images[0].get_extent()[:2])  # a lesion outside the crop must not be drawn
        ax.set_ylim(ax.images[0].get_extent()[2:])
        ax.set_xticks([])
        ax.set_yticks([])
    legend = [Rectangle((0, 0), 1, 1, color=TRAIN_COLOR, alpha=0.6, label="Només a la màscara d'entrenament"),
              Rectangle((0, 0), 1, 1, color=PREP_COLOR, alpha=0.6, label="Només a la màscara de mammo_prep"),
              Rectangle((0, 0), 1, 1, fill=False, edgecolor=GT_COLOR, linestyle=":", label="Lesió (ground truth)")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(row["image"], x=0.01, ha="left", fontsize=11, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(stats, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(stats["roi_iou"], bins=np.linspace(0, 1, 41), color=INK_SECONDARY)
    axes[0].set_title("IoU entre la ROI d'entrenament i la de mammo_prep", fontsize=10, loc="left")
    axes[0].set_xlabel("IoU")
    axes[0].set_ylabel("Imatges")

    axes[1].scatter(stats["train_area_frac"], stats["prep_area_frac"], s=10, alpha=0.5, color=INK_SECONDARY)
    axes[1].plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Àrea del retall (fracció de la imatge)", fontsize=10, loc="left")
    axes[1].set_xlabel("Entrenament")
    axes[1].set_ylabel("mammo_prep")

    lesions = stats.dropna(subset=["train_min_lesion_inside"])
    thresholds = [1.0, 0.99, 0.9, 0.5]
    labels = ["100 %", "≥ 99 %", "≥ 90 %", "≥ 50 %"]
    x = np.arange(len(thresholds))
    for offset, column, color, name in ((-0.2, "train_min_lesion_inside", TRAIN_COLOR, "Entrenament"),
                                        (0.2, "prep_min_lesion_inside", PREP_COLOR, "mammo_prep")):
        values = [(lesions[column] >= t - 1e-9).mean() for t in thresholds]
        axes[2].bar(x + offset, values, width=0.4, color=color, label=name)
        for xi, v in zip(x + offset, values):
            axes[2].annotate(f"{v:.1%}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title(f"Imatges amb lesió ({len(lesions)}): totes les lesions\ntenen dins del retall almenys...",
                      fontsize=10, loc="left")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-negatives", type=int, default=200, help="TEST images without lesions to add")
    parser.add_argument("--n-examples", type=int, default=12, help="figures per group: most different, lesion cut (4 random always)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test = TRAIN.build_dataset_records(TRAIN.ANNOTATIONS_CSV, TRAIN.NETWORK_IMAGES)
    rng = np.random.default_rng(args.seed)
    positives = [r for r in test if r["boxes"]]
    negatives = [r for r in test if not r["boxes"]]
    records = positives + [negatives[i] for i in rng.choice(len(negatives), args.n_negatives, replace=False)]
    print(f"Analysing {len(records)} TEST images ({len(positives)} with lesions, {args.n_negatives} without)...")

    with Pool(args.workers) as pool:
        stats = pd.DataFrame(pool.map(analyse, records, chunksize=4))
    stats.to_csv(ROI_DIR / "roi_stats.csv", index=False, float_format="%.4f")
    plot_summary(stats, ROI_DIR / "roi_summary.png")

    # Examples: lowest ROI IoU, lesions cut by either method, and a few random ones (typical case).
    n_random = min(4, args.n_examples)
    worst = stats.nsmallest(args.n_examples - n_random, "roi_iou").index.tolist()
    cut_by_any = stats[["train_min_lesion_inside", "prep_min_lesion_inside"]].min(axis=1)
    cut = cut_by_any[cut_by_any < 0.99].drop(worst, errors="ignore").nsmallest(args.n_examples - n_random).index.tolist()
    rest = stats.index.difference(worst + cut)
    chosen = worst + cut + rng.choice(rest, n_random, replace=False).tolist()
    examples_dir = ROI_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)
    for old_figure in examples_dir.glob("*.png"):
        old_figure.unlink()
    for rank, i in enumerate(chosen):
        kind = "different" if i in worst else "lesion_cut" if i in cut else "random"
        plot_example(records[i], stats.loc[i], examples_dir / f"{rank:02d}_{kind}_{stats.loc[i, 'image'].replace('/', '_')}.png")

    lesions = stats.dropna(subset=["train_min_lesion_inside"])
    print(f"\nROI IoU          median {stats['roi_iou'].median():.3f}  p5 {stats['roi_iou'].quantile(0.05):.3f}  "
          f"min {stats['roi_iou'].min():.3f}  (images with IoU < 0.9: {(stats['roi_iou'] < 0.9).sum()})")
    print(f"Mask Dice        median {stats['mask_dice'].median():.3f}  min {stats['mask_dice'].min():.3f}")
    print(f"Training fallback to the whole image: {(stats['train_area_frac'] > 0.99).sum()} images")
    print(f"Crop area        training median {stats['train_area_frac'].median():.1%}, "
          f"mammo_prep median {stats['prep_area_frac'].median():.1%}")
    for column, name in (("train_min_lesion_inside", "training"), ("prep_min_lesion_inside", "mammo_prep")):
        print(f"Lesions cut ({name:10s}): {(lesions[column] < 0.99).sum()} of {len(lesions)} images "
              f"have a lesion with < 99 % inside the crop")
    print(f"\n[SAVED] {ROI_DIR / 'roi_stats.csv'}\n[SAVED] {ROI_DIR / 'roi_summary.png'}\n[SAVED] {examples_dir}/")


if __name__ == "__main__":
    main()
