#!/usr/bin/env python3
"""How the breast crop changes with the size of the circle (structuring element).

Runs detect_breast_roi() with the same steps (threshold > background_threshold(), closing, opening,
largest contour, 5 % minimum) but a different circle diameter, on the same
TEST images as compare_roi.py. Size 0 = no closing/opening.

Outputs (in this folder):
    kernel_sensitivity.csv          one row per image and circle size
    kernel_sensitivity_summary.csv  one row per circle size
    kernel_sensitivity.png          the summary against the circle size
    kernel_sensitivity_example.png  mask and crop of one image for several sizes

Example:
    python kernel_sensitivity.py --sizes 0 7 15 21 31 41 51 71 101
"""
import argparse
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from compare_roi import ROI_DIR, TRAIN, box_iou, fraction_inside, load_image
from compare_all import image_key
from mammo_prep import background_threshold, detect_breast_roi

REFERENCE_SIZE = 31  # the size used to train the models
SIZES = [0, 7, 15, 21, 31, 41, 51, 71, 101]
EXAMPLE_SIZES = [0, 7, 31, 101]


def breast_mask_and_box(image, size):
    """detect_breast_roi() with a circle of diameter `size`; also returns the opened mask."""
    h, w = image.shape
    _, mask = cv2.threshold(image, background_threshold(image), 255, cv2.THRESH_BINARY)
    if size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, (0, 0, w, h), 0
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bw < 0.05 * w or bh < 0.05 * h:
        return mask, (0, 0, w, h), len(contours)
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return mask, (0, 0, w, h), len(contours)
    return mask, (x1, y1, x2, y2), len(contours)


def analyse(args):
    record, sizes = args
    image, gt = load_image(record)
    area = image.shape[0] * image.shape[1]
    reference = detect_breast_roi(image)
    rows = []
    for size in sizes:
        _, box, n_regions = breast_mask_and_box(image, size)
        if size == REFERENCE_SIZE:
            assert box == reference  # same function as the one used in training
        inside = [fraction_inside(b, box) for b in gt]
        rows.append({
            "image": image_key(record["dicom_path"]), "size": size, "n_lesions": len(gt),
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "area_frac": (box[2] - box[0]) * (box[3] - box[1]) / area,
            "whole_image": box == (0, 0, image.shape[1], image.shape[0]),
            "n_regions": n_regions,
            "iou_vs_31": box_iou(box, reference),
            "min_lesion_inside": min(inside, default=np.nan),
        })
    return rows


def summarise(stats):
    groups = stats.groupby("size")
    with_lesions = stats[stats["n_lesions"] > 0].groupby("size")
    return pd.DataFrame({
        "imatges": groups.size(),
        "imatges_amb_lesio": with_lesions.size(),
        "lesio_tallada (<99% dins)": with_lesions["min_lesion_inside"].apply(lambda v: (v < 0.99).sum()),
        "lesio_tallada (<50% dins)": with_lesions["min_lesion_inside"].apply(lambda v: (v < 0.5).sum()),
        "lesio_totalment_fora": with_lesions["min_lesion_inside"].apply(lambda v: (v <= 0).sum()),
        "imatge_sencera": groups["whole_image"].sum(),
        "area_mediana_retall": groups["area_frac"].median(),
        "regions_mediana": groups["n_regions"].median(),
        "iou_mediana_vs_31": groups["iou_vs_31"].median(),
        "imatges_iou_<0.9_vs_31": groups["iou_vs_31"].apply(lambda v: (v < 0.9).sum()),
    }).reset_index().rename(columns={"size": "diametre_cercle_px"})


def plot_summary(summary, path):
    x = summary["diametre_cercle_px"]
    panels = [
        ("lesio_tallada (<99% dins)", "Imatges amb alguna lesió tallada\n(< 99 % dins del retall)", "#e5484d"),
        ("imatge_sencera", "Imatges on no es troba la mama\n(es queda la imatge sencera)", "#52514e"),
        ("area_mediana_retall", "Àrea mediana del retall\n(fracció de la imatge)", "#2a78d6"),
        ("imatges_iou_<0.9_vs_31", f"Imatges amb un retall diferent del de {REFERENCE_SIZE} px\n(IoU < 0,9)", "#eb6834"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, (column, title, color) in zip(axes, panels):
        ax.plot(x, summary[column], "-o", color=color, linewidth=1.8)
        for xi, v in zip(x, summary[column]):
            ax.annotate(f"{v:.1%}" if column == "area_mediana_retall" else f"{v:.0f}", (xi, v),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(REFERENCE_SIZE, color="#c3c2b7", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Diàmetre del cercle (px, imatge de 1024)")
        ax.set_xticks(x)
        ax.margins(y=0.2)
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    n = int(summary["imatges"].iloc[0])
    fig.suptitle(f"Sensibilitat del retall a la mida del cercle  ·  {n} imatges de TEST "
                 f"({int(summary['imatges_amb_lesio'].iloc[0])} amb lesió)  ·  línia discontínua = {REFERENCE_SIZE} px "
                 f"(el que es fa servir)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(record, path):
    image, gt = load_image(record)
    fig, axes = plt.subplots(1, len(EXAMPLE_SIZES), figsize=(4.2 * len(EXAMPLE_SIZES), 5.6))
    for ax, size in zip(axes, EXAMPLE_SIZES):
        mask, (x1, y1, x2, y2), n_regions = breast_mask_and_box(image, size)
        rgb = np.dstack([image] * 3).astype(np.float32) / 255
        rgb[mask > 0] = 0.55 * rgb[mask > 0] + 0.45 * np.array([0.16, 0.47, 0.84])
        ax.imshow(rgb)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#eb6834", linewidth=2))
        for b in gt:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor="#f5c518",
                                   linestyle=":", linewidth=1.8))
        label = "sense morfologia" if size == 0 else f"cercle de {size} px"
        ax.set_title(f"{label}\n{n_regions} regions, retall {(x2 - x1) * (y2 - y1) / image.size:.0%}",
                     fontsize=10, loc="left")
        ax.axis("off")
    fig.suptitle(f"{image_key(record['dicom_path'])}  ·  blau = màscara després de tancament + obertura, "
                 f"taronja = retall, groc = lesió", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    sizes = sorted(set(args.sizes) | {REFERENCE_SIZE})

    # Same images as compare_roi.py (roi_stats.csv).
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test = TRAIN.build_dataset_records(TRAIN.ANNOTATIONS_CSV, TRAIN.NETWORK_IMAGES)
    by_key = {image_key(r["dicom_path"]): r for r in test}
    records = [by_key[k] for k in pd.read_csv(ROI_DIR / "roi_stats.csv")["image"]]
    print(f"Analysing {len(records)} TEST images x {len(sizes)} circle sizes...")

    with Pool(args.workers) as pool:
        stats = pd.DataFrame([row for rows in pool.map(analyse, [(r, sizes) for r in records], chunksize=4)
                              for row in rows])
    summary = summarise(stats)
    stats.to_csv(ROI_DIR / "kernel_sensitivity.csv", index=False, float_format="%.4f")
    summary.to_csv(ROI_DIR / "kernel_sensitivity_summary.csv", index=False, float_format="%.4f")
    plot_summary(summary, ROI_DIR / "kernel_sensitivity.png")

    # Example: the image with lesions whose crop changes most across sizes.
    spread = stats[stats["n_lesions"] > 0].groupby("image")["area_frac"].agg(lambda v: v.max() - v.min())
    plot_example(by_key[spread.idxmax()], ROI_DIR / "kernel_sensitivity_example.png")

    print(summary.to_string(index=False))
    for name in ("kernel_sensitivity.csv", "kernel_sensitivity_summary.csv", "kernel_sensitivity.png",
                 "kernel_sensitivity_example.png"):
        print(f"[SAVED] {ROI_DIR / name}")


if __name__ == "__main__":
    main()
