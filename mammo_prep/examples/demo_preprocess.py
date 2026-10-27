#!/usr/bin/env python3
"""Shows, step by step, what preprocess_mammogram() does to one mammogram.

Saves a figure with five panels, from the original image to the model input:
    1. original DICOM pixels, without any processing (load_dicom)
    2. windowed, at the original resolution (window_dicom)
    3. resized to 1024 on the long side, with the breast crop (detect_breast_roi)
    4. the crop
    5. model input (preprocess_mammogram), with the boxes moved by transform_boxes()
The boxes are also brought back to the DICOM with inverse_transform_boxes() and printed.

Examples:
    python demo_preprocess.py
    python demo_preprocess.py --dicom path/to/image.dicom --box 2492 1878 2596 1974
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mammo_prep import (detect_breast_roi, inverse_transform_boxes, load_dicom, preprocess_mammogram,
                        resize_keep_aspect, transform_boxes, window_dicom)

# VinDr-Mammo image with a lesion in the fatty periphery of the breast.
DEFAULT_DICOM = "/home/enric/Datasets/Original/vindr/images/02c9621d5af7d0485e20c071f605678c/40344cff30da62d2e72446944c156932.dicom"
DEFAULT_BOXES = [(2492.6, 1878.3, 2596.6, 1974.3)]

ROI_COLOR, BOX_COLOR = "#2a78d6", "#f5c518"


def draw_box(ax, box, color, style="-"):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linestyle=style, linewidth=1.8))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dicom", default=DEFAULT_DICOM)
    parser.add_argument("--box", type=float, nargs=4, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                        help="lesion box in DICOM pixels; repeat for several")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--out", default="demo_preprocess.png")
    args = parser.parse_args()
    boxes = args.box if args.box else (DEFAULT_BOXES if args.dicom == DEFAULT_DICOM else [])

    # The whole preprocessing is this call; the rest only reproduces the intermediate steps to show them.
    image, transform = preprocess_mammogram(args.dicom, img_size=args.img_size)
    final_boxes = transform_boxes(boxes, transform)
    back_to_dicom = inverse_transform_boxes(final_boxes, transform)

    raw, ds = load_dicom(args.dicom)
    windowed = window_dicom(args.dicom)
    resized, _ = resize_keep_aspect(windowed, args.img_size)
    x1, y1, x2, y2 = detect_breast_roi(resized)
    assert (x1, y1, x2, y2) == transform["roi"]
    resize_factor = resized.shape[1] / windowed.shape[1]

    fig, axes = plt.subplots(1, 5, figsize=(23, 6.5))
    # Original pixels shown from their minimum to their maximum, with no window.
    axes[0].imshow(raw, cmap="gray", vmin=raw.min(), vmax=raw.max())
    axes[0].set_title(f"1. DICOM original\n{raw.shape[1]}×{raw.shape[0]} px, valors {raw.min():.0f}–{raw.max():.0f}\n"
                      f"{getattr(ds, 'PhotometricInterpretation', '?')}, {getattr(ds, 'BitsStored', '?')} bits",
                      loc="left")
    axes[1].imshow(windowed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Windowing (breast_tissue)\n{windowed.shape[1]}×{windowed.shape[0]} px, uint8 0–255",
                      loc="left")
    axes[2].imshow(resized, cmap="gray", vmin=0, vmax=255)
    draw_box(axes[2], transform["roi"], ROI_COLOR)
    axes[2].set_title(f"3. Redimensionada ({resized.shape[1]}×{resized.shape[0]})\n"
                      f"en blau, el retall de detect_breast_roi", loc="left")
    axes[3].imshow(resized[y1:y2, x1:x2], cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(f"4. Retall de la mama\n{x2 - x1}×{y2 - y1} px", loc="left")
    axes[4].imshow(image)
    axes[4].set_title(f"5. Entrada del model ({args.img_size}×{args.img_size}×3)\n"
                      f"escala {transform['scale']:.3f}, farciment {transform['pad']}", loc="left")

    for box in boxes:
        draw_box(axes[0], box, BOX_COLOR, ":")
        draw_box(axes[1], box, BOX_COLOR, ":")
        draw_box(axes[2], [c * resize_factor for c in box], BOX_COLOR, ":")
        draw_box(axes[3], [box[0] * resize_factor - x1, box[1] * resize_factor - y1,
                           box[2] * resize_factor - x1, box[3] * resize_factor - y1], BOX_COLOR, ":")
    for box in final_boxes:
        draw_box(axes[4], box, BOX_COLOR, ":")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)

    print(f"DICOM:            {args.dicom}")
    print(f"transform:        {transform}")
    for original, final, back in zip(boxes, final_boxes, back_to_dicom):
        print(f"box DICOM {tuple(round(c, 1) for c in original)} -> image {tuple(round(float(c), 1) for c in final)} "
              f"-> DICOM {tuple(round(float(c), 1) for c in back)}")
    print(f"[SAVED] {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
