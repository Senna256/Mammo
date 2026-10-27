#!/usr/bin/env python3
"""Shows, step by step, how detect_breast_roi() finds the breast.

Top row: the structuring element (a 31 px ellipse) and each step of the mask:
    threshold > 5  ->  closing  ->  opening  ->  largest contour and its box
Bottom row: zoom on the area where closing adds most pixels and on the area
where opening removes most, with the ellipse drawn to scale.

    closing = dilate then erode: fills gaps narrower than the ellipse
    opening = erode then dilate: removes pieces narrower than the ellipse

Examples:
    python demo_breast_roi.py
    python demo_breast_roi.py --dicom path/to/image.dicom --out roi.png
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Ellipse, Rectangle

from mammo_prep import detect_breast_roi, resize_keep_aspect, window_dicom

DEFAULT_DICOM = "/home/enric/Datasets/Original/vindr/images/014d843f13a9d86fec945664e2541fe6/e82b6324ff951df4158627cf430996b7.dicom"

# Same values as detect_breast_roi().
THRESHOLD = 5
KERNEL_SIZE = 31

ADDED_COLOR, REMOVED_COLOR, BOX_COLOR, KERNEL_COLOR = "#1baf7a", "#e5484d", "#2a78d6", "#f5c518"
ZOOM = 220  # side of the zoom windows, in pixels of the 1024 image


def mask_steps(image):
    """The masks that detect_breast_roi() builds, one per step."""
    _, threshold = cv2.threshold(image, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    return kernel, threshold > 0, closed > 0, opened > 0, largest


def busiest_window(changed, size):
    """Top-left corner of the size x size window with most changed pixels."""
    density = cv2.boxFilter(changed.astype(np.float32), -1, (size, size), normalize=False)
    cy, cx = np.unravel_index(np.argmax(density), density.shape)
    h, w = changed.shape
    return int(np.clip(cx - size // 2, 0, max(w - size, 0))), int(np.clip(cy - size // 2, 0, max(h - size, 0)))


def paint(image, layers):
    """Grey image with coloured pixel layers on top."""
    rgb = np.dstack([image] * 3).astype(np.float32) / 255
    for pixels, color in layers:
        rgb[pixels] = 0.25 * rgb[pixels] + 0.75 * np.array(to_rgb(color))
    return rgb


def show_zoom(ax, rgb, corner, title):
    x, y = corner
    ax.imshow(rgb[y:y + ZOOM, x:x + ZOOM], extent=(x, x + ZOOM, y + ZOOM, y), interpolation="nearest")
    # The ellipse to scale, in a corner of the zoom.
    ax.add_patch(Ellipse((x + 25, y + 25), KERNEL_SIZE, KERNEL_SIZE, fill=False, edgecolor=KERNEL_COLOR, linewidth=2))
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dicom", default=DEFAULT_DICOM)
    parser.add_argument("--out", default="demo_breast_roi.png")
    args = parser.parse_args()

    image, _ = resize_keep_aspect(window_dicom(args.dicom), 1024)
    kernel, threshold, closed, opened, largest = mask_steps(image)
    box = detect_breast_roi(image)
    added = closed & ~threshold    # filled by the closing
    removed = closed & ~opened     # deleted by the opening
    final = np.zeros(image.shape, np.uint8)
    if largest is not None:
        cv2.drawContours(final, [largest], -1, 1, -1)
    dropped = opened & (final == 0)  # survives the opening but is not the largest region

    fig = plt.figure(figsize=(22, 11.5), layout="constrained")
    grid = fig.add_gridspec(2, 5, height_ratios=[1, 0.9])
    top = [fig.add_subplot(grid[0, i]) for i in range(5)]

    top[0].imshow(kernel, cmap=ListedColormap(["#ffffff", KERNEL_COLOR]), interpolation="nearest")
    top[0].set_xticks(np.arange(-0.5, KERNEL_SIZE, 1), minor=True)
    top[0].set_yticks(np.arange(-0.5, KERNEL_SIZE, 1), minor=True)
    top[0].grid(which="minor", color="#c3c2b7", linewidth=0.4)
    top[0].tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    top[0].set_title(f"Element estructurant: el·lipse {KERNEL_SIZE}×{KERNEL_SIZE} px\n"
                     f"(un cercle de {KERNEL_SIZE} px, {KERNEL_SIZE / image.shape[1]:.0%} de l'amplada)",
                     fontsize=10, loc="left")

    top[1].imshow(threshold, cmap="gray")
    top[1].set_title(f"1. Llindar: píxel > {THRESHOLD}\n"
                     f"{cv2.connectedComponents(threshold.astype(np.uint8))[0] - 1} regions separades",
                     fontsize=10, loc="left")
    top[2].imshow(paint(threshold.astype(np.uint8) * 255, [(added, ADDED_COLOR)]))
    top[2].set_title(f"2. Tancament (dilatar → erosionar)\nen verd, {added.sum():,} px omplerts", fontsize=10, loc="left")
    top[3].imshow(paint(opened.astype(np.uint8) * 255, [(removed, REMOVED_COLOR)]))
    top[3].set_title(f"3. Obertura (erosionar → dilatar)\nen vermell, {removed.sum():,} px eliminats",
                     fontsize=10, loc="left")
    top[4].imshow(paint(image, [(dropped, "#8a8986")]))
    if largest is not None:
        top[4].plot(largest[:, 0, 0], largest[:, 0, 1], color=BOX_COLOR, linewidth=1.2)
    x1, y1, x2, y2 = box
    top[4].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=BOX_COLOR, linewidth=2.2))
    top[4].set_title(f"4. Contorn més gran → retall\n({x1}, {y1}, {x2}, {y2})", fontsize=10, loc="left")
    for ax in top[1:]:
        ax.set_xticks([])
        ax.set_yticks([])

    close_rgb = paint(np.where(threshold, 255, 0).astype(np.uint8), [(added, ADDED_COLOR)])
    open_rgb = paint(np.where(closed, 255, 0).astype(np.uint8), [(removed, REMOVED_COLOR)])
    image_rgb = paint(image, [(added, ADDED_COLOR), (removed, REMOVED_COLOR)])
    close_corner = busiest_window(added, ZOOM)
    open_corner = busiest_window(removed, ZOOM)
    bottom = [fig.add_subplot(grid[1, i]) for i in range(4)]
    show_zoom(bottom[0], close_rgb, close_corner,
              "Zoom del tancament (màscara)\nforats i escletxes més estrets que el cercle → s'omplen")
    show_zoom(bottom[1], image_rgb, close_corner, "El mateix zoom sobre la imatge")
    show_zoom(bottom[2], open_rgb, open_corner,
              "Zoom de l'obertura (màscara)\ntrossos més estrets que el cercle → s'eliminen")
    show_zoom(bottom[3], image_rgb, open_corner, "El mateix zoom sobre la imatge")
    for ax, corner in ((top[2], close_corner), (top[3], open_corner)):
        ax.add_patch(Rectangle(corner, ZOOM, ZOOM, fill=False, edgecolor=KERNEL_COLOR, linewidth=1.5, linestyle="--"))

    text = fig.add_subplot(grid[1, 4])
    text.axis("off")
    text.text(0, 0.95, "Com funciona\n\n"
              "Imagina que passeges el cercle groc\nper la màscara.\n\n"
              "Tancament: on el cercle no hi cap\n(escletxes i forats estrets dins\nla mama), s'omple.\n\n"
              "Obertura: on el cercle no hi cap\n(text, marcadors, filaments\nprims), s'esborra.\n\n"
              "En gris (pas 4): regions que\nsobreviuen però no són la més gran.\n\n"
              "En groc discontinu: zona ampliada.",
              va="top", fontsize=11, linespacing=1.35)

    fig.suptitle(f"detect_breast_roi pas a pas  ·  {Path(args.dicom).parent.name}/{Path(args.dicom).stem}",
                 x=0.01, ha="left", fontsize=13)
    fig.savefig(args.out, dpi=110)
    print(f"Closing filled {added.sum():,} px; opening removed {removed.sum():,} px; box {box}")
    print(f"[SAVED] {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
