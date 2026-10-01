#!/usr/bin/env python3

from pathlib import Path
import argparse
import importlib.util
import json
import random

import cv2
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIG
# ============================================================

NETWORK_IMAGES = Path("/home/enric/Datasets/Original/vindr/images")
LOCAL_DATASET = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo")
ANNOTATIONS_CSV = Path("/home/enric/Datasets/Original/vindr/finding_annotations.csv")

SWIN_PREDICTIONS = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_swin/evaluation/predictions.csv"
)

VIT_PREDICTIONS = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_vit/evaluation_vit_yolo/predictions.csv"
)

OUTPUT_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/Compare vit swin/visual_comparison/outputs"
)

# This is the actual ViT training script. Its preprocessing functions are
# imported directly so this visualization cannot silently drift from training.
TRAINING_SCRIPT = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/train_vit_vindr_v3.py"
)

IMG_SIZE = 1024
IOU_THRESHOLD = 0.30
CONF_THRESHOLD = 0.25
N_PER_CATEGORY = 10
SEED = 42


# ============================================================
# LOAD THE EXACT TRAINING PREPROCESSING
# ============================================================

def load_training_module():
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "vit_training_module",
        TRAINING_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {TRAINING_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "resize_keep_aspect",
        "resize_and_pad",
        "transform_boxes_to_roi",
        "transform_boxes_to_padded_image",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "Training script does not contain the expected "
            f"coordinate/resize functions: {missing}"
        )

    return module


PREPROCESS = load_training_module()


# ============================================================
# IOU / MATCHING
# ============================================================

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt(gt_boxes, predictions, threshold):
    if not predictions:
        return [], [], list(range(len(gt_boxes))), []

    if not gt_boxes:
        return [], list(range(len(predictions))), [], []

    candidates = []

    for pi, prediction in enumerate(predictions):
        for gi, gt in enumerate(gt_boxes):
            iou = box_iou(prediction, gt)
            if iou >= threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(reverse=True)

    matched_predictions = set()
    matched_gt = set()
    matches = []

    for iou, pi, gi in candidates:
        if pi in matched_predictions or gi in matched_gt:
            continue

        matched_predictions.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, iou))

    tp = sorted(matched_predictions)
    fp = [
        i for i in range(len(predictions))
        if i not in matched_predictions
    ]
    fn = [
        i for i in range(len(gt_boxes))
        if i not in matched_gt
    ]

    return tp, fp, fn, matches


# ============================================================
# PREDICTIONS
# ============================================================

def load_predictions(path):
    df = pd.read_csv(path)

    required = {
        "image_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    df["image_id"] = df["image_id"].astype(str)
    df["confidence"] = df["confidence"].astype(float)

    return df[
        df["confidence"] >= CONF_THRESHOLD
    ].copy()


def get_predictions(df, image_id):
    rows = df[
        df["image_id"] == str(image_id)
    ]

    return [
        [
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
            float(row["confidence"]),
        ]
        for _, row in rows.iterrows()
    ]


# ============================================================
# EXACT GT LABELS
# ============================================================

def load_yolo_labels(image_id):
    path = (
        LOCAL_DATASET
        / "labels"
        / "test"
        / f"{image_id}.txt"
    )

    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            try:
                labels.append([float(v) for v in values])
            except ValueError:
                continue

    if not labels:
        return np.zeros((0, 5), dtype=np.float32)

    return np.asarray(labels, dtype=np.float32)


def labels_to_xyxy(labels, img_size):
    boxes = []

    for _, cx, cy, w, h in labels:
        cx *= img_size
        cy *= img_size
        w *= img_size
        h *= img_size

        boxes.append([
            float(cx - w / 2),
            float(cy - h / 2),
            float(cx + w / 2),
            float(cy + h / 2),
        ])

    return boxes


# ============================================================
# CORRECTED BREAST MASK / WINDOWING
# ============================================================

def get_breast_mask(image):
    """
    Detect the breast before windowing.

    The previous ROI detector thresholded the already-windowed image
    at 5. That can fail when the background is non-zero after windowing.
    Here we first separate tissue/background with Otsu and connected
    components.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image, got {image.shape}"
        )

    valid = image[np.isfinite(image)]

    if valid.size == 0:
        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    p_low, p_high = np.percentile(
        valid,
        [1, 99],
    )

    if p_high <= p_low:
        return np.ones_like(
            image,
            dtype=np.uint8,
        )

    normalized = np.clip(
        (image - p_low) / (p_high - p_low),
        0,
        1,
    )

    normalized = (
        normalized * 255
    ).astype(np.uint8)

    _, binary = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    border = np.concatenate(
        [
            binary[0, :],
            binary[-1, :],
            binary[:, 0],
            binary[:, -1],
        ]
    )

    if np.mean(border > 0) > 0.5:
        binary = cv2.bitwise_not(binary)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (31, 31),
        ),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        ),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return binary

    image_area = image.shape[0] * image.shape[1]
    candidates = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area < 0.01 * image_area:
            continue

        if width < 0.05 * image.shape[1]:
            continue

        if height < 0.05 * image.shape[0]:
            continue

        candidates.append(
            (area, label)
        )

    if not candidates:
        return binary

    _, selected_label = max(
        candidates,
        key=lambda item: item[0],
    )

    mask = np.zeros_like(
        binary,
        dtype=np.uint8,
    )

    mask[
        labels == selected_label
    ] = 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        contour = max(
            contours,
            key=cv2.contourArea,
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

    return mask


def corrected_preprocess_dicom(dicom_path):
    """
    Corrected preprocessing requested for visualization:

        DICOM
        -> MONOCHROME1 correction
        -> breast/background mask
        -> breast-only window calculation
        -> LINEAR window
        -> background = 0
        -> uint8
    """

    from mammo_prep.windowing import preprocess_window

    ds = pydicom.dcmread(
        str(dicom_path)
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    if getattr(
        ds,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":
        image = image.max() - image

    mask = get_breast_mask(
        image
    )

    if not np.any(mask > 0):
        raise RuntimeError(
            "Breast mask is empty."
        )

    masked_image = np.zeros_like(
        image,
        dtype=np.float32,
    )

    masked_image[
        mask > 0
    ] = image[
        mask > 0
    ]

    windowed = preprocess_window(
        masked_image,
        dicom_dataset=ds,
        method="breast_tissue",
        voi_func="LINEAR",
        exclude_background=True,
        output_dtype=np.uint8,
    )

    windowed[
        mask == 0
    ] = 0

    return (
        windowed,
        mask,
        image,
    )


def preprocess_for_visualization(
    dicom_path,
    image_id,
):
    """
    Build the corrected breast-cropped 1024x1024 image and transform GT
    into that corrected coordinate system.

    IMPORTANT:
    the existing Swin/ViT CSV predictions were produced with the
    training preprocessing. Therefore they are shown separately, but
    they are not re-inferred on this corrected image.
    """

    (
        windowed,
        breast_mask,
        original_image,
    ) = corrected_preprocess_dicom(
        dicom_path
    )

    resized, first_scale = (
        PREPROCESS.resize_keep_aspect(
            windowed,
            IMG_SIZE,
        )
    )

    mask_resized = cv2.resize(
        breast_mask,
        (
            resized.shape[1],
            resized.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    ys, xs = np.where(
        mask_resized > 0
    )

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(
            "Corrected breast mask is empty after resize."
        )

    roi_x1 = int(xs.min())
    roi_y1 = int(ys.min())
    roi_x2 = int(xs.max()) + 1
    roi_y2 = int(ys.max()) + 1

    crop = resized[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]

    crop_h, crop_w = crop.shape[:2]

    labels = load_yolo_labels(
        image_id
    )

    labels_roi = PREPROCESS.transform_boxes_to_roi(
        labels,
        resized.shape[1],
        resized.shape[0],
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
    )

    crop_bgr = cv2.cvtColor(
        crop,
        cv2.COLOR_GRAY2BGR,
    )

    (
        final_image,
        final_scale,
        pad_x,
        pad_y,
    ) = PREPROCESS.resize_and_pad(
        crop_bgr,
        IMG_SIZE,
    )

    labels_final = (
        PREPROCESS.transform_boxes_to_padded_image(
            labels_roi,
            crop_w,
            crop_h,
            final_scale,
            pad_x,
            pad_y,
            IMG_SIZE,
        )
    )

    gt_boxes = labels_to_xyxy(
        labels_final,
        IMG_SIZE,
    )

    roi_debug = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.rectangle(
        roi_debug,
        (roi_x1, roi_y1),
        (roi_x2 - 1, roi_y2 - 1),
        (255, 255, 255),
        3,
    )

    original_debug = cv2.normalize(
        original_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    stages = {
        "01_original_dicom.png": original_debug,
        "02_breast_mask.png": breast_mask,
        "03_windowed_corrected.png": windowed,
        "04_resized.png": resized,
        "05_roi.png": roi_debug,
        "06_cropped_breast.png": crop,
        "07_model_input_1024.png": final_image.copy(),
    }

    metadata = {
        "dicom": str(dicom_path),
        "image_id": str(image_id),
        "original_shape": list(original_image.shape),
        "resized_shape": list(resized.shape),
        "corrected_roi": [
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,
        ],
        "crop_shape": [
            crop_h,
            crop_w,
        ],
        "final_shape": list(final_image.shape),
        "first_resize_scale": float(first_scale),
        "final_resize_scale": float(final_scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "num_gt_boxes": len(gt_boxes),
        "preprocessing": (
            "Breast/background separation with Otsu + connected "
            "components, breast-only windowing, ROI crop, resize "
            "and padding to 1024."
        ),
        "prediction_warning": (
            "Existing Swin/ViT predictions were generated using "
            "the original training preprocessing. They are not "
            "new predictions on this corrected image."
        ),
    }

    return (
        final_image,
        gt_boxes,
        stages,
        metadata,
    )


# ============================================================
# DICOM PATH
# ============================================================

def build_study_map(annotations):
    rows = annotations[
        annotations["split"].astype(str) == "test"
    ]

    return {
        str(row["image_id"]): str(row["study_id"])
        for _, row in rows.iterrows()
    }


def find_dicom(image_id, study_map):
    image_id = str(image_id)
    study_id = study_map.get(image_id)

    if study_id is not None:
        path = (
            NETWORK_IMAGES
            / study_id
            / f"{image_id}.dicom"
        )

        if path.exists():
            return path

    matches = list(
        NETWORK_IMAGES.glob(
            f"*/{image_id}.dicom"
        )
    )

    return matches[0] if matches else None


# ============================================================
# IMAGE-LEVEL TP/FN/FP/TN
# ============================================================

def classify(gt_boxes, predictions):
    tp, fp, fn, matches = match_predictions_to_gt(
        gt_boxes,
        predictions,
        IOU_THRESHOLD,
    )

    if gt_boxes:
        image_class = "TP" if matches else "FN"
    else:
        image_class = "FP" if predictions else "TN"

    return {
        "class": image_class,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
    }


def category_for(swin_class, vit_class):
    return {
        ("TP", "FN"): "swin_tp_vit_fn",
        ("FN", "TP"): "swin_fn_vit_tp",
        ("FP", "TN"): "swin_fp_vit_tn",
        ("TN", "FP"): "swin_tn_vit_fp",
        ("TP", "TP"): "both_tp",
        ("FN", "FN"): "both_fn",
        ("FP", "FP"): "both_fp",
        ("TN", "TN"): "both_tn",
    }.get((swin_class, vit_class))


# ============================================================
# FIGURE: GT | SWIN | VIT
# ============================================================

def draw_box(ax, box, color, label, linewidth=2.5):
    x1, y1, x2, y2 = box[:4]

    if x2 <= x1 or y2 <= y1:
        return

    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
    )

    ax.text(
        x1,
        max(2, y1 - 5),
        label,
        color=color,
        fontsize=8,
        fontweight="bold",
        bbox={
            "facecolor": "black",
            "alpha": 0.65,
            "pad": 1.5,
            "edgecolor": "none",
        },
    )


def draw_gt(ax, image, gt_boxes):
    ax.imshow(image, cmap="gray")
    ax.set_title(
        "Ground Truth (corrected preprocessing)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axis("off")

    for i, box in enumerate(gt_boxes, 1):
        draw_box(
            ax,
            box,
            "lime",
            f"GT {i}",
            3.0,
        )


def draw_predictions(
    ax,
    image,
    predictions,
    title,
    color,
):
    ax.imshow(image, cmap="gray")
    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
    )
    ax.axis("off")

    for i, prediction in enumerate(predictions, 1):
        draw_box(
            ax,
            prediction,
            color,
            f"{i}: {prediction[4]:.2f}",
            2.5,
        )


def save_comparison(
    image,
    image_id,
    gt_boxes,
    swin_boxes,
    vit_boxes,
    swin_info,
    vit_info,
    category,
    path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
    )

    draw_gt(
        axes[0],
        image,
        gt_boxes,
    )

    draw_predictions(
        axes[1],
        image,
        swin_boxes,
        "Swin + YOLO predictions*",
        "cyan",
    )

    draw_predictions(
        axes[2],
        image,
        vit_boxes,
        "ViT + YOLO predictions*",
        "magenta",
    )

    fig.suptitle(
        (
            f"{image_id} | {category} | "
            f"IoU ≥ {IOU_THRESHOLD:.2f}"
        ),
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.060,
        (
            "* Existing predictions were generated with the training "
            "preprocessing; the corrected image is for visualization "
            "and is not a new inference.",
        ),
        ha="center",
        fontsize=8,
    )

    fig.text(
        0.5,
        0.035,
        (
            f"Swin: {swin_info['class']}    |    "
            f"ViT: {vit_info['class']}    |    "
            f"GT: {len(gt_boxes)}    |    "
            f"Swin predictions: {len(swin_boxes)}    |    "
            f"ViT predictions: {len(vit_boxes)}"
        ),
        ha="center",
        fontsize=10,
    )

    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color="lime",
                lw=3,
                label="Ground truth",
            ),
            plt.Line2D(
                [0],
                [0],
                color="cyan",
                lw=3,
                label="Swin + YOLO prediction",
            ),
            plt.Line2D(
                [0],
                [0],
                color="magenta",
                lw=3,
                label="ViT + YOLO prediction",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.88,
        bottom=0.15,
        wspace=0.04,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    global IOU_THRESHOLD
    global CONF_THRESHOLD

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-per-category",
        type=int,
        default=N_PER_CATEGORY,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    IOU_THRESHOLD = args.iou
    CONF_THRESHOLD = args.conf

    random.seed(args.seed)
    np.random.seed(args.seed)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "VISUAL COMPARISON: "
        "GT vs SWIN + YOLO vs ViT + YOLO"
    )
    print("=" * 70)
    print(
        f"IoU threshold:        {IOU_THRESHOLD}"
    )
    print(
        f"Confidence threshold: {CONF_THRESHOLD}"
    )
    print(
        f"Images/category:      {args.n_per_category}"
    )
    print(
        f"Training preprocessing: {TRAINING_SCRIPT}"
    )
    print(
        f"Output:               {OUTPUT_DIR}"
    )
    print()

    print("[LOAD] Annotations...")
    annotations = pd.read_csv(
        ANNOTATIONS_CSV
    )

    print("[LOAD] Swin predictions...")
    swin_predictions = load_predictions(
        SWIN_PREDICTIONS
    )

    print("[LOAD] ViT predictions...")
    vit_predictions = load_predictions(
        VIT_PREDICTIONS
    )

    study_map = build_study_map(
        annotations
    )

    test_ids = set(
        annotations.loc[
            annotations["split"].astype(str) == "test",
            "image_id",
        ].astype(str)
    )

    common_ids = sorted(
        test_ids
        & set(swin_predictions["image_id"])
        & set(vit_predictions["image_id"])
    )

    print(
        f"[INFO] Common test images: "
        f"{len(common_ids)}"
    )

    categories = {
        "swin_tp_vit_fn": [],
        "swin_fn_vit_tp": [],
        "swin_fp_vit_tn": [],
        "swin_tn_vit_fp": [],
        "both_tp": [],
        "both_fn": [],
        "both_fp": [],
        "both_tn": [],
    }

    csv_rows = []

    print(
        "[EVAL] Reconstructing the exact model input..."
    )

    for n, image_id in enumerate(
        common_ids,
        1,
    ):
        dicom_path = find_dicom(
            image_id,
            study_map,
        )

        if dicom_path is None:
            print(
                f"[WARNING] DICOM not found: {image_id}"
            )
            continue

        try:
            (
                model_input,
                gt_boxes,
                stages,
                metadata,
            ) = preprocess_for_visualization(
                dicom_path,
                image_id,
            )

            swin_boxes = get_predictions(
                swin_predictions,
                image_id,
            )

            vit_boxes = get_predictions(
                vit_predictions,
                image_id,
            )

            swin_info = classify(
                gt_boxes,
                swin_boxes,
            )

            vit_info = classify(
                gt_boxes,
                vit_boxes,
            )

            category = category_for(
                swin_info["class"],
                vit_info["class"],
            )

            if category is None:
                continue

            categories[
                category
            ].append(
                {
                    "image_id": image_id,
                    "image": model_input,
                    "gt": gt_boxes,
                    "swin": swin_boxes,
                    "vit": vit_boxes,
                    "swin_info": swin_info,
                    "vit_info": vit_info,
                    "stages": stages,
                    "metadata": metadata,
                }
            )

            csv_rows.append(
                {
                    "image_id": image_id,
                    "category": category,
                    "swin_class": swin_info["class"],
                    "vit_class": vit_info["class"],
                    "gt_boxes": len(gt_boxes),
                    "swin_predictions": len(swin_boxes),
                    "vit_predictions": len(vit_boxes),
                    "swin_tp": len(swin_info["tp"]),
                    "swin_fp": len(swin_info["fp"]),
                    "swin_fn": len(swin_info["fn"]),
                    "vit_tp": len(vit_info["tp"]),
                    "vit_fp": len(vit_info["fp"]),
                    "vit_fn": len(vit_info["fn"]),
                }
            )

        except Exception as exc:
            print(
                f"[WARNING] {image_id}: "
                f"{type(exc).__name__}: {exc}"
            )

        if n % 100 == 0 or n == len(common_ids):
            print(
                f"       {n}/{len(common_ids)}"
            )

    print()
    print("=" * 70)
    print("CATEGORY COUNTS")
    print("=" * 70)

    for category, items in categories.items():
        print(
            f"{category:25s}: {len(items):5d}"
        )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        OUTPUT_DIR / "comparison_results.csv",
        index=False,
    )

    print()
    print(
        "[VIS] Generating GT / Swin / ViT figures..."
    )

    for category, items in categories.items():
        if not items:
            continue

        random.shuffle(items)

        category_dir = (
            OUTPUT_DIR
            / category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected = items[
            :args.n_per_category
        ]

        print(
            f"  {category}: "
            f"{len(selected)} images"
        )

        for item in selected:
            image_id = item["image_id"]

            save_comparison(
                item["image"],
                image_id,
                item["gt"],
                item["swin"],
                item["vit"],
                item["swin_info"],
                item["vit_info"],
                category,
                category_dir
                / f"{image_id}.png",
            )

            debug_dir = (
                OUTPUT_DIR
                / "preprocessing_debug"
                / image_id
            )

            debug_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for name, stage in item["stages"].items():
                cv2.imwrite(
                    str(debug_dir / name),
                    stage,
                )

            (
                debug_dir / "metadata.json"
            ).write_text(
                json.dumps(
                    item["metadata"],
                    indent=2,
                ),
                encoding="utf-8",
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        f"Figures: {OUTPUT_DIR}"
    )
    print(
        "Exact preprocessing stages:"
    )
    print(
        f"{OUTPUT_DIR / 'preprocessing_debug'}"
    )
    print()
    print(
        "The image shown in the three-panel figures is:"
    )
    print(
        "preprocessing_debug/<image_id>/"
        "05_model_input_1024.png"
    )


if __name__ == "__main__":
    main()
