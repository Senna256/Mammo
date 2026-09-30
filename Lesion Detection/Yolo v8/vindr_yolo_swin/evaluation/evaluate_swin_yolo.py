#!/usr/bin/env python3

import csv
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils.nms import non_max_suppression

from swin_yolo_train_v2 import (
    VindrSwinDataset,
    SwinYOLO,
    collate_fn,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_IMAGES = Path(
    "/home/enric/Datasets/Original/vindr/images"
)

ANNOTATIONS_CSV = Path(
    "/home/enric/Datasets/Original/vindr/finding_annotations.csv"
)

LOCAL_DATASET = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

CHECKPOINT = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/train_v2/best.pt"
)

OUTPUT_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/evaluation"
)

VISUALIZATION_DIR = (
    OUTPUT_DIR / "visualizations"
)

IMG_SIZE = 1024

BATCH_SIZE = 1

NUM_WORKERS = 4

NUM_CLASSES = 1

CONF_THRESHOLD = 0.001

NMS_IOU_THRESHOLD = 0.70

MAX_DETECTIONS = 300

VISUAL_CONF_THRESHOLD = 0.25

MAX_VISUALIZATIONS = 100

MAX_EVAL_IMAGES = None

IOU_THRESHOLDS = np.arange(
    0.1,
    0.96,
    0.05,
)

CURVE_IOU = 0.50

FROC_FPPI_POINTS = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IOU
# ============================================================

def box_iou(
    boxes1,
    boxes2,
):

    if (
        len(boxes1) == 0
        or
        len(boxes2) == 0
    ):

        return np.zeros(
            (
                len(boxes1),
                len(boxes2),
            ),
            dtype=np.float32,
        )

    boxes1 = np.asarray(
        boxes1,
        dtype=np.float32,
    )

    boxes2 = np.asarray(
        boxes2,
        dtype=np.float32,
    )

    area1 = (
        np.maximum(
            0,
            boxes1[:, 2]
            -
            boxes1[:, 0],
        )
        *
        np.maximum(
            0,
            boxes1[:, 3]
            -
            boxes1[:, 1],
        )
    )

    area2 = (
        np.maximum(
            0,
            boxes2[:, 2]
            -
            boxes2[:, 0],
        )
        *
        np.maximum(
            0,
            boxes2[:, 3]
            -
            boxes2[:, 1],
        )
    )

    x1 = np.maximum(
        boxes1[:, None, 0],
        boxes2[None, :, 0],
    )

    y1 = np.maximum(
        boxes1[:, None, 1],
        boxes2[None, :, 1],
    )

    x2 = np.minimum(
        boxes1[:, None, 2],
        boxes2[None, :, 2],
    )

    y2 = np.minimum(
        boxes1[:, None, 3],
        boxes2[None, :, 3],
    )

    intersection = (
        np.maximum(
            0,
            x2 - x1,
        )
        *
        np.maximum(
            0,
            y2 - y1,
        )
    )

    union = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection
    )

    return intersection / (
        union + 1e-9
    )


# ============================================================
# BOX CONVERSION
# ============================================================

def xywhn_to_xyxy(
    box,
):

    box = np.asarray(
        box,
        dtype=np.float32,
    )

    if box.shape[0] == 5:

        x_center = box[1]
        y_center = box[2]
        width = box[3]
        height = box[4]

    elif box.shape[0] == 4:

        x_center = box[0]
        y_center = box[1]
        width = box[2]
        height = box[3]

    else:

        raise ValueError(
            f"Unexpected box shape: {box.shape}"
        )

    x1 = (
        x_center
        -
        width / 2.0
    )

    y1 = (
        y_center
        -
        height / 2.0
    )

    x2 = (
        x_center
        +
        width / 2.0
    )

    y2 = (
        y_center
        +
        height / 2.0
    )

    return np.array(
        [
            x1 * IMG_SIZE,
            y1 * IMG_SIZE,
            x2 * IMG_SIZE,
            y2 * IMG_SIZE,
        ],
        dtype=np.float32,
    )


# ============================================================
# AP
# ============================================================

def compute_ap(
    recall,
    precision,
):

    if len(recall) == 0:

        return 0.0

    recall = np.asarray(
        recall,
        dtype=np.float64,
    )

    precision = np.asarray(
        precision,
        dtype=np.float64,
    )

    mrec = np.concatenate(
        [
            [0.0],
            recall,
            [1.0],
        ]
    )

    mpre = np.concatenate(
        [
            [0.0],
            precision,
            [0.0],
        ]
    )

    for i in range(
        len(mpre) - 1,
        0,
        -1,
    ):

        mpre[i - 1] = max(
            mpre[i - 1],
            mpre[i],
        )

    indices = np.where(
        mrec[1:]
        !=
        mrec[:-1]
    )[0]

    ap = np.sum(
        (
            mrec[indices + 1]
            -
            mrec[indices]
        )
        *
        mpre[indices + 1]
    )

    return float(ap)


# ============================================================
# DETECT OUTPUT DECODING
# ============================================================

def decode_detect_output(
    model,
    output,
):

    if isinstance(
        output,
        dict,
    ):

        if (
            "boxes" in output
            and
            "scores" in output
        ):

            return model.detect._inference(
                output
            )

    if isinstance(
        output,
        tuple,
    ):

        if len(output) == 1:

            output = output[0]

        elif (
            len(output) > 0
            and
            isinstance(
                output[0],
                torch.Tensor,
            )
        ):

            output = output[0]

    return output


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
):

    print(
        f"[CHECKPOINT] Loading: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if "model" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain 'model'."
        )

    state_dict = checkpoint[
        "model"
    ]

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    if missing_keys:

        print(
            "[CHECKPOINT] Missing keys:",
            len(missing_keys),
        )

        for key in missing_keys[:20]:

            print(
                f"  {key}"
            )

    if unexpected_keys:

        print(
            "[CHECKPOINT] Unexpected keys:",
            len(unexpected_keys),
        )

        for key in unexpected_keys[:20]:

            print(
                f"  {key}"
            )

    if "epoch" in checkpoint:

        print(
            f"[CHECKPOINT] Epoch: "
            f"{checkpoint['epoch']}"
        )

    if "best_val_loss" in checkpoint:

        print(
            f"[CHECKPOINT] Best validation loss: "
            f"{checkpoint['best_val_loss']}"
        )

    return checkpoint


# ============================================================
# VISUALIZATION
# ============================================================

def draw_boxes(
    image,
    boxes,
    color,
    thickness=2,
):

    image = image.copy()

    for box in boxes:

        if len(box) < 4:

            continue

        x1, y1, x2, y2 = (
            box[:4]
        )

        x1 = int(
            round(x1)
        )

        y1 = int(
            round(y1)
        )

        x2 = int(
            round(x2)
        )

        y2 = int(
            round(y2)
        )

        cv2.rectangle(
            image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            thickness,
        )

    return image


def draw_predictions(
    image,
    predictions,
):

    image = image.copy()

    for prediction in predictions:

        if len(prediction) < 5:

            continue

        x1, y1, x2, y2 = (
            prediction[:4]
        )

        conf = float(
            prediction[4]
        )

        if conf < VISUAL_CONF_THRESHOLD:

            continue

        x1 = int(
            round(x1)
        )

        y1 = int(
            round(y1)
        )

        x2 = int(
            round(x2)
        )

        y2 = int(
            round(y2)
        )

        cv2.rectangle(
            image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            f"{conf:.3f}",
            (
                x1,
                max(
                    y1 - 5,
                    15,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# GT EXTRACTION
# ============================================================

def get_ground_truth_for_image(
    targets,
    batch_index,
):

    batch_idx = (
        targets["batch_idx"]
        .detach()
        .cpu()
        .numpy()
    )

    boxes = (
        targets["bboxes"]
        .detach()
        .cpu()
        .numpy()
    )

    mask = (
        batch_idx
        ==
        batch_index
    )

    image_boxes = boxes[
        mask
    ]

    if image_boxes.size == 0:

        return np.zeros(
            (
                0,
                4,
            ),
            dtype=np.float32,
        )

    ground_truth_boxes = []

    for box in image_boxes:

        ground_truth_boxes.append(
            xywhn_to_xyxy(
                box
            )
        )

    return np.asarray(
        ground_truth_boxes,
        dtype=np.float32,
    ).reshape(
        -1,
        4,
    )


# ============================================================
# AP STATISTICS
# ============================================================

def collect_ap_statistics(
    predictions,
    ground_truths,
    iou_threshold,
):

    prediction_records = []

    total_gt = 0

    for gts in ground_truths.values():

        total_gt += len(gts)

    for image_id, preds in predictions.items():

        for prediction in preds:

            if len(prediction) < 5:

                continue

            prediction_records.append(
                (
                    image_id,
                    float(
                        prediction[4]
                    ),
                    prediction[:4],
                )
            )

    prediction_records.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    if total_gt == 0:

        return (
            0.0,
            0,
            0,
            0,
        )

    matched = {
        image_id: set()
        for image_id in ground_truths
    }

    tp_values = []

    fp_values = []

    for (
        image_id,
        confidence,
        pred_box,
    ) in prediction_records:

        gts = ground_truths.get(
            image_id,
            np.zeros(
                (
                    0,
                    4,
                ),
                dtype=np.float32,
            ),
        )

        if len(gts) == 0:

            tp_values.append(0)

            fp_values.append(1)

            continue

        ious = box_iou(
            np.asarray(
                [pred_box],
                dtype=np.float32,
            ),
            gts,
        )[0]

        order = np.argsort(
            -ious
        )

        found_match = False

        for gt_index in order:

            gt_index = int(
                gt_index
            )

            if (
                gt_index
                in
                matched[image_id]
            ):

                continue

            if (
                ious[gt_index]
                >=
                iou_threshold
            ):

                matched[
                    image_id
                ].add(
                    gt_index
                )

                found_match = True

                break

        if found_match:

            tp_values.append(1)

            fp_values.append(0)

        else:

            tp_values.append(0)

            fp_values.append(1)

    if len(tp_values) == 0:

        return (
            0.0,
            0,
            0,
            total_gt,
        )

    tp_values = np.asarray(
        tp_values,
        dtype=np.float64,
    )

    fp_values = np.asarray(
        fp_values,
        dtype=np.float64,
    )

    cumulative_tp = np.cumsum(
        tp_values
    )

    cumulative_fp = np.cumsum(
        fp_values
    )

    recall = (
        cumulative_tp
        /
        max(
            total_gt,
            1,
        )
    )

    precision = (
        cumulative_tp
        /
        np.maximum(
            cumulative_tp
            +
            cumulative_fp,
            1e-12,
        )
    )

    ap = compute_ap(
        recall,
        precision,
    )

    final_tp = int(
        cumulative_tp[-1]
    )

    final_fp = int(
        cumulative_fp[-1]
    )

    final_fn = int(
        total_gt
        -
        final_tp
    )

    return (
        ap,
        final_tp,
        final_fp,
        final_fn,
    )


# ============================================================
# CURVES (PR / FROC / TPR-vs-IoU)
# ============================================================

def build_prediction_records(
    predictions,
):

    prediction_records = []

    for image_id, preds in predictions.items():

        for prediction in preds:

            if len(prediction) < 5:
                continue

            prediction_records.append(
                (
                    image_id,
                    float(prediction[4]),
                    prediction[:4],
                )
            )

    prediction_records.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return prediction_records


def match_predictions(
    prediction_records,
    ground_truths,
    iou_threshold,
):

    matched = {
        image_id: set()
        for image_id in ground_truths
    }

    tp_values = []
    fp_values = []
    confidences = []

    for (
        image_id,
        confidence,
        pred_box,
    ) in prediction_records:

        gts = ground_truths.get(
            image_id,
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
        )

        confidences.append(confidence)

        if len(gts) == 0:

            tp_values.append(0)
            fp_values.append(1)
            continue

        ious = box_iou(
            np.asarray(
                [pred_box],
                dtype=np.float32,
            ),
            gts,
        )[0]

        order = np.argsort(
            -ious
        )

        found_match = False

        for gt_index in order:

            gt_index = int(gt_index)

            if gt_index in matched[image_id]:
                continue

            if ious[gt_index] >= iou_threshold:

                matched[image_id].add(
                    gt_index
                )

                found_match = True
                break

        if found_match:

            tp_values.append(1)
            fp_values.append(0)

        else:

            tp_values.append(0)
            fp_values.append(1)

    return (
        np.asarray(
            tp_values,
            dtype=np.float64,
        ),
        np.asarray(
            fp_values,
            dtype=np.float64,
        ),
        np.asarray(
            confidences,
            dtype=np.float64,
        ),
    )


def compute_pr_and_froc_curves(
    predictions,
    ground_truths,
    iou_threshold,
):

    total_gt = int(
        sum(
            len(gts)
            for gts in ground_truths.values()
        )
    )

    num_images = int(
        len(predictions)
    )

    prediction_records = build_prediction_records(
        predictions
    )

    if (
        total_gt == 0
        or len(prediction_records) == 0
        or num_images == 0
    ):

        return {
            "recall": np.array(
                [0.0],
                dtype=np.float64,
            ),
            "precision": np.array(
                [1.0],
                dtype=np.float64,
            ),
            "fpi": np.array(
                [0.0],
                dtype=np.float64,
            ),
            "sensitivity": np.array(
                [0.0],
                dtype=np.float64,
            ),
            "conf_thresholds": np.array(
                [],
                dtype=np.float64,
            ),
            "ap": 0.0,
            "froc_points": {
                str(fppi): 0.0
                for fppi in FROC_FPPI_POINTS
            },
        }

    tp_values, fp_values, confidences = (
        match_predictions(
            prediction_records,
            ground_truths,
            iou_threshold,
        )
    )

    cumulative_tp = np.cumsum(
        tp_values
    )

    cumulative_fp = np.cumsum(
        fp_values
    )

    recall = (
        cumulative_tp
        / max(total_gt, 1)
    )

    precision = (
        cumulative_tp
        /
        np.maximum(
            cumulative_tp
            + cumulative_fp,
            1e-12,
        )
    )

    ap = compute_ap(
        recall,
        precision,
    )

    sensitivity = recall.copy()

    fpi = (
        cumulative_fp
        / max(num_images, 1)
    )

    # Include origin so the curve starts at (0, 0).
    fpi_curve = np.concatenate(
        [
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            fpi,
        ]
    )

    sensitivity_curve = np.concatenate(
        [
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            sensitivity,
        ]
    )

    froc_points = {}

    for target_fppi in FROC_FPPI_POINTS:

        valid = (
            fpi_curve
            <= target_fppi
        )

        if np.any(valid):

            sens_value = float(
                np.max(
                    sensitivity_curve[valid]
                )
            )

        else:

            sens_value = 0.0

        froc_points[
            str(target_fppi)
        ] = sens_value

    return {
        "recall": recall,
        "precision": precision,
        "fpi": fpi_curve,
        "sensitivity": sensitivity_curve,
        "conf_thresholds": confidences,
        "ap": float(ap),
        "froc_points": froc_points,
    }


def save_curve_csv(
    path,
    headers,
    rows,
):

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save_pr_curve_plot(
    path,
    recall,
    precision,
    ap,
    iou_threshold,
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=(
            f"AP@IoU={iou_threshold:.2f}"
            f" = {ap:.4f}"
        ),
    )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_froc_curve_plot(
    path,
    fpi,
    sensitivity,
    sampled_points,
    iou_threshold,
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpi,
        sensitivity,
        linewidth=2,
        label=f"IoU={iou_threshold:.2f}",
    )

    sampled_x = [
        float(x)
        for x in sampled_points.keys()
    ]

    sampled_y = [
        float(y)
        for y in sampled_points.values()
    ]

    plt.scatter(
        sampled_x,
        sampled_y,
        s=40,
        zorder=3,
        label="Standard FPI points",
    )

    plt.xscale("log")
    plt.xlim(0.2, 10.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positives Per Image (FPI)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("FROC Curve")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_tpr_iou_curve_plot(
    path,
    iou_thresholds,
    tpr_values,
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        iou_thresholds,
        tpr_values,
        marker="o",
        linewidth=2,
    )

    plt.xlim(
        float(np.min(iou_thresholds)),
        float(np.max(iou_thresholds)),
    )

    plt.ylim(0.0, 1.0)
    plt.xlabel("IoU Threshold")
    plt.ylabel("TPR")
    plt.title("TPR vs IoU Threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SWIN-TINY + YOLOv8 TEST EVALUATION"
    )

    print(
        "=" * 70
        + "\n"
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print()

    print(
        f"Network images: "
        f"{NETWORK_IMAGES}"
    )

    print(
        f"Annotations CSV: "
        f"{ANNOTATIONS_CSV}"
    )

    print(
        f"Local labels: "
        f"{LOCAL_DATASET}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT}"
    )

    print(
        f"Evaluation limit: "
        f"{MAX_EVAL_IMAGES if MAX_EVAL_IMAGES is not None else 'ALL'}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if VISUALIZATION_DIR.exists():

        print(
            "[INIT] Removing old visualizations..."
        )

        for file in VISUALIZATION_DIR.iterdir():

            if file.is_file():

                file.unlink()

            elif file.is_dir():

                shutil.rmtree(
                    file
                )

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DATASET
    # ========================================================

    print(
        "[INIT] Loading test dataset..."
    )

    test_dataset = VindrSwinDataset(
        network_images=NETWORK_IMAGES,
        annotations_csv=ANNOTATIONS_CSV,
        local_dataset=LOCAL_DATASET,
        split="test",
        img_size=IMG_SIZE,
    )

    total_test_images = len(
        test_dataset
    )

    print(
        f"[INIT] Test images: "
        f"{total_test_images:,}"
    )

    if MAX_EVAL_IMAGES is None:

        eval_count = total_test_images

    else:

        eval_count = min(
            MAX_EVAL_IMAGES,
            total_test_images,
        )

    print(
        f"[INIT] Images to evaluate: "
        f"{eval_count:,}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        collate_fn=collate_fn,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n[INIT] Building model..."
    )

    print(
        "[INIT] Loading "
        "swin_tiny_patch4_window7_224..."
    )

    model = SwinYOLO(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        pretrained=False,
    )

    print(
        "[INIT] Creating YOLOv8 Detect head..."
    )

    load_checkpoint(
        model,
        CHECKPOINT,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "[INIT] Model ready."
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    print(
        "\n[EVAL] Running inference..."
    )

    predictions = {}

    ground_truths = {}

    prediction_rows = []

    visualization_count = 0

    evaluated_images = 0

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="TEST",
            unit="batch",
            total=(
                eval_count
                if BATCH_SIZE == 1
                else None
            ),
        )

        for (
            images,
            targets,
            paths,
        ) in progress:

            if (
                MAX_EVAL_IMAGES is not None
                and
                evaluated_images
                >=
                MAX_EVAL_IMAGES
            ):

                break

            remaining = (
                MAX_EVAL_IMAGES
                -
                evaluated_images
                if MAX_EVAL_IMAGES is not None
                else None
            )

            if (
                remaining is not None
                and
                images.shape[0]
                >
                remaining
            ):

                images = images[
                    :remaining
                ]

            current_batch_size = (
                images.shape[0]
            )

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):

                output = model(
                    images
                )

                decoded = (
                    decode_detect_output(
                        model,
                        output,
                    )
                )

            detections = (
                non_max_suppression(
                    decoded,
                    conf_thres=CONF_THRESHOLD,
                    iou_thres=NMS_IOU_THRESHOLD,
                    classes=None,
                    agnostic=False,
                    max_det=MAX_DETECTIONS,
                )
            )

            for batch_index in range(
                current_batch_size
            ):

                path = paths[
                    batch_index
                ]

                image_id = Path(
                    path
                ).stem

                pred = (
                    detections[
                        batch_index
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                predictions[
                    image_id
                ] = pred

                gt = (
                    get_ground_truth_for_image(
                        targets,
                        batch_index,
                    )
                )

                ground_truths[
                    image_id
                ] = gt

                for detection in pred:

                    if len(detection) < 5:

                        continue

                    prediction_rows.append(
                        {
                            "image_id": image_id,
                            "x1": float(
                                detection[0]
                            ),
                            "y1": float(
                                detection[1]
                            ),
                            "x2": float(
                                detection[2]
                            ),
                            "y2": float(
                                detection[3]
                            ),
                            "confidence": float(
                                detection[4]
                            ),
                            "class": int(
                                detection[5]
                            )
                            if len(detection) > 5
                            else 0,
                        }
                    )

                # ============================================
                # VISUALIZATION
                # ============================================

                if (
                    visualization_count
                    <
                    MAX_VISUALIZATIONS
                    and
                    (
                        len(gt) > 0
                        or
                        any(
                            len(detection) >= 5
                            and
                            float(
                                detection[4]
                            )
                            >=
                            VISUAL_CONF_THRESHOLD
                            for detection in pred
                        )
                    )
                ):

                    image_tensor = (
                        images[
                            batch_index
                        ]
                        .detach()
                        .float()
                        .cpu()
                    )

                    image_array = (
                        image_tensor
                        .permute(
                            1,
                            2,
                            0,
                        )
                        .numpy()
                    )

                    image_array = (
                        np.clip(
                            image_array,
                            0,
                            1,
                        )
                        * 255.0
                    ).astype(
                        np.uint8
                    )

                    if (
                        image_array.ndim == 3
                        and
                        image_array.shape[2] == 3
                    ):

                        image_bgr = cv2.cvtColor(
                            image_array,
                            cv2.COLOR_RGB2BGR,
                        )

                    else:

                        image_bgr = cv2.cvtColor(
                            image_array[:, :, 0],
                            cv2.COLOR_GRAY2BGR,
                        )

                    # GT = GREEN

                    image_bgr = draw_boxes(
                        image_bgr,
                        gt,
                        (
                            0,
                            255,
                            0,
                        ),
                        2,
                    )

                    # PRED = RED

                    image_bgr = draw_predictions(
                        image_bgr,
                        pred,
                    )

                    output_path = (
                        VISUALIZATION_DIR
                        /
                        f"{image_id}.jpg"
                    )

                    cv2.imwrite(
                        str(
                            output_path
                        ),
                        image_bgr,
                    )

                    visualization_count += 1

                evaluated_images += 1

            progress.set_postfix(
                images=evaluated_images
            )

            if (
                MAX_EVAL_IMAGES is not None
                and
                evaluated_images
                >=
                MAX_EVAL_IMAGES
            ):

                break

    print(
        "\n[EVAL] Inference complete."
    )

    print(
        f"[EVAL] Images evaluated: "
        f"{len(predictions):,}"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_csv = (
        OUTPUT_DIR
        /
        "predictions.csv"
    )

    with open(
        predictions_csv,
        "w",
        newline="",
    ) as f:

        fieldnames = [
            "image_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "class",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            prediction_rows
        )

    print(
        f"[SAVE] Predictions: "
        f"{predictions_csv}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    print(
        "\n[EVAL] Computing metrics..."
    )

    metrics = {}

    ap_values = []

    # New:
    # Store complete statistics for every IoU threshold.
    iou_statistics = {}

    total_tp = 0

    total_fp = 0

    total_fn = 0

    for threshold in IOU_THRESHOLDS:

        (
            ap,
            tp,
            fp,
            fn,
        ) = collect_ap_statistics(
            predictions,
            ground_truths,
            float(threshold),
        )

        ap_values.append(
            ap
        )

        threshold_key = (
            f"{threshold:.2f}"
        )

        ap_key = (
            f"AP@{threshold:.2f}"
        )

        # TPR = TP / (TP + FN)
        tpr = (
            tp
            /
            max(
                tp + fn,
                1,
            )
        )

        # Keep the original AP keys.
        metrics[
            ap_key
        ] = float(
            ap
        )

        # Add TP, FP, FN and TPR for every IoU.
        metrics[
            f"TP@{threshold:.2f}"
        ] = int(
            tp
        )

        metrics[
            f"FP@{threshold:.2f}"
        ] = int(
            fp
        )

        metrics[
            f"FN@{threshold:.2f}"
        ] = int(
            fn
        )

        metrics[
            f"TPR@{threshold:.2f}"
        ] = float(
            tpr
        )

        # Also store a structured version.
        iou_statistics[
            threshold_key
        ] = {
            "AP": float(ap),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TPR": float(tpr),
        }

        # The global precision/recall/F1 remain
        # defined at IoU = 0.50.
        if np.isclose(
            threshold,
            0.50,
        ):

            total_tp = tp

            total_fp = fp

            total_fn = fn

    metrics[
        "iou_statistics"
    ] = iou_statistics

    map50 = metrics[
        "AP@0.50"
    ]

    # Standard mAP@0.50:0.95:
    # mean AP over IoU thresholds 0.50, 0.55, ..., 0.95.
    standard_map_thresholds = [
        threshold
        for threshold in IOU_THRESHOLDS
        if threshold >= 0.50
    ]

    standard_map_values = [
        metrics[
            f"AP@{threshold:.2f}"
        ]
        for threshold in standard_map_thresholds
    ]

    map50_95 = float(
        np.mean(
            standard_map_values
        )
    )

    precision = (
        total_tp
        /
        max(
            total_tp
            +
            total_fp,
            1,
        )
    )

    recall = (
        total_tp
        /
        max(
            total_tp
            +
            total_fn,
            1,
        )
    )

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        max(
            precision
            +
            recall,
            1e-12,
        )
    )

    metrics[
        "mAP50"
    ] = float(
        map50
    )

    metrics[
        "mAP50-95"
    ] = float(
        map50_95
    )

    metrics[
        "precision"
    ] = float(
        precision
    )

    metrics[
        "recall"
    ] = float(
        recall
    )

    metrics[
        "f1"
    ] = float(
        f1
    )

    metrics[
        "TP"
    ] = int(
        total_tp
    )

    metrics[
        "FP"
    ] = int(
        total_fp
    )

    metrics[
        "FN"
    ] = int(
        total_fn
    )

    metrics[
        "num_images"
    ] = int(
        len(predictions)
    )

    metrics[
        "num_ground_truth_boxes"
    ] = int(
        sum(
            len(boxes)
            for boxes
            in
            ground_truths.values()
        )
    )

    metrics[
        "num_predictions"
    ] = int(
        sum(
            len(preds)
            for preds
            in
            predictions.values()
        )
    )

    # ========================================================
    # CURVES
    # ========================================================

    print(
        "[EVAL] Building PR/FROC curves..."
    )

    curves = compute_pr_and_froc_curves(
        predictions,
        ground_truths,
        CURVE_IOU,
    )

    metrics[
        "curve_iou"
    ] = float(
        CURVE_IOU
    )

    metrics[
        "pr_ap"
    ] = float(
        curves["ap"]
    )

    metrics[
        "froc_sensitivity_at_fpi"
    ] = {
        key: float(value)
        for key, value in curves[
            "froc_points"
        ].items()
    }

    pr_curve_png = (
        OUTPUT_DIR
        / "pr_curve.png"
    )

    pr_curve_csv = (
        OUTPUT_DIR
        / "pr_curve.csv"
    )

    froc_curve_png = (
        OUTPUT_DIR
        / "froc_curve.png"
    )

    froc_curve_csv = (
        OUTPUT_DIR
        / "froc_curve.csv"
    )

    tpr_iou_curve_png = (
        OUTPUT_DIR
        / "tpr_vs_iou_curve.png"
    )

    tpr_iou_curve_csv = (
        OUTPUT_DIR
        / "tpr_vs_iou_curve.csv"
    )

    save_pr_curve_plot(
        pr_curve_png,
        curves["recall"],
        curves["precision"],
        curves["ap"],
        CURVE_IOU,
    )

    save_froc_curve_plot(
        froc_curve_png,
        curves["fpi"],
        curves["sensitivity"],
        curves[
            "froc_points"
        ],
        CURVE_IOU,
    )

    tpr_vs_iou_x = np.asarray(
        IOU_THRESHOLDS,
        dtype=np.float64,
    )

    tpr_vs_iou_y = np.asarray(
        [
            iou_statistics[
                f"{threshold:.2f}"
            ]["TPR"]
            for threshold in IOU_THRESHOLDS
        ],
        dtype=np.float64,
    )

    save_tpr_iou_curve_plot(
        tpr_iou_curve_png,
        tpr_vs_iou_x,
        tpr_vs_iou_y,
    )

    save_curve_csv(
        pr_curve_csv,
        [
            "recall",
            "precision",
        ],
        zip(
            curves["recall"],
            curves["precision"],
        ),
    )

    save_curve_csv(
        froc_curve_csv,
        [
            "fpi",
            "sensitivity",
        ],
        zip(
            curves["fpi"],
            curves[
                "sensitivity"
            ],
        ),
    )

    save_curve_csv(
        tpr_iou_curve_csv,
        [
            "iou_threshold",
            "tpr",
        ],
        zip(
            tpr_vs_iou_x,
            tpr_vs_iou_y,
        ),
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Images evaluated: "
        f"{metrics['num_images']:,}"
    )

    print(
        f"GT boxes: "
        f"{metrics['num_ground_truth_boxes']:,}"
    )

    print(
        f"Predictions: "
        f"{metrics['num_predictions']:,}"
    )

    print()

    print(
        f"mAP@0.50:      "
        f"{map50:.6f}"
    )

    print(
        f"mAP@0.50:0.95: "
        f"{map50_95:.6f}"
    )

    print(
        f"Precision:     "
        f"{precision:.6f}"
    )

    print(
        f"Recall:        "
        f"{recall:.6f}"
    )

    print(
        f"F1:            "
        f"{f1:.6f}"
    )

    print()

    print(
        f"TP: {total_tp}"
    )

    print(
        f"FP: {total_fp}"
    )

    print(
        f"FN: {total_fn}"
    )

    print()

    # ========================================================
    # AP / TP / FP / FN / TPR BY IOU
    # ========================================================

    print(
        "METRICS BY IOU THRESHOLD"
    )

    print(
        "-" * 75
    )

    print(
        f"{'IoU':>6} "
        f"{'AP':>12} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'TPR':>12}"
    )

    print(
        "-" * 75
    )

    for threshold in IOU_THRESHOLDS:

        threshold_key = (
            f"{threshold:.2f}"
        )

        stats = (
            iou_statistics[
                threshold_key
            ]
        )

        print(
            f"{float(threshold):6.2f} "
            f"{stats['AP']:12.6f} "
            f"{stats['TP']:8d} "
            f"{stats['FP']:8d} "
            f"{stats['FN']:8d} "
            f"{stats['TPR']:12.6f}"
        )

    print(
        "-" * 75
    )

    print(
        "=" * 70
    )

    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    metrics_json = (
        OUTPUT_DIR
        /
        "metrics.json"
    )

    with open(
        metrics_json,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(
        f"\n[SAVE] Metrics: "
        f"{metrics_json}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        /
        "summary.txt"
    )

    with open(
        summary_path,
        "w",
    ) as f:

        f.write(
            "SWIN-TINY + YOLOv8 TEST EVALUATION\n"
        )

        f.write(
            "=" * 60
            +
            "\n\n"
        )

        f.write(
            f"Checkpoint: "
            f"{CHECKPOINT}\n"
        )

        f.write(
            f"Test images: "
            f"{len(predictions):,}\n"
        )

        f.write(
            f"GT boxes: "
            f"{metrics['num_ground_truth_boxes']:,}\n"
        )

        f.write(
            f"Predictions: "
            f"{metrics['num_predictions']:,}\n\n"
        )

        f.write(
            f"mAP@0.50: "
            f"{map50:.6f}\n"
        )

        f.write(
            f"mAP@0.50:0.95: "
            f"{map50_95:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{precision:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{recall:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{f1:.6f}\n\n"
        )

        f.write(
            f"TP: "
            f"{total_tp}\n"
        )

        f.write(
            f"FP: "
            f"{total_fp}\n"
        )

        f.write(
            f"FN: "
            f"{total_fn}\n\n"
        )

        # ====================================================
        # COMPLETE IOU TABLE
        # ====================================================

        f.write(
            "METRICS BY IOU THRESHOLD\n"
        )

        f.write(
            "-" * 75
            +
            "\n"
        )

        f.write(
            f"{'IoU':>6} "
            f"{'AP':>12} "
            f"{'TP':>8} "
            f"{'FP':>8} "
            f"{'FN':>8} "
            f"{'TPR':>12}\n"
        )

        f.write(
            "-" * 75
            +
            "\n"
        )

        for threshold in IOU_THRESHOLDS:

            threshold_key = (
                f"{threshold:.2f}"
            )

            stats = (
                iou_statistics[
                    threshold_key
                ]
            )

            f.write(
                f"{float(threshold):6.2f} "
                f"{stats['AP']:12.6f} "
                f"{stats['TP']:8d} "
                f"{stats['FP']:8d} "
                f"{stats['FN']:8d} "
                f"{stats['TPR']:12.6f}\n"
            )

    print(
        f"[SAVE] Summary: "
        f"{summary_path}"
    )

    print(
        f"[SAVE] Visualizations: "
        f"{VISUALIZATION_DIR}"
    )

    print(
        f"[SAVE] Visualizations generated: "
        f"{visualization_count}"
    )

    print(
        f"[SAVE] PR curve: "
        f"{pr_curve_png}"
    )

    print(
        f"[SAVE] FROC curve: "
        f"{froc_curve_png}"
    )

    print(
        f"[SAVE] TPR vs IoU curve: "
        f"{tpr_iou_curve_png}"
    )

    print(
        "\n[DONE]"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()