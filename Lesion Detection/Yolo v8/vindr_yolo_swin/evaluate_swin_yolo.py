#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import cv2
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

VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

IMG_SIZE = 1024

BATCH_SIZE = 1

NUM_WORKERS = 4

NUM_CLASSES = 1

CONF_THRESHOLD = 0.001

NMS_IOU_THRESHOLD = 0.70

MAX_DETECTIONS = 300

VISUAL_CONF_THRESHOLD = 0.25

MAX_VISUALIZATIONS = 100

IOU_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05,
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

    if len(boxes1) == 0 or len(boxes2) == 0:

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
            boxes1[:, 2] - boxes1[:, 0],
        )
        *
        np.maximum(
            0,
            boxes1[:, 3] - boxes1[:, 1],
        )
    )

    area2 = (
        np.maximum(
            0,
            boxes2[:, 2] - boxes2[:, 0],
        )
        *
        np.maximum(
            0,
            boxes2[:, 3] - boxes2[:, 1],
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

    else:

        x_center = box[0]
        y_center = box[1]
        width = box[2]
        height = box[3]

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

            decoded = model.detect._inference(
                output
            )

            return decoded

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
        f"[CHECKPOINT] Loading: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if "model" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain 'model'."
        )

    state_dict = checkpoint["model"]

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

        x1, y1, x2, y2, conf, cls = (
            prediction[:6]
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
            f"{float(conf):.3f}",
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
# MATCHING
# ============================================================

def match_predictions(
    predictions,
    ground_truths,
    iou_threshold,
):

    if len(predictions) == 0:

        return (
            0,
            0,
            len(ground_truths),
        )

    if len(ground_truths) == 0:

        return (
            0,
            len(predictions),
            0,
        )

    predictions = sorted(
        predictions,
        key=lambda x: float(x[4]),
        reverse=True,
    )

    matched_gt = set()

    tp = 0

    fp = 0

    for prediction in predictions:

        pred_box = prediction[:4]

        ious = box_iou(
            np.asarray(
                [pred_box],
                dtype=np.float32,
            ),
            ground_truths,
        )[0]

        best_index = int(
            np.argmax(ious)
        )

        best_iou = float(
            ious[best_index]
        )

        if (
            best_iou >= iou_threshold
            and
            best_index not in matched_gt
        ):

            tp += 1

            matched_gt.add(
                best_index
            )

        else:

            fp += 1

    fn = (
        len(ground_truths)
        -
        len(matched_gt)
    )

    return (
        tp,
        fp,
        fn,
    )


# ============================================================
# DATASET GT EXTRACTION
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
# AP DATA
# ============================================================

def collect_ap_statistics(
    predictions,
    ground_truths,
    iou_threshold,
):

    prediction_records = []

    total_gt = 0

    for image_id, gts in ground_truths.items():

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

    for image_id, confidence, pred_box in (
        prediction_records
    ):

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

            if gt_index in matched[image_id]:

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

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    print(
        f"[INIT] Test images: "
        f"{len(test_dataset):,}"
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

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="TEST",
            unit="batch",
        )

        for batch_number, (
            images,
            targets,
            paths,
        ) in enumerate(progress):

            if batch_number >= 10:
                break

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

            if len(detections) > 0:

                n = len(detections[0])

                if n > 0:

                    max_conf = float(
                        detections[0][:, 4].max()
                    )

                    mean_conf = float(
                        detections[0][:, 4].mean()
                    )

                    print(
                        f"\n[DEBUG] detections={n} "
                        f"max_conf={max_conf:.6f} "
                        f"mean_conf={mean_conf:.6f}"
                    )

                else:

                    print(
                        "\n[DEBUG] NMS returned 0 detections"
                    )

            for batch_index, path in enumerate(paths):

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

                gt = get_ground_truth_for_image(
                    targets,
                    batch_index,
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
                ):

                    valid_predictions = [
                        detection
                        for detection in pred
                        if len(detection) >= 5
                        and
                        detection[4]
                        >= VISUAL_CONF_THRESHOLD
                    ]

                    if (
                        len(valid_predictions) > 0
                        or
                        len(gt) > 0
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
                            image_array.shape[2]
                            == 3
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

                        # GT = green

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

                        # PRED = red

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

        metrics[
            f"AP@{threshold:.2f}"
        ] = float(
            ap
        )

        if np.isclose(
            threshold,
            0.50,
        ):

            total_tp = tp

            total_fp = fp

            total_fn = fn

    map50 = metrics[
        "AP@0.50"
    ]

    map50_95 = float(
        np.mean(
            ap_values
        )
    )

    precision = (
        total_tp
        /
        max(
            total_tp + total_fp,
            1,
        )
    )

    recall = (
        total_tp
        /
        max(
            total_tp + total_fn,
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
            precision + recall,
            1e-12,
        )
    )

    metrics["mAP50"] = float(
        map50
    )

    metrics["mAP50-95"] = float(
        map50_95
    )

    metrics["precision"] = float(
        precision
    )

    metrics["recall"] = float(
        recall
    )

    metrics["f1"] = float(
        f1
    )

    metrics["TP"] = int(
        total_tp
    )

    metrics["FP"] = int(
        total_fp
    )

    metrics["FN"] = int(
        total_fn
    )

    metrics["num_images"] = int(
        len(predictions)
    )

    metrics["num_ground_truth_boxes"] = int(
        sum(
            len(boxes)
            for boxes in ground_truths.values()
        )
    )

    metrics["num_predictions"] = int(
        sum(
            len(preds)
            for preds in predictions.values()
        )
    )

    # ========================================================
    # PRINT METRICS
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
        f"mAP@0.50:     {map50:.6f}"
    )

    print(
        f"mAP@0.50:0.95: {map50_95:.6f}"
    )

    print(
        f"Precision:    {precision:.6f}"
    )

    print(
        f"Recall:       {recall:.6f}"
    )

    print(
        f"F1:           {f1:.6f}"
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

    print(
        f"GT boxes: "
        f"{metrics['num_ground_truth_boxes']:,}"
    )

    print(
        f"Predictions: "
        f"{metrics['num_predictions']:,}"
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
            f"Checkpoint: {CHECKPOINT}\n"
        )

        f.write(
            f"Test images: {len(predictions):,}\n"
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
            f"mAP@0.50: {map50:.6f}\n"
        )

        f.write(
            f"mAP@0.50:0.95: {map50_95:.6f}\n"
        )

        f.write(
            f"Precision: {precision:.6f}\n"
        )

        f.write(
            f"Recall: {recall:.6f}\n"
        )

        f.write(
            f"F1: {f1:.6f}\n\n"
        )

        f.write(
            f"TP: {total_tp}\n"
        )

        f.write(
            f"FP: {total_fp}\n"
        )

        f.write(
            f"FN: {total_fn}\n\n"
        )

        f.write(
            "AP BY IOU THRESHOLD\n"
        )

        f.write(
            "-" * 40
            +
            "\n"
        )

        for threshold in IOU_THRESHOLDS:

            key = (
                f"AP@{threshold:.2f}"
            )

            f.write(
                f"{key}: "
                f"{metrics[key]:.6f}\n"
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
        "\n[DONE]"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()