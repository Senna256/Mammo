#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics.utils.nms import non_max_suppression


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_SCRIPT = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/Positives train/train_positives_4k.py"
)

CHECKPOINT = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/train_v2_4k/best.pt"
)

OUTPUT_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/evaluation/positives/outputs"
)

IMG_SIZE = 1024

CONF_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300

IOU_THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def find_training_script():
    if TRAINING_SCRIPT.exists():
        return TRAINING_SCRIPT

    root = Path(
        "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin"
    )

    candidates = sorted(
        root.rglob("*.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for path in candidates:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue

        if "class SwinYOLO" in text and "class VindrSwinDataset" in text:
            return path

    raise FileNotFoundError(
        "Could not find the Swin training script containing "
        "SwinYOLO and VindrSwinDataset."
    )


def load_training_module():
    path = find_training_script()

    print(f"[INIT] Training script: {path}")

    spec = importlib.util.spec_from_file_location(
        "swin_training_module",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import training script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["swin_training_module"] = module
    spec.loader.exec_module(module)

    return module


def extract_prediction_tensor(output):
    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            return output

    if isinstance(output, (list, tuple)):
        for item in output:
            try:
                tensor = extract_prediction_tensor(item)
            except RuntimeError:
                continue

            if tensor is not None:
                return tensor

    if isinstance(output, dict):
        for item in output.values():
            try:
                tensor = extract_prediction_tensor(item)
            except RuntimeError:
                continue

            if tensor is not None:
                return tensor

    raise RuntimeError(
        "Could not find the YOLO prediction tensor in model output."
    )


def xywhn_to_xyxy(boxes, img_size):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    )

    cx = boxes[:, 0] * img_size
    cy = boxes[:, 1] * img_size
    w = boxes[:, 2] * img_size
    h = boxes[:, 3] * img_size

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return np.stack(
        [x1, y1, x2, y2],
        axis=1,
    )


def calculate_iou(box1, box2):
    x1 = max(
        float(box1[0]),
        float(box2[0]),
    )
    y1 = max(
        float(box1[1]),
        float(box2[1]),
    )

    x2 = min(
        float(box1[2]),
        float(box2[2]),
    )
    y2 = min(
        float(box1[3]),
        float(box2[3]),
    )

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area1 = (
        max(0.0, float(box1[2]) - float(box1[0]))
        * max(0.0, float(box1[3]) - float(box1[1]))
    )

    area2 = (
        max(0.0, float(box2[2]) - float(box2[0]))
        * max(0.0, float(box2[3]) - float(box2[1]))
    )

    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def match_predictions(
    predictions,
    ground_truths,
    iou_threshold,
):
    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for prediction in predictions:
        best_iou = 0.0
        best_gt = None

        for gt_index, gt_box in enumerate(ground_truths):
            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index

        if (
            best_gt is not None
            and best_iou >= iou_threshold
        ):
            matched_gt.add(best_gt)
            tp += 1
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn


def compute_ap(
    image_predictions,
    image_ground_truths,
    iou_threshold,
):
    detections = []

    total_gt = 0

    for image_id in image_ground_truths:
        total_gt += len(
            image_ground_truths[image_id]
        )

        for prediction in image_predictions.get(
            image_id,
            [],
        ):
            detections.append(
                (
                    prediction["confidence"],
                    image_id,
                    prediction["box"],
                )
            )

    if total_gt == 0:
        return 0.0

    detections.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched = {
        image_id: set()
        for image_id in image_ground_truths
    }

    tp = np.zeros(
        len(detections),
        dtype=np.float64,
    )

    fp = np.zeros(
        len(detections),
        dtype=np.float64,
    )

    for index, (
        confidence,
        image_id,
        prediction_box,
    ) in enumerate(detections):

        gt_boxes = image_ground_truths[image_id]

        best_iou = 0.0
        best_gt = None

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched[image_id]:
                continue

            iou = calculate_iou(
                prediction_box,
                gt_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index

        if (
            best_gt is not None
            and best_iou >= iou_threshold
        ):
            matched[image_id].add(best_gt)
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recall = cumulative_tp / max(
        total_gt,
        1,
    )

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12,
    )

    recall_points = np.concatenate(
        [
            [0.0],
            recall,
            [1.0],
        ]
    )

    precision_points = np.concatenate(
        [
            [1.0],
            precision,
            [0.0],
        ]
    )

    for i in range(
        len(precision_points) - 2,
        -1,
        -1,
    ):
        precision_points[i] = max(
            precision_points[i],
            precision_points[i + 1],
        )

    ap = 0.0

    for i in range(
        1,
        len(recall_points),
    ):
        ap += (
            recall_points[i]
            - recall_points[i - 1]
        ) * precision_points[i]

    return float(ap)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SWIN + YOLO POSITIVE-ONLY TEST EVALUATION")
    print("=" * 70)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    training = load_training_module()

    print()
    print(f"[INIT] Device: {DEVICE}")
    print(f"[INIT] Checkpoint: {CHECKPOINT}")

    print()
    print("[INIT] Loading test dataset...")

    dataset = training.VindrSwinDataset(
        training.NETWORK_IMAGES,
        training.ANNOTATIONS_CSV,
        training.LOCAL_DATASET,
        "test",
        training.IMG_SIZE,
    )

    print(
        f"[INIT] Test images: {len(dataset):,}"
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=training.collate_fn,
    )

    print()
    print("[INIT] Building model...")

    model = training.SwinYOLO(
        img_size=training.IMG_SIZE,
        num_classes=training.NUM_CLASSES,
        pretrained=False,
    )

    model = model.to(DEVICE)

    model.detect.stride = torch.tensor(
        training.DETECT_STRIDES,
        dtype=torch.float32,
        device=DEVICE,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if "model" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain a 'model' state_dict."
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    print(
        f"[INIT] Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print()
    print("[EVAL] Running inference...")

    image_predictions = {}
    image_ground_truths = {}
    prediction_rows = []

    with torch.no_grad():
        for index, (
            images,
            targets,
            paths,
        ) in enumerate(loader, start=1):

            image_id = Path(
                paths[0]
            ).stem

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            output = model(images)

            prediction_tensor = extract_prediction_tensor(
                output
            )

            detections = non_max_suppression(
                prediction_tensor,
                conf_thres=CONF_THRESHOLD,
                iou_thres=NMS_IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
            )[0]

            if detections is None:
                detections = torch.empty(
                    (0, 6),
                    device=DEVICE,
                )

            detections = detections.detach().cpu().numpy()

            gt_boxes = xywhn_to_xyxy(
                targets["bboxes"].detach().cpu().numpy(),
                training.IMG_SIZE,
            )

            image_ground_truths[image_id] = [
                box
                for box in gt_boxes
            ]

            image_predictions[image_id] = []

            for detection in detections:
                x1, y1, x2, y2, confidence, cls = detection

                prediction = {
                    "box": np.array(
                        [x1, y1, x2, y2],
                        dtype=np.float32,
                    ),
                    "confidence": float(confidence),
                    "class": int(cls),
                }

                image_predictions[image_id].append(
                    prediction
                )

                prediction_rows.append(
                    {
                        "image_id": image_id,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(confidence),
                        "class": int(cls),
                    }
                )

            if index == 1 or index % 100 == 0:
                print(
                    f"[EVAL] {index:,}/{len(dataset):,}"
                )

    print()
    print("[EVAL] Computing metrics...")

    metrics = {}

    tp_by_iou = {}
    fp_by_iou = {}
    fn_by_iou = {}
    tpr_by_iou = {}
    precision_by_iou = {}
    f1_by_iou = {}
    ap_by_iou = {}

    for iou_threshold in IOU_THRESHOLDS:
        tp = 0
        fp = 0
        fn = 0

        for image_id in image_ground_truths:
            image_tp, image_fp, image_fn = match_predictions(
                image_predictions.get(
                    image_id,
                    [],
                ),
                image_ground_truths[image_id],
                iou_threshold,
            )

            tp += image_tp
            fp += image_fp
            fn += image_fn

        tpr = tp / max(
            tp + fn,
            1,
        )

        precision = tp / max(
            tp + fp,
            1,
        )

        f1 = (
            2.0 * precision * tpr
            / max(
                precision + tpr,
                1e-12,
            )
        )

        ap = compute_ap(
            image_predictions,
            image_ground_truths,
            iou_threshold,
        )

        key = f"{iou_threshold:.2f}"

        tp_by_iou[key] = tp
        fp_by_iou[key] = fp
        fn_by_iou[key] = fn
        tpr_by_iou[key] = tpr
        precision_by_iou[key] = precision
        f1_by_iou[key] = f1
        ap_by_iou[key] = ap

        metrics[f"TP@{key}"] = tp
        metrics[f"FP@{key}"] = fp
        metrics[f"FN@{key}"] = fn
        metrics[f"TPR@{key}"] = tpr
        metrics[f"Precision@{key}"] = precision
        metrics[f"F1@{key}"] = f1
        metrics[f"AP@{key}"] = ap

    metrics["test_images"] = len(
        image_ground_truths
    )

    metrics["ground_truth_boxes"] = sum(
        len(boxes)
        for boxes in image_ground_truths.values()
    )

    metrics["predictions_after_nms"] = sum(
        len(predictions)
        for predictions in image_predictions.values()
    )

    metrics["confidence_threshold"] = CONF_THRESHOLD
    metrics["nms_iou"] = NMS_IOU_THRESHOLD

    metrics["TPR_by_IoU"] = tpr_by_iou
    metrics["AP_by_IoU"] = ap_by_iou

    metrics["TP_by_IoU"] = tp_by_iou
    metrics["FP_by_IoU"] = fp_by_iou
    metrics["FN_by_IoU"] = fn_by_iou

    metrics["iou_thresholds"] = [
        float(x)
        for x in IOU_THRESHOLDS
    ]

    metrics["mAP@0.50"] = ap_by_iou["0.50"]

    metrics["mAP@0.50:0.95"] = float(
        np.mean([
            ap_by_iou[f"{x:.2f}"]
            for x in IOU_THRESHOLDS
            if x >= 0.50
        ])
    )

    metrics["TP@0.50"] = tp_by_iou["0.50"]
    metrics["FP@0.50"] = fp_by_iou["0.50"]
    metrics["FN@0.50"] = fn_by_iou["0.50"]
    metrics["TPR@0.50"] = tpr_by_iou["0.50"]
    metrics["Precision@0.50"] = precision_by_iou["0.50"]
    metrics["F1@0.50"] = f1_by_iou["0.50"]

    evaluation = {
        "checkpoint": str(CHECKPOINT),
        "metrics": metrics,
    }

    json_path = (
        OUTPUT_DIR
        / "test_evaluation.json"
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump(
            evaluation,
            f,
            indent=4,
        )

    predictions_path = (
        OUTPUT_DIR
        / "predictions.csv"
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    thresholds = [
        float(x)
        for x in IOU_THRESHOLDS
    ]

    tpr_values = [
        tpr_by_iou[f"{x:.2f}"]
        for x in thresholds
    ]

    ap_values = [
        ap_by_iou[f"{x:.2f}"]
        for x in thresholds
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, tpr_values, marker="o")
    plt.xlabel("IoU threshold")
    plt.ylabel("TPR / Recall")
    plt.title("Swin + YOLO Positive-Only: TPR vs IoU")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    tpr_plot = OUTPUT_DIR / "tpr_vs_iou.png"
    plt.savefig(tpr_plot, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, ap_values, marker="o")
    plt.xlabel("IoU threshold")
    plt.ylabel("AP")
    plt.title("Swin + YOLO Positive-Only: AP vs IoU")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ap_plot = OUTPUT_DIR / "ap_vs_iou.png"
    plt.savefig(ap_plot, dpi=200)
    plt.close()

    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Swin + YOLO Positive-Only Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"Test images: {metrics['test_images']}\n")
        f.write(f"Ground truth boxes: {metrics['ground_truth_boxes']}\n")
        f.write(f"Predictions after NMS: {metrics['predictions_after_nms']}\n")
        f.write(f"mAP@0.50: {metrics['mAP@0.50']:.6f}\n")
        f.write(f"mAP@0.50:0.95: {metrics['mAP@0.50:0.95']:.6f}\n")
        f.write(f"Precision@0.50: {metrics['Precision@0.50']:.6f}\n")
        f.write(f"TPR@0.50: {metrics['TPR@0.50']:.6f}\n")
        f.write(f"F1@0.50: {metrics['F1@0.50']:.6f}\n")
        f.write(f"TP@0.50: {metrics['TP@0.50']}\n")
        f.write(f"FP@0.50: {metrics['FP@0.50']}\n")
        f.write(f"FN@0.50: {metrics['FN@0.50']}\n\n")
        f.write("IoU metrics\n")
        f.write("-" * 60 + "\n")
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            f.write(
                f"IoU {key}: "
                f"TPR={tpr_by_iou[key]:.6f} | "
                f"Precision={precision_by_iou[key]:.6f} | "
                f"AP={ap_by_iou[key]:.6f} | "
                f"TP={tp_by_iou[key]} | "
                f"FP={fp_by_iou[key]} | "
                f"FN={fn_by_iou[key]}\n"
            )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for iou_threshold in IOU_THRESHOLDS:
        key = f"{iou_threshold:.2f}"

        print(
            f"IoU {key}: "
            f"TPR={tpr_by_iou[key]:.4f} | "
            f"Precision={precision_by_iou[key]:.4f} | "
            f"AP={ap_by_iou[key]:.4f}"
        )

    print()
    print(f"[SAVE] {json_path}")
    print(f"[SAVE] {predictions_path}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {tpr_plot}")
    print(f"[SAVE] {ap_plot}")


if __name__ == "__main__":
    main()
