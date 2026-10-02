#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils.nms import non_max_suppression


NETWORK_IMAGES = Path("/home/enric/Datasets/Original/vindr/images")
ANNOTATIONS_CSV = Path("/home/enric/Datasets/Original/vindr/finding_annotations.csv")
LOCAL_DATASET = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo")
VIT_ROOT = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit")
TRAINING_SCRIPT = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/train_positives/train_vit_positives.py")
CHECKPOINT = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/train_positives/outputs/best.pt")
OUTPUT_DIR = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/evaluation_positives/outputs")

IMG_SIZE = 1024
BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_CLASSES = 1

CONF_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300
MAX_EVAL_IMAGES = None

IOU_THRESHOLDS = np.round(np.arange(0.10, 0.96, 0.05), 2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_training_module():
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    path = TRAINING_SCRIPT
    print(f"[MODULE] {path}")

    spec = importlib.util.spec_from_file_location(
        "vit_positive_only_training",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def box_iou(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    area1 = (
        np.maximum(0, boxes1[:, 2] - boxes1[:, 0])
        * np.maximum(0, boxes1[:, 3] - boxes1[:, 1])
    )
    area2 = (
        np.maximum(0, boxes2[:, 2] - boxes2[:, 0])
        * np.maximum(0, boxes2[:, 3] - boxes2[:, 1])
    )

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / np.maximum(union, 1e-9)


def xywhn_to_xyxy(box):
    box = np.asarray(box, dtype=np.float32)

    if box.shape[0] == 5:
        cx, cy, w, h = box[1:5]
    elif box.shape[0] == 4:
        cx, cy, w, h = box
    else:
        raise ValueError(f"Unexpected box shape: {box.shape}")

    return np.array(
        [
            (cx - w / 2) * IMG_SIZE,
            (cy - h / 2) * IMG_SIZE,
            (cx + w / 2) * IMG_SIZE,
            (cy + h / 2) * IMG_SIZE,
        ],
        dtype=np.float32,
    )


def get_gt(targets, batch_index):
    batch_idx = targets["batch_idx"].detach().cpu().numpy()
    boxes = targets["bboxes"].detach().cpu().numpy()

    image_boxes = boxes[batch_idx == batch_index]

    if len(image_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.asarray(
        [xywhn_to_xyxy(box) for box in image_boxes],
        dtype=np.float32,
    ).reshape(-1, 4)


def find_prediction_tensor(output):
    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            return output
        raise RuntimeError(f"Unexpected prediction tensor shape: {output.shape}")

    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return find_prediction_tensor(item)
            except RuntimeError:
                pass

    if isinstance(output, dict):
        for item in output.values():
            try:
                return find_prediction_tensor(item)
            except RuntimeError:
                pass

    raise RuntimeError(
        f"Could not find YOLO prediction tensor in {type(output)}"
    )


def compute_ap(recall, precision):
    if len(recall) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]

    return float(
        np.sum(
            (mrec[indices + 1] - mrec[indices])
            * mpre[indices + 1]
        )
    )


def evaluate_at_iou(predictions, ground_truths, iou_threshold):
    records = []

    total_gt = sum(len(boxes) for boxes in ground_truths.values())

    for image_id, preds in predictions.items():
        for pred in preds:
            if len(pred) >= 5:
                records.append(
                    (
                        image_id,
                        float(pred[4]),
                        np.asarray(pred[:4], dtype=np.float32),
                    )
                )

    records.sort(key=lambda x: x[1], reverse=True)

    matched = {
        image_id: set()
        for image_id in ground_truths
    }

    tp_values = []
    fp_values = []

    for image_id, confidence, pred_box in records:
        gts = ground_truths.get(
            image_id,
            np.zeros((0, 4), dtype=np.float32),
        )

        if len(gts) == 0:
            tp_values.append(0)
            fp_values.append(1)
            continue

        ious = box_iou(pred_box[None, :], gts)[0]
        order = np.argsort(-ious)

        matched_gt = False

        for gt_index in order:
            gt_index = int(gt_index)

            if gt_index in matched[image_id]:
                continue

            if ious[gt_index] >= iou_threshold:
                matched[image_id].add(gt_index)
                matched_gt = True
                break

        if matched_gt:
            tp_values.append(1)
            fp_values.append(0)
        else:
            tp_values.append(0)
            fp_values.append(1)

    if not tp_values:
        return 0.0, 0, 0, total_gt

    tp_values = np.asarray(tp_values, dtype=np.float64)
    fp_values = np.asarray(fp_values, dtype=np.float64)

    cumulative_tp = np.cumsum(tp_values)
    cumulative_fp = np.cumsum(fp_values)

    recall = cumulative_tp / max(total_gt, 1)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12,
    )

    ap = compute_ap(recall, precision)

    tp = int(cumulative_tp[-1])
    fp = int(cumulative_fp[-1])
    fn = int(total_gt - tp)

    return ap, tp, fp, fn


def load_checkpoint(model):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if "model" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain 'model'.")

    missing, unexpected = model.load_state_dict(
        checkpoint["model"],
        strict=False,
    )

    if missing:
        print(f"[CHECKPOINT] Missing keys: {len(missing)}")

    if unexpected:
        print(f"[CHECKPOINT] Unexpected keys: {len(unexpected)}")

    print(f"[CHECKPOINT] Epoch: {checkpoint.get('epoch', 'N/A')}")

    if "best_val_loss" in checkpoint:
        print(
            f"[CHECKPOINT] Best val loss: "
            f"{checkpoint['best_val_loss']}"
        )


def save_plot(x, y, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("IoU threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_summary(metrics):
    path = OUTPUT_DIR / "summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("ViT-Base + YOLOv8 POSITIVE-ONLY TEST EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"Test images: {metrics['test_images']:,}\n")
        f.write(
            f"Ground truth boxes: "
            f"{metrics['ground_truth_boxes']:,}\n"
        )
        f.write(
            f"Predictions after NMS: "
            f"{metrics['predictions_after_nms']:,}\n\n"
        )

        f.write(
            f"mAP@0.50: {metrics['mAP@0.50']:.6f}\n"
        )
        f.write(
            f"mAP@0.50:0.95: "
            f"{metrics['mAP@0.50:0.95']:.6f}\n"
        )
        f.write(
            f"Precision@0.50: "
            f"{metrics['precision@0.50']:.6f}\n"
        )
        f.write(
            f"Recall / TPR@0.50: "
            f"{metrics['recall@0.50']:.6f}\n"
        )
        f.write(
            f"F1@0.50: "
            f"{metrics['f1@0.50']:.6f}\n\n"
        )

        f.write(
            f"TP@0.50: {metrics['TP@0.50']}\n"
        )
        f.write(
            f"FP@0.50: {metrics['FP@0.50']}\n"
        )
        f.write(
            f"FN@0.50: {metrics['FN@0.50']}\n\n"
        )

        f.write("METRICS BY IOU THRESHOLD\n")
        f.write("-" * 70 + "\n")
        f.write("IoU     AP        TP      FP      FN      TPR\n")
        f.write("-" * 70 + "\n")

        for i, threshold in enumerate(metrics["iou_thresholds"]):
            f.write(
                f"{threshold:.2f}    "
                f"{metrics['ap_by_iou'][i]:.6f}    "
                f"{metrics['tp_by_iou'][i]:6d}  "
                f"{metrics['fp_by_iou'][i]:6d}  "
                f"{metrics['fn_by_iou'][i]:6d}  "
                f"{metrics['tpr_by_iou'][i]:.6f}\n"
            )

    return path


def main():
    print("=" * 80)
    print("ViT-Base + YOLOv8 POSITIVE-ONLY TEST EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    module = import_training_module()

    VindrViTDataset = module.VindrViTDataset
    ViTYOLO = module.ViTYOLO
    collate_fn = module.collate_fn

    print("\n[INIT] Loading test dataset...")

    dataset = VindrViTDataset(
        NETWORK_IMAGES,
        ANNOTATIONS_CSV,
        LOCAL_DATASET,
        "test",
        IMG_SIZE,
    )

    total_images = len(dataset)

    eval_count = (
        total_images
        if MAX_EVAL_IMAGES is None
        else min(MAX_EVAL_IMAGES, total_images)
    )

    print(f"[INIT] Test images: {total_images:,}")
    print(f"[INIT] Images to evaluate: {eval_count:,}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("\n[INIT] Building model...")

    model = ViTYOLO(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    load_checkpoint(model)

    model = model.to(DEVICE)

    model.detect.stride = torch.tensor(
        (8.0, 16.0, 32.0),
        dtype=torch.float32,
        device=DEVICE,
    )

    model.eval()

    predictions = {}
    ground_truths = {}
    prediction_rows = []

    evaluated = 0
    amp_enabled = DEVICE.type == "cuda"

    print("\n[EVAL] Running inference...")

    with torch.no_grad():
        progress = tqdm(
            loader,
            total=eval_count,
            desc="TEST",
            unit="image",
            dynamic_ncols=True,
        )

        for images, targets, paths in progress:
            if (
                MAX_EVAL_IMAGES is not None
                and evaluated >= MAX_EVAL_IMAGES
            ):
                break

            remaining = (
                None
                if MAX_EVAL_IMAGES is None
                else MAX_EVAL_IMAGES - evaluated
            )

            if remaining is not None and len(images) > remaining:
                images = images[:remaining]

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled,
            ):
                raw_output = model(images)
                prediction_tensor = find_prediction_tensor(raw_output)

            detections = non_max_suppression(
                prediction_tensor,
                conf_thres=CONF_THRESHOLD,
                iou_thres=NMS_IOU_THRESHOLD,
                classes=None,
                agnostic=False,
                max_det=MAX_DETECTIONS,
            )

            for batch_index, detection in enumerate(detections):
                image_id = Path(
                    paths[batch_index]
                ).stem

                pred = (
                    detection
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                gt = get_gt(
                    targets,
                    batch_index,
                )

                predictions[image_id] = pred
                ground_truths[image_id] = gt

                for row in pred:
                    if len(row) < 5:
                        continue

                    prediction_rows.append(
                        {
                            "image_id": image_id,
                            "x1": float(row[0]),
                            "y1": float(row[1]),
                            "x2": float(row[2]),
                            "y2": float(row[3]),
                            "confidence": float(row[4]),
                            "class": (
                                int(row[5])
                                if len(row) > 5
                                else 0
                            ),
                        }
                    )

                evaluated += 1

            progress.set_postfix(
                images=evaluated
            )

            if (
                MAX_EVAL_IMAGES is not None
                and evaluated >= MAX_EVAL_IMAGES
            ):
                break

    print("\n[EVAL] Inference complete.")

    predictions_csv = OUTPUT_DIR / "predictions.csv"

    with open(
        predictions_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "class",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"[SAVE] Predictions: {predictions_csv}")

    print("\n[EVAL] Computing metrics by IoU...")

    ap_by_iou = []
    tp_by_iou = []
    fp_by_iou = []
    fn_by_iou = []
    tpr_by_iou = []

    for threshold in IOU_THRESHOLDS:
        ap, tp, fp, fn = evaluate_at_iou(
            predictions,
            ground_truths,
            float(threshold),
        )

        tpr = tp / max(tp + fn, 1)

        ap_by_iou.append(float(ap))
        tp_by_iou.append(int(tp))
        fp_by_iou.append(int(fp))
        fn_by_iou.append(int(fn))
        tpr_by_iou.append(float(tpr))

    index_050 = int(
        np.where(
            np.isclose(
                IOU_THRESHOLDS,
                0.50,
            )
        )[0][0]
    )

    map50 = ap_by_iou[index_050]

    map50_95 = float(
        np.mean(
            [
                ap_by_iou[i]
                for i, threshold
                in enumerate(IOU_THRESHOLDS)
                if threshold >= 0.50
            ]
        )
    )

    tp50 = tp_by_iou[index_050]
    fp50 = fp_by_iou[index_050]
    fn50 = fn_by_iou[index_050]

    precision50 = tp50 / max(tp50 + fp50, 1)
    recall50 = tp50 / max(tp50 + fn50, 1)

    f150 = (
        2 * precision50 * recall50
        / max(precision50 + recall50, 1e-12)
    )

    metrics = {
        "checkpoint": str(CHECKPOINT),
        "test_images": len(predictions),
        "ground_truth_boxes": sum(
            len(x) for x in ground_truths.values()
        ),
        "predictions_after_nms": sum(
            len(x) for x in predictions.values()
        ),
        "image_size": IMG_SIZE,
        "confidence_threshold": CONF_THRESHOLD,
        "nms_iou": NMS_IOU_THRESHOLD,

        "iou_thresholds": [
            float(x) for x in IOU_THRESHOLDS
        ],

        "ap_by_iou": ap_by_iou,
        "tp_by_iou": tp_by_iou,
        "fp_by_iou": fp_by_iou,
        "fn_by_iou": fn_by_iou,
        "tpr_by_iou": tpr_by_iou,

        "mAP@0.50": map50,
        "mAP@0.50:0.95": map50_95,

        "precision@0.50": precision50,
        "recall@0.50": recall50,
        "tpr@0.50": recall50,
        "f1@0.50": f150,

        "TP@0.50": tp50,
        "FP@0.50": fp50,
        "FN@0.50": fn50,
    }

    metrics_json = OUTPUT_DIR / "metrics.json"

    with open(
        metrics_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    summary_path = save_summary(metrics)

    tpr_plot = save_plot(
        np.asarray(IOU_THRESHOLDS),
        np.asarray(tpr_by_iou),
        "TPR / Recall",
        "ViT + YOLO (positive-only training) — TPR vs IoU",
        "tpr_vs_iou.png",
    )

    ap_plot = save_plot(
        np.asarray(IOU_THRESHOLDS),
        np.asarray(ap_by_iou),
        "AP",
        "ViT + YOLO (positive-only training) — AP vs IoU",
        "ap_vs_iou.png",
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(f"Test images: {metrics['test_images']:,}")
    print(
        f"Ground truth boxes: "
        f"{metrics['ground_truth_boxes']:,}"
    )
    print(
        f"Predictions after NMS: "
        f"{metrics['predictions_after_nms']:,}"
    )
    print()

    print(f"mAP@0.50: {map50:.6f}")
    print(f"mAP@0.50:0.95: {map50_95:.6f}")
    print(f"Precision@0.50: {precision50:.6f}")
    print(f"Recall / TPR@0.50: {recall50:.6f}")
    print(f"F1@0.50: {f150:.6f}")

    print("\nIoU      AP        TP      FP      FN      TPR")
    print("-" * 65)

    for i, threshold in enumerate(IOU_THRESHOLDS):
        print(
            f"{threshold:.2f}    "
            f"{ap_by_iou[i]:.6f}    "
            f"{tp_by_iou[i]:6d}  "
            f"{fp_by_iou[i]:6d}  "
            f"{fn_by_iou[i]:6d}  "
            f"{tpr_by_iou[i]:.6f}"
        )

    print()
    print(f"[SAVE] {metrics_json}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {tpr_plot}")
    print(f"[SAVE] {ap_plot}")
    print(f"[SAVE] {predictions_csv}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
