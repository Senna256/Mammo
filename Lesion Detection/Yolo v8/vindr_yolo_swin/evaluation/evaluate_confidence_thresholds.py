#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils.nms import non_max_suppression

from evaluate_swin_yolo import (
    NETWORK_IMAGES,
    ANNOTATIONS_CSV,
    LOCAL_DATASET,
    CHECKPOINT,
    OUTPUT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    NMS_IOU_THRESHOLD,
    MAX_DETECTIONS,
    DEVICE,
    VindrSwinDataset,
    SwinYOLO,
    collate_fn,
    decode_detect_output,
    load_checkpoint,
    get_ground_truth_for_image,
    box_iou,
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLDS = [
    0.001,
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

MATCH_IOU_THRESHOLD = 0.50

OUTPUT_FILE = (
    OUTPUT_DIR
    / "confidence_threshold_analysis.csv"
)


# ============================================================
# MATCHING
# ============================================================

def match_predictions_at_threshold(
    predictions,
    ground_truths,
    iou_threshold,
):
    """
    Match predictions against GT boxes.

    Predictions must already be filtered by
    confidence threshold.

    Matching is performed greedily in descending
    confidence order.

    Returns:
        TP
        FP
        FN
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_id in ground_truths:

        gts = ground_truths[image_id]

        preds = predictions.get(
            image_id,
            np.zeros(
                (0, 6),
                dtype=np.float32,
            ),
        )

        # ----------------------------------------------------
        # No GT
        # ----------------------------------------------------

        if len(gts) == 0:

            total_fp += len(preds)

            continue

        # ----------------------------------------------------
        # No predictions
        # ----------------------------------------------------

        if len(preds) == 0:

            total_fn += len(gts)

            continue

        # ----------------------------------------------------
        # Sort predictions by confidence
        # ----------------------------------------------------

        preds = sorted(
            preds,
            key=lambda x: float(x[4]),
            reverse=True,
        )

        matched_gt = set()

        for prediction in preds:

            pred_box = prediction[:4]

            ious = box_iou(
                np.asarray(
                    [pred_box],
                    dtype=np.float32,
                ),
                gts,
            )[0]

            if len(ious) == 0:

                total_fp += 1

                continue

            # Highest IoU GT first
            gt_order = np.argsort(
                -ious
            )

            found_match = False

            for gt_index in gt_order:

                gt_index = int(
                    gt_index
                )

                if gt_index in matched_gt:
                    continue

                if (
                    float(ious[gt_index])
                    >= iou_threshold
                ):

                    matched_gt.add(
                        gt_index
                    )

                    total_tp += 1

                    found_match = True

                    break

            if not found_match:

                total_fp += 1

        # ----------------------------------------------------
        # Unmatched GT = FN
        # ----------------------------------------------------

        total_fn += (
            len(gts)
            -
            len(matched_gt)
        )

    return (
        total_tp,
        total_fp,
        total_fn,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    ground_truths,
    confidence_threshold,
    iou_threshold,
):
    """
    Calculate detection metrics for a given
    confidence threshold.
    """

    filtered_predictions = {}

    total_predictions = 0

    for image_id, preds in predictions.items():

        filtered = []

        for prediction in preds:

            if len(prediction) < 5:
                continue

            confidence = float(
                prediction[4]
            )

            if (
                confidence
                >=
                confidence_threshold
            ):

                filtered.append(
                    prediction
                )

        filtered_predictions[
            image_id
        ] = np.asarray(
            filtered,
            dtype=np.float32,
        ).reshape(
            -1,
            6,
        )

        total_predictions += len(
            filtered
        )

    (
        tp,
        fp,
        fn,
    ) = match_predictions_at_threshold(
        filtered_predictions,
        ground_truths,
        iou_threshold,
    )

    precision = (
        tp
        /
        max(
            tp + fp,
            1,
        )
    )

    recall = (
        tp
        /
        max(
            tp + fn,
            1,
        )
    )

    if (
        precision + recall
        >
        0
    ):

        f1 = (
            2.0
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0.0

    return {
        "confidence_threshold": (
            confidence_threshold
        ),
        "predictions": (
            total_predictions
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "SWIN-TINY + YOLOv8"
    )
    print(
        "CONFIDENCE THRESHOLD ANALYSIS"
    )
    print("=" * 75)

    print()
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
        f"Checkpoint:"
    )
    print(
        CHECKPOINT
    )

    print()
    print(
        f"Matching IoU threshold: "
        f"{MATCH_IOU_THRESHOLD:.2f}"
    )

    print()
    print(
        "Confidence thresholds:"
    )

    for threshold in CONFIDENCE_THRESHOLDS:

        print(
            f"  {threshold:.3f}"
        )

    # ========================================================
    # DATASET
    # ========================================================

    print()
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

    print()
    print(
        "[INIT] Building model..."
    )

    model = SwinYOLO(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        pretrained=False,
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

    print()
    print(
        "[EVAL] Running inference..."
    )

    predictions = {}

    ground_truths = {}

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="TEST",
            unit="batch",
        )

        for (
            images,
            targets,
            paths,
        ) in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    DEVICE.type
                    == "cuda"
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
                    conf_thres=0.001,
                    iou_thres=(
                        NMS_IOU_THRESHOLD
                    ),
                    classes=None,
                    agnostic=False,
                    max_det=(
                        MAX_DETECTIONS
                    ),
                )
            )

            for (
                batch_index,
                path,
            ) in enumerate(paths):

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

    print()
    print(
        "[EVAL] Inference complete."
    )

    print(
        f"[EVAL] Images evaluated: "
        f"{len(predictions):,}"
    )

    total_gt = sum(
        len(boxes)
        for boxes
        in ground_truths.values()
    )

    total_predictions = sum(
        len(preds)
        for preds
        in predictions.values()
    )

    print(
        f"[EVAL] GT boxes: "
        f"{total_gt:,}"
    )

    print(
        f"[EVAL] Raw predictions: "
        f"{total_predictions:,}"
    )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    print()
    print(
        "[ANALYSIS] Computing metrics..."
    )

    results = []

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        metrics = calculate_metrics(
            predictions,
            ground_truths,
            confidence_threshold,
            MATCH_IOU_THRESHOLD,
        )

        results.append(
            metrics
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 95)
    print(
        "CONFIDENCE THRESHOLD RESULTS"
    )
    print("=" * 95)

    print(
        f"{'Conf':>8} "
        f"{'Pred':>8} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )

    print("-" * 95)

    for result in results:

        print(
            f"{result['confidence_threshold']:>8.3f} "
            f"{result['predictions']:>8d} "
            f"{result['TP']:>8d} "
            f"{result['FP']:>8d} "
            f"{result['FN']:>8d} "
            f"{result['precision']:>12.6f} "
            f"{result['recall']:>10.6f} "
            f"{result['f1']:>10.6f}"
        )

    print("=" * 95)

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"[SAVE] "
        f"{OUTPUT_FILE}"
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as f:

        fieldnames = [
            "confidence_threshold",
            "predictions",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "f1",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "ANALYSIS FINISHED"
    )
    print("=" * 75)

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The original model predictions were "
        "generated with confidence >= 0.001."
    )

    print(
        "Only the confidence threshold used "
        "for counting TP/FP/FN changes."
    )

    print(
        "The model and checkpoint are unchanged."
    )

    print()
    print("=" * 75)


if __name__ == "__main__":

    main()