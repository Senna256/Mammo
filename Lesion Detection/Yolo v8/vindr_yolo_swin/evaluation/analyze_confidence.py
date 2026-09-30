#!/usr/bin/env python3

"""
Confidence / TP-FP-FN analysis for Swin + YOLO.

This script DOES NOT run model inference.

It uses:
    predictions.csv
    YOLO ground-truth labels

and analyzes:

    - TP vs FP confidence distribution
    - false negatives
    - confidence bins
    - high-confidence FP
    - low-confidence TP
    - confidence histograms
    - cumulative TP/FP curves

IoU threshold:
    0.50

Expected prediction CSV columns:

    image_id
    x1
    y1
    x2
    y2
    confidence
    class
"""

from pathlib import Path
import csv
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin"
)

EVALUATION_DIR = (
    BASE_DIR
    / "evaluation"
)

PREDICTIONS_CSV = (
    EVALUATION_DIR
    / "predictions.csv"
)

# YOLO dataset
YOLO_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

# We will automatically search these possible locations.
LABEL_DIR_CANDIDATES = [
    YOLO_DIR / "test" / "labels",
    YOLO_DIR / "labels" / "test",
    YOLO_DIR / "val" / "labels",
    YOLO_DIR / "labels" / "val",
]


OUTPUT_DIR = (
    EVALUATION_DIR
    / "confidence_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PARAMETERS
# ============================================================

IOU_THRESHOLD = 0.50

# Confidence ranges used for detailed analysis
CONFIDENCE_BINS = [
    0.00,
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
    1.00,
]

# Interesting examples
LOW_CONF_TP_THRESHOLD = 0.10
HIGH_CONF_TP_THRESHOLD = 0.50
HIGH_CONF_FP_THRESHOLD = 0.50


# ============================================================
# IOU
# ============================================================

def box_iou(
    box1,
    box2,
):
    """
    Calculate pairwise IoU.

    Boxes:
        [x1, y1, x2, y2]
    """

    if len(box1) == 0 or len(box2) == 0:
        return np.zeros(
            (
                len(box1),
                len(box2),
            ),
            dtype=np.float32,
        )

    box1 = np.asarray(
        box1,
        dtype=np.float32,
    )

    box2 = np.asarray(
        box2,
        dtype=np.float32,
    )

    area1 = (
        np.maximum(
            box1[:, 2] - box1[:, 0],
            0,
        )
        *
        np.maximum(
            box1[:, 3] - box1[:, 1],
            0,
        )
    )

    area2 = (
        np.maximum(
            box2[:, 2] - box2[:, 0],
            0,
        )
        *
        np.maximum(
            box2[:, 3] - box2[:, 1],
            0,
        )
    )

    inter_x1 = np.maximum(
        box1[:, None, 0],
        box2[None, :, 0],
    )

    inter_y1 = np.maximum(
        box1[:, None, 1],
        box2[None, :, 1],
    )

    inter_x2 = np.minimum(
        box1[:, None, 2],
        box2[None, :, 2],
    )

    inter_y2 = np.minimum(
        box1[:, None, 3],
        box2[None, :, 3],
    )

    inter_w = np.maximum(
        inter_x2 - inter_x1,
        0,
    )

    inter_h = np.maximum(
        inter_y2 - inter_y1,
        0,
    )

    intersection = (
        inter_w
        *
        inter_h
    )

    union = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection
    )

    iou = (
        intersection
        /
        np.maximum(
            union,
            1e-12,
        )
    )

    return iou


# ============================================================
# FIND LABEL DIRECTORY
# ============================================================

def find_label_directory():
    """
    Find the directory containing YOLO test labels.
    """

    print(
        "\n[SEARCH] Looking for YOLO label directory..."
    )

    for directory in LABEL_DIR_CANDIDATES:

        if not directory.exists():
            continue

        txt_files = list(
            directory.glob("*.txt")
        )

        if len(txt_files) == 0:
            continue

        print(
            f"[FOUND] {directory}"
        )

        print(
            f"[FOUND] Label files: "
            f"{len(txt_files):,}"
        )

        return directory

    raise FileNotFoundError(
        "\nCould not find YOLO label directory.\n"
        "Checked:\n"
        +
        "\n".join(
            str(x)
            for x in LABEL_DIR_CANDIDATES
        )
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(
    image_ids,
    label_dir,
):
    """
    Load YOLO labels.

    YOLO format:

        class x_center y_center width height

    All coordinates are assumed normalized [0,1].

    Convert to 1024x1024 pixel coordinates because
    the evaluation predictions are expressed in that space.
    """

    ground_truths = {}

    missing_labels = []

    total_boxes = 0

    for image_id in image_ids:

        label_path = (
            label_dir
            /
            f"{image_id}.txt"
        )

        if not label_path.exists():

            ground_truths[image_id] = []

            missing_labels.append(
                image_id
            )

            continue

        boxes = []

        with open(
            label_path,
            "r",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) < 5:
                    continue

                cls = int(
                    float(values[0])
                )

                xc = float(
                    values[1]
                )

                yc = float(
                    values[2]
                )

                w = float(
                    values[3]
                )

                h = float(
                    values[4]
                )

                x1 = (
                    xc - w / 2.0
                ) * 1024.0

                y1 = (
                    yc - h / 2.0
                ) * 1024.0

                x2 = (
                    xc + w / 2.0
                ) * 1024.0

                y2 = (
                    yc + h / 2.0
                ) * 1024.0

                boxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        cls,
                    ]
                )

        ground_truths[image_id] = boxes

        total_boxes += len(boxes)

    print(
        f"[GT] Images: "
        f"{len(ground_truths):,}"
    )

    print(
        f"[GT] Boxes: "
        f"{total_boxes:,}"
    )

    if missing_labels:

        print(
            f"[WARNING] Missing labels: "
            f"{len(missing_labels):,}"
        )

        missing_path = (
            OUTPUT_DIR
            /
            "missing_labels.txt"
        )

        with open(
            missing_path,
            "w",
        ) as f:

            for image_id in missing_labels:
                f.write(
                    f"{image_id}\n"
                )

    return ground_truths


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():
    """
    Load predictions.csv.
    """

    if not PREDICTIONS_CSV.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{PREDICTIONS_CSV}"
        )

    print(
        f"\n[LOAD] "
        f"{PREDICTIONS_CSV}"
    )

    df = pd.read_csv(
        PREDICTIONS_CSV
    )

    required_columns = [
        "image_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column "
                f"'{column}' in predictions.csv"
            )

    if "class" not in df.columns:

        df["class"] = 0

    print(
        f"[PRED] Predictions: "
        f"{len(df):,}"
    )

    print(
        f"[PRED] Images: "
        f"{df['image_id'].nunique():,}"
    )

    return df


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions(
    predictions_df,
    ground_truths,
):
    """
    Greedy confidence-sorted matching.

    Each prediction can match only one GT.
    Each GT can match only one prediction.

    Returns:

        prediction_results
        false_negatives
    """

    prediction_results = []

    false_negatives = []

    grouped = (
        predictions_df
        .groupby(
            "image_id",
            sort=False,
        )
    )

    all_image_ids = set(
        ground_truths.keys()
    )

    all_image_ids.update(
        predictions_df[
            "image_id"
        ].unique()
    )

    print(
        "\n[MATCH] Matching predictions..."
    )

    for image_id in all_image_ids:

        if image_id in ground_truths:

            gt_boxes = (
                ground_truths[
                    image_id
                ]
            )

        else:

            gt_boxes = []

        if image_id in grouped.groups:

            pred_df = (
                grouped.get_group(
                    image_id
                )
                .sort_values(
                    "confidence",
                    ascending=False,
                )
            )

        else:

            pred_df = pd.DataFrame(
                columns=predictions_df.columns
            )

        matched_gt = set()

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        for _, row in pred_df.iterrows():

            pred_box = np.array(
                [
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"],
                ],
                dtype=np.float32,
            )

            best_iou = 0.0
            best_gt_index = -1

            if len(gt_boxes) > 0:

                gt_array = np.asarray(
                    [
                        box[:4]
                        for box in gt_boxes
                    ],
                    dtype=np.float32,
                )

                ious = box_iou(
                    pred_box.reshape(
                        1,
                        4,
                    ),
                    gt_array,
                )[0]

                order = np.argsort(
                    ious
                )[::-1]

                for gt_index in order:

                    if int(
                        gt_index
                    ) in matched_gt:

                        continue

                    candidate_iou = float(
                        ious[gt_index]
                    )

                    if (
                        candidate_iou
                        >= IOU_THRESHOLD
                    ):

                        best_iou = (
                            candidate_iou
                        )

                        best_gt_index = int(
                            gt_index
                        )

                        break

            is_tp = (
                best_gt_index >= 0
                and
                best_iou
                >= IOU_THRESHOLD
            )

            if is_tp:

                matched_gt.add(
                    best_gt_index
                )

                result = (
                    "TP"
                )

            else:

                result = (
                    "FP"
                )

            prediction_results.append(
                {
                    "image_id": image_id,
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "confidence": float(
                        row["confidence"]
                    ),
                    "class": int(
                        row["class"]
                    ),
                    "result": result,
                    "iou": float(
                        best_iou
                    ),
                    "matched_gt_index": (
                        best_gt_index
                    ),
                }
            )

        # ----------------------------------------------------
        # False negatives
        # ----------------------------------------------------

        for gt_index, gt_box in enumerate(
            gt_boxes
        ):

            if gt_index in matched_gt:
                continue

            false_negatives.append(
                {
                    "image_id": image_id,
                    "x1": float(
                        gt_box[0]
                    ),
                    "y1": float(
                        gt_box[1]
                    ),
                    "x2": float(
                        gt_box[2]
                    ),
                    "y2": float(
                        gt_box[3]
                    ),
                    "class": int(
                        gt_box[4]
                    ),
                    "gt_index": int(
                        gt_index
                    ),
                }
            )

    prediction_results = pd.DataFrame(
        prediction_results
    )

    false_negatives = pd.DataFrame(
        false_negatives
    )

    return (
        prediction_results,
        false_negatives,
    )


# ============================================================
# CONFIDENCE BIN ANALYSIS
# ============================================================

def confidence_bin_analysis(
    results,
):
    """
    Compute TP / FP statistics by confidence interval.
    """

    rows = []

    for lower, upper in zip(
        CONFIDENCE_BINS[:-1],
        CONFIDENCE_BINS[1:],
    ):

        if upper >= 1.0:

            mask = (
                results["confidence"]
                >= lower
            )

        else:

            mask = (
                (results["confidence"] >= lower)
                &
                (results["confidence"] < upper)
            )

        subset = results[
            mask
        ]

        predictions = len(
            subset
        )

        tp = int(
            (
                subset["result"]
                == "TP"
            ).sum()
        )

        fp = int(
            (
                subset["result"]
                == "FP"
            ).sum()
        )

        precision = (
            tp
            /
            predictions
            if predictions > 0
            else 0.0
        )

        rows.append(
            {
                "confidence_min": lower,
                "confidence_max": upper,
                "predictions": predictions,
                "TP": tp,
                "FP": fp,
                "precision": precision,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results,
    false_negatives,
):
    """
    Print main analysis.
    """

    tp = int(
        (
            results["result"]
            == "TP"
        ).sum()
    )

    fp = int(
        (
            results["result"]
            == "FP"
        ).sum()
    )

    fn = len(
        false_negatives
    )

    precision = (
        tp
        /
        (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp
        /
        (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONFIDENCE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Predictions: {len(results):,}"
    )

    print(
        f"TP:          {tp:,}"
    )

    print(
        f"FP:          {fp:,}"
    )

    print(
        f"FN:          {fn:,}"
    )

    print(
        f"Precision:   {precision:.6f}"
    )

    print(
        f"Recall:      {recall:.6f}"
    )

    print(
        f"F1:          {f1:.6f}"
    )

    print(
        "\nConfidence statistics:"
    )

    print(
        results[
            "confidence"
        ].describe()
    )

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    tp_df = results[
        results["result"] == "TP"
    ]

    fp_df = results[
        results["result"] == "FP"
    ]

    print(
        "\nTP confidence:"
    )

    if len(tp_df) > 0:
        print(
            tp_df[
                "confidence"
            ].describe()
        )

    print(
        "\nFP confidence:"
    )

    if len(fp_df) > 0:
        print(
            fp_df[
                "confidence"
            ].describe()
        )

    # --------------------------------------------------------
    # High-confidence FP
    # --------------------------------------------------------

    high_fp = fp_df[
        fp_df["confidence"]
        >= HIGH_CONF_FP_THRESHOLD
    ]

    print(
        "\n"
        f"High-confidence FP "
        f"(>= {HIGH_CONF_FP_THRESHOLD:.2f}): "
        f"{len(high_fp):,}"
    )

    # --------------------------------------------------------
    # Low-confidence TP
    # --------------------------------------------------------

    low_tp = tp_df[
        tp_df["confidence"]
        < LOW_CONF_TP_THRESHOLD
    ]

    print(
        f"Low-confidence TP "
        f"(< {LOW_CONF_TP_THRESHOLD:.2f}): "
        f"{len(low_tp):,}"
    )

    # --------------------------------------------------------
    # High-confidence TP
    # --------------------------------------------------------

    high_tp = tp_df[
        tp_df["confidence"]
        >= HIGH_CONF_TP_THRESHOLD
    ]

    print(
        f"High-confidence TP "
        f"(>= {HIGH_CONF_TP_THRESHOLD:.2f}): "
        f"{len(high_tp):,}"
    )

    print(
        "=" * 70
    )


# ============================================================
# HISTOGRAM
# ============================================================

def plot_confidence_histogram(
    results,
):
    """
    Plot TP and FP confidence distributions.
    """

    tp = results[
        results["result"] == "TP"
    ]["confidence"]

    fp = results[
        results["result"] == "FP"
    ]["confidence"]

    plt.figure(
        figsize=(10, 6)
    )

    bins = np.linspace(
        0.0,
        1.0,
        51,
    )

    plt.hist(
        tp,
        bins=bins,
        alpha=0.6,
        label="TP",
    )

    plt.hist(
        fp,
        bins=bins,
        alpha=0.6,
        label="FP",
    )

    plt.xlabel(
        "Confidence"
    )

    plt.ylabel(
        "Number of predictions"
    )

    plt.title(
        "Prediction confidence: TP vs FP"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output_path = (
        OUTPUT_DIR
        /
        "confidence_histogram_tp_vs_fp.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"[SAVE] {output_path}"
    )


# ============================================================
# CUMULATIVE CURVE
# ============================================================

def plot_cumulative_curve(
    results,
):
    """
    Show how many TP and FP remain as the
    confidence threshold increases.
    """

    thresholds = np.linspace(
        0.0,
        1.0,
        101,
    )

    tp_counts = []
    fp_counts = []

    precision_values = []
    recall_values = []

    total_tp = (
        results["result"]
        == "TP"
    ).sum()

    total_gt = (
        total_tp
        +
        results["result"]
        .eq("FP")
        .sum()
    )

    # FN must be accounted for through GT count.
    # total_gt here is reconstructed from TP+FN
    # later using the actual FN count.

    # We don't know FN directly here, so use
    # total GT = TP + FN from external summary
    # if available.

    for threshold in thresholds:

        subset = results[
            results["confidence"]
            >= threshold
        ]

        tp = int(
            (
                subset["result"]
                == "TP"
            ).sum()
        )

        fp = int(
            (
                subset["result"]
                == "FP"
            ).sum()
        )

        tp_counts.append(
            tp
        )

        fp_counts.append(
            fp
        )

        precision = (
            tp
            /
            (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        precision_values.append(
            precision
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thresholds,
        tp_counts,
        label="TP",
    )

    plt.plot(
        thresholds,
        fp_counts,
        label="FP",
    )

    plt.xlabel(
        "Confidence threshold"
    )

    plt.ylabel(
        "Number of detections"
    )

    plt.title(
        "TP and FP vs confidence threshold"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.xlim(
        0,
        1,
    )

    output_path = (
        OUTPUT_DIR
        /
        "tp_fp_vs_confidence_threshold.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"[SAVE] {output_path}"
    )


# ============================================================
# PRECISION CURVE
# ============================================================

def plot_precision_curve(
    results,
    false_negatives,
):
    """
    Precision and recall as confidence threshold changes.
    """

    thresholds = np.linspace(
        0.001,
        0.999,
        200,
    )

    precisions = []
    recalls = []
    f1s = []

    total_gt = (
        len(
            false_negatives
        )
        +
        (
            results["result"]
            == "TP"
        ).sum()
    )

    for threshold in thresholds:

        subset = results[
            results["confidence"]
            >= threshold
        ]

        tp = int(
            (
                subset["result"]
                == "TP"
            ).sum()
        )

        fp = int(
            (
                subset["result"]
                == "FP"
            ).sum()
        )

        precision = (
            tp
            /
            (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp
            /
            total_gt
            if total_gt > 0
            else 0.0
        )

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1s.append(
            f1
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thresholds,
        precisions,
        label="Precision",
    )

    plt.plot(
        thresholds,
        recalls,
        label="Recall",
    )

    plt.plot(
        thresholds,
        f1s,
        label="F1",
    )

    plt.xlabel(
        "Confidence threshold"
    )

    plt.ylabel(
        "Metric"
    )

    plt.title(
        "Precision / Recall / F1 vs confidence"
    )

    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output_path = (
        OUTPUT_DIR
        /
        "precision_recall_f1_vs_confidence.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"[SAVE] {output_path}"
    )


# ============================================================
# SAVE INTERESTING CASES
# ============================================================

def save_interesting_cases(
    results,
    false_negatives,
):
    """
    Save CSVs for the cases we want to inspect.
    """

    tp = results[
        results["result"] == "TP"
    ].copy()

    fp = results[
        results["result"] == "FP"
    ].copy()

    # --------------------------------------------------------
    # Low confidence TP
    # --------------------------------------------------------

    low_tp = tp[
        tp["confidence"]
        < LOW_CONF_TP_THRESHOLD
    ].sort_values(
        "confidence"
    )

    low_tp_path = (
        OUTPUT_DIR
        /
        "low_confidence_tp.csv"
    )

    low_tp.to_csv(
        low_tp_path,
        index=False,
    )

    # --------------------------------------------------------
    # High confidence TP
    # --------------------------------------------------------

    high_tp = tp[
        tp["confidence"]
        >= HIGH_CONF_TP_THRESHOLD
    ].sort_values(
        "confidence",
        ascending=False,
    )

    high_tp_path = (
        OUTPUT_DIR
        /
        "high_confidence_tp.csv"
    )

    high_tp.to_csv(
        high_tp_path,
        index=False,
    )

    # --------------------------------------------------------
    # High confidence FP
    # --------------------------------------------------------

    high_fp = fp[
        fp["confidence"]
        >= HIGH_CONF_FP_THRESHOLD
    ].sort_values(
        "confidence",
        ascending=False,
    )

    high_fp_path = (
        OUTPUT_DIR
        /
        "high_confidence_fp.csv"
    )

    high_fp.to_csv(
        high_fp_path,
        index=False,
    )

    # --------------------------------------------------------
    # All FPs
    # --------------------------------------------------------

    fp_path = (
        OUTPUT_DIR
        /
        "all_false_positives.csv"
    )

    fp.to_csv(
        fp_path,
        index=False,
    )

    # --------------------------------------------------------
    # All FNs
    # --------------------------------------------------------

    fn_path = (
        OUTPUT_DIR
        /
        "false_negatives.csv"
    )

    false_negatives.to_csv(
        fn_path,
        index=False,
    )

    print(
        "\n[SAVE] Interesting cases:"
    )

    print(
        f"  {low_tp_path}"
    )

    print(
        f"  {high_tp_path}"
    )

    print(
        f"  {high_fp_path}"
    )

    print(
        f"  {fp_path}"
    )

    print(
        f"  {fn_path}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    results,
    false_negatives,
):
    """
    Save textual summary.
    """

    tp = int(
        (
            results["result"]
            == "TP"
        ).sum()
    )

    fp = int(
        (
            results["result"]
            == "FP"
        ).sum()
    )

    fn = len(
        false_negatives
    )

    precision = (
        tp
        /
        (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp
        /
        (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    tp_df = results[
        results["result"] == "TP"
    ]

    fp_df = results[
        results["result"] == "FP"
    ]

    high_fp = fp_df[
        fp_df["confidence"]
        >= HIGH_CONF_FP_THRESHOLD
    ]

    low_tp = tp_df[
        tp_df["confidence"]
        < LOW_CONF_TP_THRESHOLD
    ]

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
            "SWIN + YOLO CONFIDENCE ANALYSIS\n"
        )

        f.write(
            "=" * 60
            +
            "\n\n"
        )

        f.write(
            f"IoU threshold: "
            f"{IOU_THRESHOLD:.2f}\n"
        )

        f.write(
            f"Predictions: {len(results):,}\n"
        )

        f.write(
            f"TP: {tp:,}\n"
        )

        f.write(
            f"FP: {fp:,}\n"
        )

        f.write(
            f"FN: {fn:,}\n"
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
            "CONFIDENCE STATISTICS\n"
        )

        f.write(
            "-" * 60
            +
            "\n"
        )

        f.write(
            "\nAll predictions:\n"
        )

        f.write(
            results[
                "confidence"
            ]
            .describe()
            .to_string()
        )

        f.write(
            "\n\nTP:\n"
        )

        f.write(
            tp_df[
                "confidence"
            ]
            .describe()
            .to_string()
        )

        f.write(
            "\n\nFP:\n"
        )

        f.write(
            fp_df[
                "confidence"
            ]
            .describe()
            .to_string()
        )

        f.write(
            "\n\n"
        )

        f.write(
            f"Low-confidence TP "
            f"(< {LOW_CONF_TP_THRESHOLD:.2f}): "
            f"{len(low_tp):,}\n"
        )

        f.write(
            f"High-confidence FP "
            f"(>= {HIGH_CONF_FP_THRESHOLD:.2f}): "
            f"{len(high_fp):,}\n"
        )

    print(
        f"[SAVE] {summary_path}"
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
        "SWIN + YOLO CONFIDENCE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nThis script does NOT run model inference."
    )

    print(
        "It analyzes the existing predictions.csv."
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    predictions_df = (
        load_predictions()
    )

    # --------------------------------------------------------
    # Find labels
    # --------------------------------------------------------

    label_dir = (
        find_label_directory()
    )

    # --------------------------------------------------------
    # Image IDs
    # --------------------------------------------------------

    image_ids = (
        predictions_df[
            "image_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truths = (
        load_ground_truth(
            image_ids,
            label_dir,
        )
    )

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    (
        results,
        false_negatives,
    ) = match_predictions(
        predictions_df,
        ground_truths,
    )

    # --------------------------------------------------------
    # Save complete classification
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        /
        "confidence_analysis.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print(
        f"\n[SAVE] {results_path}"
    )

    # --------------------------------------------------------
    # Save FN
    # --------------------------------------------------------

    fn_path = (
        OUTPUT_DIR
        /
        "false_negatives.csv"
    )

    false_negatives.to_csv(
        fn_path,
        index=False,
    )

    print(
        f"[SAVE] {fn_path}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results,
        false_negatives,
    )

    # --------------------------------------------------------
    # Confidence bins
    # --------------------------------------------------------

    bins_df = (
        confidence_bin_analysis(
            results
        )
    )

    bins_path = (
        OUTPUT_DIR
        /
        "confidence_bins.csv"
    )

    bins_df.to_csv(
        bins_path,
        index=False,
    )

    print(
        f"[SAVE] {bins_path}"
    )

    print(
        "\n"
        + bins_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Interesting cases
    # --------------------------------------------------------

    save_interesting_cases(
        results,
        false_negatives,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print(
        "\n[PLOT] Creating plots..."
    )

    plot_confidence_histogram(
        results
    )

    plot_cumulative_curve(
        results
    )

    plot_precision_curve(
        results,
        false_negatives,
    )

    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    save_summary(
        results,
        false_negatives,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nThe most important files are:"
    )

    print(
        "  confidence_analysis.csv"
    )

    print(
        "  confidence_bins.csv"
    )

    print(
        "  false_negatives.csv"
    )

    print(
        "  high_confidence_fp.csv"
    )

    print(
        "  low_confidence_tp.csv"
    )

    print(
        "  confidence_histogram_tp_vs_fp.png"
    )

    print(
        "  precision_recall_f1_vs_confidence.png"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()