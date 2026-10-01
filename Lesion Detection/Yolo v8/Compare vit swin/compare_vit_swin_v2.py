from pathlib import Path

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SWIN_METRICS = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/evaluation/metrics.json"
)

VIT_METRICS = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/evaluation_vit_yolo/metrics.json"
)

OUTPUT_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/Compare vit swin/metrics_comparison"
)


# ============================================================
# HELPERS
# ============================================================

def get_value(data, keys, default=np.nan):
    for key in keys:
        if key in data:
            return data[key]

    return default


def get_iou_series(data):
    """
    Supports both the new evaluator format and the older
    Swin evaluator format.
    """

    if "iou_thresholds" in data:
        return np.array(
            data["iou_thresholds"],
            dtype=float,
        )

    thresholds = []

    for key in data.keys():
        if key.startswith("AP@"):
            try:
                threshold = float(
                    key.split("@", 1)[1]
                )
                thresholds.append(threshold)
            except ValueError:
                pass

    if thresholds:
        return np.array(
            sorted(set(thresholds)),
            dtype=float,
        )

    raise ValueError(
        "Could not find IoU thresholds in metrics.json."
    )


def get_series(data, metric, iou_thresholds):
    """
    Return a metric indexed by IoU threshold.

    Supports:
      - new evaluator:
            ap_by_iou
            tp_by_iou
            fp_by_iou
            fn_by_iou
            tpr_by_iou

      - older evaluator:
            AP@0.50
            TP@0.50
            ...
    """

    new_key = f"{metric}_by_iou"

    if new_key in data:
        values = data[new_key]

        if isinstance(values, dict):
            return np.array(
                [
                    float(values.get(
                        f"{threshold:.2f}",
                        values.get(
                            str(threshold),
                            np.nan,
                        ),
                    ))
                    for threshold in iou_thresholds
                ],
                dtype=float,
            )

        return np.array(
            values,
            dtype=float,
        )

    values = []

    for threshold in iou_thresholds:
        candidates = [
            f"{metric.upper()}@{threshold:.2f}",
            f"{metric}@{threshold:.2f}",
        ]

        value = np.nan

        for key in candidates:
            if key in data:
                value = float(data[key])
                break

        values.append(value)

    return np.array(
        values,
        dtype=float,
    )


def load_metrics(path, model_name):
    if not path.exists():
        raise FileNotFoundError(
            f"\n[{model_name}] metrics.json not found:\n{path}\n"
        )

    with open(path, "r") as f:
        data = json.load(f)

    print()
    print(f"[LOAD] {model_name}")
    print(f"Path: {path}")

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("SWIN + YOLO vs ViT + YOLO")
    print("METRICS COMPARISON")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    swin = load_metrics(
        SWIN_METRICS,
        "Swin + YOLO",
    )

    vit = load_metrics(
        VIT_METRICS,
        "ViT + YOLO",
    )

    # --------------------------------------------------------
    # IOU THRESHOLDS
    # --------------------------------------------------------

    swin_iou = get_iou_series(swin)
    vit_iou = get_iou_series(vit)

    common_iou = np.array(
        sorted(
            set(
                np.round(swin_iou, 2)
            )
            &
            set(
                np.round(vit_iou, 2)
            )
        ),
        dtype=float,
    )

    if len(common_iou) == 0:
        raise ValueError(
            "Swin and ViT do not share any IoU thresholds."
        )

    print()
    print(
        f"[INFO] Common IoU thresholds: "
        f"{len(common_iou)}"
    )

    # --------------------------------------------------------
    # SERIES
    # --------------------------------------------------------

    swin_ap = get_series(
        swin,
        "ap",
        common_iou,
    )

    vit_ap = get_series(
        vit,
        "ap",
        common_iou,
    )

    swin_tp = get_series(
        swin,
        "tp",
        common_iou,
    )

    vit_tp = get_series(
        vit,
        "tp",
        common_iou,
    )

    swin_fp = get_series(
        swin,
        "fp",
        common_iou,
    )

    vit_fp = get_series(
        vit,
        "fp",
        common_iou,
    )

    swin_fn = get_series(
        swin,
        "fn",
        common_iou,
    )

    vit_fn = get_series(
        vit,
        "fn",
        common_iou,
    )

    swin_tpr = get_series(
        swin,
        "tpr",
        common_iou,
    )

    vit_tpr = get_series(
        vit,
        "tpr",
        common_iou,
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    rows = []

    summary_metrics = [
        (
            "mAP@0.50",
            ["mAP@0.50", "mAP50"],
            ["mAP@0.50", "mAP50"],
        ),
        (
            "mAP@0.50:0.95",
            ["mAP@0.50:0.95", "mAP50-95"],
            ["mAP@0.50:0.95", "mAP50-95"],
        ),
        (
            "Precision@0.50",
            ["precision@0.50", "precision"],
            ["precision@0.50", "precision"],
        ),
        (
            "Recall / TPR@0.50",
            ["tpr@0.50", "recall@0.50", "recall"],
            ["tpr@0.50", "recall@0.50", "recall"],
        ),
        (
            "F1@0.50",
            ["f1@0.50", "f1"],
            ["f1@0.50", "f1"],
        ),
        (
            "TP@0.50",
            ["TP@0.50", "TP"],
            ["TP@0.50", "TP"],
        ),
        (
            "FP@0.50",
            ["FP@0.50", "FP"],
            ["FP@0.50", "FP"],
        ),
        (
            "FN@0.50",
            ["FN@0.50", "FN"],
            ["FN@0.50", "FN"],
        ),
        (
            "Predictions",
            [
                "predictions_after_nms",
                "predictions",
                "num_predictions",
            ],
            [
                "predictions_after_nms",
                "predictions",
                "num_predictions",
            ],
        ),
        (
            "Ground truth boxes",
            [
                "ground_truth_boxes",
                "num_ground_truth_boxes",
            ],
            [
                "ground_truth_boxes",
                "num_ground_truth_boxes",
            ],
        ),
        (
            "Test images",
            [
                "test_images",
                "num_images",
            ],
            [
                "test_images",
                "num_images",
            ],
        ),
    ]

    for metric_name, swin_keys, vit_keys in summary_metrics:

        swin_value = get_value(
            swin,
            swin_keys,
        )

        vit_value = get_value(
            vit,
            vit_keys,
        )

        rows.append(
            {
                "Metric": metric_name,
                "Swin + YOLO": swin_value,
                "ViT + YOLO": vit_value,
            }
        )

    summary_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # IOU TABLE
    # --------------------------------------------------------

    iou_df = pd.DataFrame(
        {
            "IoU": common_iou,
            "Swin AP": swin_ap,
            "ViT AP": vit_ap,
            "Swin TP": swin_tp,
            "ViT TP": vit_tp,
            "Swin FP": swin_fp,
            "ViT FP": vit_fp,
            "Swin FN": swin_fn,
            "ViT FN": vit_fn,
            "Swin TPR": swin_tpr,
            "ViT TPR": vit_tpr,
        }
    )

    # --------------------------------------------------------
    # SAVE TABLES
    # --------------------------------------------------------

    summary_csv = (
        OUTPUT_DIR
        / "summary_comparison.csv"
    )

    iou_csv = (
        OUTPUT_DIR
        / "iou_comparison.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    iou_df.to_csv(
        iou_csv,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
                if pd.notna(x)
                else "NaN"
            ),
        )
    )

    print()
    print("=" * 80)
    print("TPR BY IoU")
    print("=" * 80)

    print(
        iou_df[
            [
                "IoU",
                "Swin TPR",
                "ViT TPR",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # PLOT TPR
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        common_iou,
        swin_tpr,
        marker="o",
        label="Swin + YOLO",
    )

    plt.plot(
        common_iou,
        vit_tpr,
        marker="o",
        label="ViT + YOLO",
    )

    plt.xlabel("IoU threshold")
    plt.ylabel("TPR / Recall")
    plt.title("TPR vs IoU threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(common_iou)
    plt.tight_layout()

    tpr_plot = (
        OUTPUT_DIR
        / "tpr_vs_iou_comparison.png"
    )

    plt.savefig(
        tpr_plot,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # PLOT AP
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        common_iou,
        swin_ap,
        marker="o",
        label="Swin + YOLO",
    )

    plt.plot(
        common_iou,
        vit_ap,
        marker="o",
        label="ViT + YOLO",
    )

    plt.xlabel("IoU threshold")
    plt.ylabel("AP")
    plt.title("AP vs IoU threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(common_iou)
    plt.tight_layout()

    ap_plot = (
        OUTPUT_DIR
        / "ap_vs_iou_comparison.png"
    )

    plt.savefig(
        ap_plot,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE TEXT REPORT
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "comparison_report.txt"
    )

    with open(
        report_path,
        "w",
    ) as f:

        f.write(
            "SWIN + YOLO vs ViT + YOLO\n"
        )

        f.write(
            "METRICS COMPARISON\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            summary_df.to_string(
                index=False,
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "TPR / AP BY IoU\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            iou_df.to_string(
                index=False,
            )
        )

        f.write(
            "\n"
        )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(summary_csv)
    print(iou_csv)
    print(tpr_plot)
    print(ap_plot)
    print(report_path)

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
