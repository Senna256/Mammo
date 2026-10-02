#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SWIN_JSON = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/evaluation/positives/outputs/test_evaluation.json")
VIT_JSON = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/evaluation_positives/outputs/metrics.json")
OUTPUT_DIR = Path("/home/enric/Mammo/Lesion Detection/Yolo v8/Compare vit swin/compare_positive_train")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"JSON not found:\n{path}")
    with path.open("r") as f:
        return json.load(f)


def unwrap_metrics(data):
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        return data["metrics"]
    return data


def get_iou_series(data):
    metrics = unwrap_metrics(data)

    if "iou_thresholds" in metrics:
        return np.array(metrics["iou_thresholds"], dtype=float)

    for key in (
        "AP_by_IoU", "ap_by_iou",
        "TPR_by_IoU", "tpr_by_iou",
        "TP_by_IoU", "tp_by_iou",
        "FP_by_IoU", "fp_by_iou",
        "FN_by_IoU", "fn_by_iou",
    ):
        if key in metrics and isinstance(metrics[key], dict):
            thresholds = []
            for threshold in metrics[key]:
                try:
                    thresholds.append(float(threshold))
                except (TypeError, ValueError):
                    pass
            if thresholds:
                return np.array(sorted(set(thresholds)), dtype=float)

    thresholds = []
    for key in metrics:
        if isinstance(key, str) and key.startswith("AP@"):
            try:
                thresholds.append(float(key.split("@", 1)[1]))
            except ValueError:
                pass

    if thresholds:
        return np.array(sorted(set(thresholds)), dtype=float)

    raise ValueError(f"Could not find IoU thresholds. Available keys: {list(metrics.keys())}")


def get_series(data, metric):
    metrics = unwrap_metrics(data)

    aliases = {
        "AP": ("ap_by_iou", "AP_by_IoU"),
        "TPR": ("tpr_by_iou", "TPR_by_IoU"),
        "TP": ("tp_by_iou", "TP_by_IoU"),
        "FP": ("fp_by_iou", "FP_by_IoU"),
        "FN": ("fn_by_iou", "FN_by_IoU"),
        "Precision": ("precision_by_iou", "Precision_by_IoU"),
        "F1": ("f1_by_iou", "F1_by_IoU"),
    }

    for key in aliases.get(metric, ()):
        if key in metrics and isinstance(metrics[key], dict):
            result = {}
            for threshold, value in metrics[key].items():
                try:
                    result[float(threshold)] = float(value)
                except (TypeError, ValueError):
                    pass
            if result:
                return result

    result = {}
    prefix = metric + "@"
    for key, value in metrics.items():
        if isinstance(key, str) and key.startswith(prefix):
            try:
                result[float(key.split("@", 1)[1])] = float(value)
            except (TypeError, ValueError):
                pass
    return result


def scalar(metrics, *keys):
    for key in keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                pass
    return np.nan


def summary(data):
    metrics = unwrap_metrics(data)
    ap = get_series(data, "AP")
    tpr = get_series(data, "TPR")
    precision = get_series(data, "Precision")
    f1 = get_series(data, "F1")

    map50 = scalar(metrics, "mAP@0.50", "mAP50", "map50")
    if np.isnan(map50):
        map50 = ap.get(0.50, np.nan)

    map5095 = scalar(metrics, "mAP@0.50:0.95", "mAP50-95", "map50_95")
    required = {round(x, 2) for x in np.arange(0.50, 0.951, 0.05)}
    if np.isnan(map5095) and required.issubset({round(x, 2) for x in ap}):
        map5095 = float(np.mean([ap[x] for x in ap if 0.50 <= x <= 0.95]))

    tpr50 = scalar(metrics, "TPR@0.50", "tpr@0.50", "recall@0.50", "recall")
    if np.isnan(tpr50):
        tpr50 = tpr.get(0.50, np.nan)

    precision50 = scalar(metrics, "Precision@0.50", "precision@0.50", "precision")
    if np.isnan(precision50):
        precision50 = precision.get(0.50, np.nan)

    f150 = scalar(metrics, "F1@0.50", "f1@0.50", "f1")
    if np.isnan(f150):
        f150 = f1.get(0.50, np.nan)

    return {
        "mAP@0.50": map50,
        "mAP@0.50:0.95": map5095,
        "Precision@0.50": precision50,
        "Recall / TPR@0.50": tpr50,
        "F1@0.50": f150,
        "TP@0.50": scalar(metrics, "TP@0.50", "TP"),
        "FP@0.50": scalar(metrics, "FP@0.50", "FP"),
        "FN@0.50": scalar(metrics, "FN@0.50", "FN"),
        "Predictions": scalar(metrics, "predictions_after_nms", "predictions", "Predictions"),
        "Ground truth boxes": scalar(metrics, "ground_truth_boxes", "Ground truth boxes"),
        "Test images": scalar(metrics, "test_images", "Test images"),
    }


def build_iou_table(swin, vit):

    def extract_metric_at_iou(data, metric, iou):
        metrics = unwrap_metrics(data)

        aliases = {
            "AP": [
                "ap_by_iou",
                "AP_by_IoU",
            ],
            "TPR": [
                "tpr_by_iou",
                "TPR_by_IoU",
            ],
            "TP": [
                "tp_by_iou",
                "TP_by_IoU",
            ],
            "FP": [
                "fp_by_iou",
                "FP_by_IoU",
            ],
            "FN": [
                "fn_by_iou",
                "FN_by_IoU",
            ],
            "Precision": [
                "precision_by_iou",
                "Precision_by_IoU",
            ],
            "F1": [
                "f1_by_iou",
                "F1_by_IoU",
            ],
        }

        target_iou = round(
            float(iou),
            2,
        )

        thresholds = metrics.get(
            "iou_thresholds"
        )

        for key in aliases.get(
            metric,
            [],
        ):

            if key not in metrics:
                continue

            values = metrics[key]

            # List format:
            # iou_thresholds[i] <-> values[i]
            if isinstance(
                values,
                list,
            ):

                if thresholds is None:
                    continue

                if len(values) != len(
                    thresholds
                ):
                    raise ValueError(
                        f"{key}: "
                        f"{len(values)} values but "
                        f"{len(thresholds)} IoU thresholds."
                    )

                for threshold, value in zip(
                    thresholds,
                    values,
                ):

                    threshold = round(
                        float(threshold),
                        2,
                    )

                    if threshold == target_iou:
                        return float(value)

            # Dictionary format:
            # {IoU: value}
            elif isinstance(
                values,
                dict,
            ):

                for threshold, value in (
                    values.items()
                ):

                    try:
                        threshold = round(
                            float(threshold),
                            2,
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if threshold == target_iou:
                        return float(value)

        # Precision is not stored in the ViT JSON.
        # Derive it from TP and FP.
        if metric == "Precision":

            tp = extract_metric_at_iou(
                data,
                "TP",
                iou,
            )

            fp = extract_metric_at_iou(
                data,
                "FP",
                iou,
            )

            if (
                not np.isnan(tp)
                and not np.isnan(fp)
                and (tp + fp) > 0
            ):
                return tp / (
                    tp + fp
                )

        # Flat format:
        # TPR@0.50, AP@0.50, ...
        possible_keys = [
            f"{metric}@{target_iou:.2f}",
            f"{metric}@{target_iou:.1f}",
            f"{metric.lower()}@{target_iou:.2f}",
            f"{metric.lower()}@{target_iou:.1f}",
        ]

        for key in possible_keys:

            if key in metrics:

                try:
                    return float(
                        metrics[key]
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    return np.nan

        return np.nan

    swin_ious = {
        round(
            float(x),
            2,
        )
        for x in get_iou_series(swin)
    }

    vit_ious = {
        round(
            float(x),
            2,
        )
        for x in get_iou_series(vit)
    }

    common_ious = sorted(
        swin_ious & vit_ious
    )

    rows = []

    for iou in common_ious:

        rows.append(
            {
                "IoU": iou,

                "Swin AP": extract_metric_at_iou(
                    swin,
                    "AP",
                    iou,
                ),

                "ViT AP": extract_metric_at_iou(
                    vit,
                    "AP",
                    iou,
                ),

                "Swin TPR": extract_metric_at_iou(
                    swin,
                    "TPR",
                    iou,
                ),

                "ViT TPR": extract_metric_at_iou(
                    vit,
                    "TPR",
                    iou,
                ),

                "Swin Precision": extract_metric_at_iou(
                    swin,
                    "Precision",
                    iou,
                ),

                "ViT Precision": extract_metric_at_iou(
                    vit,
                    "Precision",
                    iou,
                ),

                "Swin TP": extract_metric_at_iou(
                    swin,
                    "TP",
                    iou,
                ),

                "ViT TP": extract_metric_at_iou(
                    vit,
                    "TP",
                    iou,
                ),

                "Swin FP": extract_metric_at_iou(
                    swin,
                    "FP",
                    iou,
                ),

                "ViT FP": extract_metric_at_iou(
                    vit,
                    "FP",
                    iou,
                ),

                "Swin FN": extract_metric_at_iou(
                    swin,
                    "FN",
                    iou,
                ),

                "ViT FN": extract_metric_at_iou(
                    vit,
                    "FN",
                    iou,
                ),
            }
        )

    return pd.DataFrame(rows)

def plot_series(table, y1, y2, ylabel, title, filename):
    if table.empty:
        return
    plt.figure(figsize=(9, 6))
    plt.plot(table["IoU"], table[y1], marker="o", label="Swin + YOLO")
    plt.plot(table["IoU"], table[y2], marker="o", label="ViT + YOLO")
    plt.xlabel("IoU threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SWIN + YOLO vs ViT + YOLO — POSITIVE-ONLY TRAINING")
    print("METRICS COMPARISON")
    print("=" * 80)

    swin = load_json(SWIN_JSON)
    vit = load_json(VIT_JSON)

    print(f"[Swin] IoUs: {get_iou_series(swin).tolist()}")
    print(f"[ViT]  IoUs: {get_iou_series(vit).tolist()}")

    swin_summary = summary(swin)
    vit_summary = summary(vit)

    summary_df = pd.DataFrame({
        "Metric": list(swin_summary),
        "Swin + YOLO": list(swin_summary.values()),
        "ViT + YOLO": [vit_summary[k] for k in swin_summary],
    })

    table = build_iou_table(swin, vit)

    summary_df.to_csv(OUTPUT_DIR / "summary_comparison.csv", index=False)
    table.to_csv(OUTPUT_DIR / "iou_comparison.csv", index=False)

    plot_series(
        table, "Swin TPR", "ViT TPR", "TPR / Recall",
        "TPR vs IoU — Positive-only training", "TPR_vs_IoU.png"
    )
    plot_series(
        table, "Swin AP", "ViT AP", "AP",
        "AP vs IoU — Positive-only training", "AP_vs_IoU.png"
    )

    with (OUTPUT_DIR / "comparison_report.txt").open("w") as f:
        f.write("SWIN + YOLO vs ViT + YOLO — POSITIVE-ONLY TRAINING\n\n")
        f.write("SUMMARY\n=======\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nCOMMON IoU THRESHOLDS\n=====================\n")
        f.write(table.to_string(index=False))
        f.write(
            "\n\nNOTE\n====\n"
            "The comparison uses only IoU thresholds available in both JSON files.\n"
            "mAP@0.50:0.95 is reported only when all thresholds from 0.50 to 0.95 are available.\n"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}")
    print(summary_df.to_string(index=False))
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
