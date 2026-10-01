from pathlib import Path

import csv
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils.nms import non_max_suppression


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

VIT_ROOT = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit"
)

CHECKPOINT = (
    VIT_ROOT
    / "train_v2"
    / "best.pt"
)

OUTPUT_DIR = (
    VIT_ROOT
    / "evaluation_vit_yolo"
)

VISUALIZATION_DIR = (
    OUTPUT_DIR
    / "visualizations"
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

IOU_THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.96,
        0.05,
    ),
    2,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRAINING MODULE
# ============================================================

def find_training_module():
    candidates = []

    for path in VIT_ROOT.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        required = [
            "class ViTYOLO",
            "class VindrViTDataset",
            "def collate_fn",
        ]

        if all(item in text for item in required):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not find the ViT + YOLO training module. "
            "Expected a .py file containing "
            "ViTYOLO, VindrViTDataset and collate_fn."
        )

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    selected = candidates[0]

    print(
        "[MODULE] Using training module:"
    )
    print(
        f"  {selected}"
    )

    return selected


def import_training_module():
    module_path = find_training_module()

    spec = importlib.util.spec_from_file_location(
        "vit_yolo_training_module",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import training module: {module_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


# ============================================================
# IOU
# ============================================================

def box_iou(
    boxes1,
    boxes2,
):
    if (
        len(boxes1) == 0
        or len(boxes2) == 0
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
            0.0,
            boxes1[:, 2]
            - boxes1[:, 0],
        )
        *
        np.maximum(
            0.0,
            boxes1[:, 3]
            - boxes1[:, 1],
        )
    )

    area2 = (
        np.maximum(
            0.0,
            boxes2[:, 2]
            - boxes2[:, 0],
        )
        *
        np.maximum(
            0.0,
            boxes2[:, 3]
            - boxes2[:, 1],
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
            0.0,
            x2 - x1,
        )
        *
        np.maximum(
            0.0,
            y2 - y1,
        )
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return (
        intersection
        /
        np.maximum(
            union,
            1e-9,
        )
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

    return np.array(
        [
            (
                x_center
                - width / 2.0
            )
            * IMG_SIZE,

            (
                y_center
                - height / 2.0
            )
            * IMG_SIZE,

            (
                x_center
                + width / 2.0
            )
            * IMG_SIZE,

            (
                y_center
                + height / 2.0
            )
            * IMG_SIZE,
        ],
        dtype=np.float32,
    )


# ============================================================
# GROUND TRUTH
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

    image_boxes = boxes[mask]

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
# DETECT OUTPUT
# ============================================================

def find_prediction_tensor(
    output,
):
    if isinstance(
        output,
        torch.Tensor,
    ):
        if output.ndim == 3:
            return output

        raise RuntimeError(
            "Tensor output does not have "
            f"expected 3 dimensions: {output.shape}"
        )

    if isinstance(
        output,
        (tuple, list),
    ):
        for item in output:
            try:
                result = find_prediction_tensor(
                    item
                )
                if result is not None:
                    return result
            except RuntimeError:
                continue

    if isinstance(
        output,
        dict,
    ):
        for value in output.values():
            try:
                result = find_prediction_tensor(
                    value
                )
                if result is not None:
                    return result
            except RuntimeError:
                continue

    raise RuntimeError(
        "Could not find a YOLO prediction tensor "
        f"inside output of type {type(output)}."
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    print(
        "[CHECKPOINT] Loading:"
    )
    print(
        f"  {CHECKPOINT}"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    if "model" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain the "
            "'model' state_dict."
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
            f"[CHECKPOINT] Missing keys: "
            f"{len(missing_keys)}"
        )

        for key in missing_keys[:20]:
            print(
                f"  {key}"
            )

    if unexpected_keys:
        print(
            f"[CHECKPOINT] Unexpected keys: "
            f"{len(unexpected_keys)}"
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

    for index in range(
        len(mpre) - 1,
        0,
        -1,
    ):
        mpre[index - 1] = max(
            mpre[index - 1],
            mpre[index],
        )

    indices = np.where(
        mrec[1:]
        !=
        mrec[:-1]
    )[0]

    return float(
        np.sum(
            (
                mrec[indices + 1]
                - mrec[indices]
            )
            *
            mpre[indices + 1]
        )
    )


# ============================================================
# MATCHING + AP STATISTICS
# ============================================================

def collect_statistics(
    predictions,
    ground_truths,
    iou_threshold,
):
    prediction_records = []

    total_gt = sum(
        len(boxes)
        for boxes in ground_truths.values()
    )

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
                    np.asarray(
                        prediction[:4],
                        dtype=np.float32,
                    ),
                )
            )

    prediction_records.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    if total_gt == 0:
        return (
            0.0,
            0,
            len(prediction_records),
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
            pred_box.reshape(
                1,
                4,
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
            + cumulative_fp,
            1e-12,
        )
    )

    ap = compute_ap(
        recall,
        precision,
    )

    tp = int(
        cumulative_tp[-1]
    ) if len(cumulative_tp) else 0

    fp = int(
        cumulative_fp[-1]
    ) if len(cumulative_fp) else 0

    fn = int(
        total_gt
        - tp
    )

    return (
        ap,
        tp,
        fp,
        fn,
    )


# ============================================================
# VISUALIZATION
# ============================================================

def draw_boxes(
    image,
    boxes,
    color,
    label,
):
    image = image.copy()

    for box in boxes:
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in box[:4]
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    y1 - 5,
                    15,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
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

        confidence = float(
            prediction[4]
        )

        if (
            confidence
            <
            VISUAL_CONF_THRESHOLD
        ):
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in prediction[:4]
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            f"Pred {confidence:.3f}",
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
# PLOTS
# ============================================================

def save_tpr_plot(
    metrics,
):
    thresholds = np.asarray(
        metrics["iou_thresholds"],
        dtype=np.float32,
    )

    tpr = np.asarray(
        metrics["tpr_by_iou"],
        dtype=np.float32,
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.plot(
        thresholds,
        tpr,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "IoU threshold"
    )

    plt.ylabel(
        "TPR / Recall"
    )

    plt.title(
        "ViT + YOLO — TPR vs IoU"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "tpr_vs_iou.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


def save_ap_plot(
    metrics,
):
    thresholds = np.asarray(
        metrics["iou_thresholds"],
        dtype=np.float32,
    )

    ap = np.asarray(
        metrics["ap_by_iou"],
        dtype=np.float32,
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.plot(
        thresholds,
        ap,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "IoU threshold"
    )

    plt.ylabel(
        "AP"
    )

    plt.title(
        "ViT + YOLO — AP vs IoU"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "ap_vs_iou.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return output_path


# ============================================================
# SUMMARY
# ============================================================

def save_summary(
    metrics,
):
    summary_path = (
        OUTPUT_DIR
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ViT-Base + YOLOv8 TEST EVALUATION\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Checkpoint: {CHECKPOINT}\n"
        )

        file.write(
            f"Test images: "
            f"{metrics['test_images']:,}\n"
        )

        file.write(
            f"Ground truth boxes: "
            f"{metrics['ground_truth_boxes']:,}\n"
        )

        file.write(
            f"Predictions after NMS: "
            f"{metrics['predictions_after_nms']:,}\n\n"
        )

        file.write(
            f"mAP@0.50: "
            f"{metrics['mAP@0.50']:.6f}\n"
        )

        file.write(
            f"mAP@0.50:0.95: "
            f"{metrics['mAP@0.50:0.95']:.6f}\n"
        )

        file.write(
            f"Precision@0.50: "
            f"{metrics['precision@0.50']:.6f}\n"
        )

        file.write(
            f"Recall / TPR@0.50: "
            f"{metrics['recall@0.50']:.6f}\n"
        )

        file.write(
            f"F1@0.50: "
            f"{metrics['f1@0.50']:.6f}\n\n"
        )

        file.write(
            f"TP@0.50: "
            f"{metrics['TP@0.50']}\n"
        )

        file.write(
            f"FP@0.50: "
            f"{metrics['FP@0.50']}\n"
        )

        file.write(
            f"FN@0.50: "
            f"{metrics['FN@0.50']}\n\n"
        )

        file.write(
            "METRICS BY IOU THRESHOLD\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            "IoU     AP          TP       FP       FN       TPR\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for index, threshold in enumerate(
            metrics["iou_thresholds"]
        ):
            file.write(
                f"{threshold:.2f}    "
                f"{metrics['ap_by_iou'][index]:.6f}    "
                f"{metrics['tp_by_iou'][index]:7d}  "
                f"{metrics['fp_by_iou'][index]:7d}  "
                f"{metrics['fn_by_iou'][index]:7d}  "
                f"{metrics['tpr_by_iou'][index]:.6f}\n"
            )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ViT-Base + YOLOv8 TEST EVALUATION")
    print("=" * 80)
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
        f"Network images: {NETWORK_IMAGES}"
    )

    print(
        f"Annotations: {ANNOTATIONS_CSV}"
    )

    print(
        f"Local dataset: {LOCAL_DATASET}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    if not NETWORK_IMAGES.exists():
        raise FileNotFoundError(
            f"Network images directory not found:\n"
            f"{NETWORK_IMAGES}"
        )

    if not ANNOTATIONS_CSV.exists():
        raise FileNotFoundError(
            f"Annotations CSV not found:\n"
            f"{ANNOTATIONS_CSV}"
        )

    if not LOCAL_DATASET.exists():
        raise FileNotFoundError(
            f"Local dataset not found:\n"
            f"{LOCAL_DATASET}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if VISUALIZATION_DIR.exists():
        print(
            "[INIT] Removing old visualizations..."
        )

        for path in VISUALIZATION_DIR.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(
                    path
                )

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # IMPORT TRAINING MODULE
    # --------------------------------------------------------

    training_module = (
        import_training_module()
    )

    VindrViTDataset = (
        training_module.VindrViTDataset
    )

    ViTYOLO = (
        training_module.ViTYOLO
    )

    collate_fn = (
        training_module.collate_fn
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print(
        "[INIT] Loading test dataset..."
    )

    test_dataset = VindrViTDataset(
        NETWORK_IMAGES,
        ANNOTATIONS_CSV,
        LOCAL_DATASET,
        "test",
        IMG_SIZE,
    )

    total_test_images = len(
        test_dataset
    )

    if MAX_EVAL_IMAGES is None:
        eval_count = total_test_images
    else:
        eval_count = min(
            MAX_EVAL_IMAGES,
            total_test_images,
        )

    print(
        f"[INIT] Test images: "
        f"{total_test_images:,}"
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
        drop_last=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "[INIT] Building ViT + YOLO model..."
    )

    model = ViTYOLO(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    model.detect.stride = torch.tensor(
        (
            8.0,
            16.0,
            32.0,
        ),
        dtype=torch.float32,
    )

    load_checkpoint(
        model
    )

    model = model.to(
        DEVICE
    )

    model.detect.stride = torch.tensor(
        (
            8.0,
            16.0,
            32.0,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    model.eval()

    print(
        "[INIT] Model ready."
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print(
        "[EVAL] Running inference..."
    )

    predictions = {}

    ground_truths = {}

    prediction_rows = []

    visualization_count = 0

    evaluated_images = 0

    amp_enabled = (
        DEVICE.type == "cuda"
    )

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="TEST",
            unit="image",
            total=eval_count,
            dynamic_ncols=True,
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
                enabled=amp_enabled,
            ):
                raw_output = model(
                    images
                )

                prediction_tensor = (
                    find_prediction_tensor(
                        raw_output
                    )
                )

            detections = (
                non_max_suppression(
                    prediction_tensor,
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

                gt = (
                    get_ground_truth_for_image(
                        targets,
                        batch_index,
                    )
                )

                predictions[
                    image_id
                ] = pred

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
                            "class": (
                                int(
                                    detection[5]
                                )
                                if len(detection) > 5
                                else 0
                            ),
                        }
                    )

                # ------------------------------------------------
                # VISUALIZATION
                # ------------------------------------------------

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
                            0.0,
                            1.0,
                        )
                        *
                        255.0
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

                    image_bgr = draw_boxes(
                        image_bgr,
                        gt,
                        (
                            0,
                            255,
                            0,
                        ),
                        "GT",
                    )

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

    print()
    print(
        "[EVAL] Inference complete."
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions_csv = (
        OUTPUT_DIR
        /
        "predictions.csv"
    )

    with open(
        predictions_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

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
            file,
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

    # --------------------------------------------------------
    # METRICS AT EVERY IOU
    # --------------------------------------------------------

    print()
    print(
        "[EVAL] Computing metrics for "
        "all IoU thresholds..."
    )

    ap_by_iou = []
    tp_by_iou = []
    fp_by_iou = []
    fn_by_iou = []
    tpr_by_iou = []

    for threshold in IOU_THRESHOLDS:

        (
            ap,
            tp,
            fp,
            fn,
        ) = collect_statistics(
            predictions,
            ground_truths,
            float(threshold),
        )

        tpr = (
            tp
            /
            max(
                tp + fn,
                1,
            )
        )

        ap_by_iou.append(
            float(ap)
        )

        tp_by_iou.append(
            int(tp)
        )

        fp_by_iou.append(
            int(fp)
        )

        fn_by_iou.append(
            int(fn)
        )

        tpr_by_iou.append(
            float(tpr)
        )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    threshold_to_index = {
        round(
            float(threshold),
            2,
        ): index
        for index, threshold
        in enumerate(
            IOU_THRESHOLDS
        )
    }

    index_050 = (
        threshold_to_index[0.50]
    )

    map50 = (
        ap_by_iou[index_050]
    )

    map50_95_values = [
        ap_by_iou[index]
        for index, threshold
        in enumerate(
            IOU_THRESHOLDS
        )
        if (
            float(threshold)
            >=
            0.50
        )
    ]

    map50_95 = float(
        np.mean(
            map50_95_values
        )
    )

    tp50 = (
        tp_by_iou[index_050]
    )

    fp50 = (
        fp_by_iou[index_050]
    )

    fn50 = (
        fn_by_iou[index_050]
    )

    precision50 = (
        tp50
        /
        max(
            tp50 + fp50,
            1,
        )
    )

    recall50 = (
        tp50
        /
        max(
            tp50 + fn50,
            1,
        )
    )

    f150 = (
        2.0
        *
        precision50
        *
        recall50
        /
        max(
            precision50
            + recall50,
            1e-12,
        )
    )

    total_ground_truth_boxes = sum(
        len(boxes)
        for boxes
        in ground_truths.values()
    )

    total_predictions = sum(
        len(preds)
        for preds
        in predictions.values()
    )

    # --------------------------------------------------------
    # METRICS JSON
    # --------------------------------------------------------

    metrics = {
        "checkpoint": str(
            CHECKPOINT
        ),

        "test_images": int(
            len(predictions)
        ),

        "ground_truth_boxes": int(
            total_ground_truth_boxes
        ),

        "predictions_after_nms": int(
            total_predictions
        ),

        "image_size": int(
            IMG_SIZE
        ),

        "confidence_threshold": float(
            CONF_THRESHOLD
        ),

        "nms_iou": float(
            NMS_IOU_THRESHOLD
        ),

        "iou_thresholds": [
            float(value)
            for value
            in IOU_THRESHOLDS
        ],

        "ap_by_iou": ap_by_iou,

        "tp_by_iou": tp_by_iou,

        "fp_by_iou": fp_by_iou,

        "fn_by_iou": fn_by_iou,

        "tpr_by_iou": tpr_by_iou,

        "mAP@0.50": float(
            map50
        ),

        "mAP@0.50:0.95": float(
            map50_95
        ),

        "precision@0.50": float(
            precision50
        ),

        "recall@0.50": float(
            recall50
        ),

        "tpr@0.50": float(
            recall50
        ),

        "f1@0.50": float(
            f150
        ),

        "TP@0.50": int(
            tp50
        ),

        "FP@0.50": int(
            fp50
        ),

        "FN@0.50": int(
            fn50
        ),
    }

    metrics_json = (
        OUTPUT_DIR
        /
        "metrics.json"
    )

    with open(
        metrics_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_path = save_summary(
        metrics
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    tpr_plot = save_tpr_plot(
        metrics
    )

    ap_plot = save_ap_plot(
        metrics
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(
        f"Test images: "
        f"{metrics['test_images']:,}"
    )

    print(
        f"Ground truth boxes: "
        f"{metrics['ground_truth_boxes']:,}"
    )

    print(
        f"Predictions after NMS: "
        f"{metrics['predictions_after_nms']:,}"
    )

    print()

    print(
        f"mAP@0.50: "
        f"{metrics['mAP@0.50']:.6f}"
    )

    print(
        f"mAP@0.50:0.95: "
        f"{metrics['mAP@0.50:0.95']:.6f}"
    )

    print(
        f"Precision@0.50: "
        f"{metrics['precision@0.50']:.6f}"
    )

    print(
        f"Recall / TPR@0.50: "
        f"{metrics['recall@0.50']:.6f}"
    )

    print(
        f"F1@0.50: "
        f"{metrics['f1@0.50']:.6f}"
    )

    print()

    print(
        "IoU      AP        TP      FP      FN      TPR"
    )

    print(
        "-" * 60
    )

    for index, threshold in enumerate(
        IOU_THRESHOLDS
    ):
        print(
            f"{threshold:.2f}    "
            f"{ap_by_iou[index]:.6f}    "
            f"{tp_by_iou[index]:4d}    "
            f"{fp_by_iou[index]:5d}    "
            f"{fn_by_iou[index]:4d}    "
            f"{tpr_by_iou[index]:.6f}"
        )

    print()
    print(
        f"[SAVE] Metrics: {metrics_json}"
    )

    print(
        f"[SAVE] Summary: {summary_path}"
    )

    print(
        f"[SAVE] TPR plot: {tpr_plot}"
    )

    print(
        f"[SAVE] AP plot: {ap_plot}"
    )

    print(
        f"[SAVE] Predictions: {predictions_csv}"
    )

    print(
        f"[SAVE] Visualizations: "
        f"{VISUALIZATION_DIR}"
    )

    print(
        f"[SAVE] Visualizations generated: "
        f"{visualization_count}"
    )

    print()
    print(
        "[DONE]"
    )


if __name__ == "__main__":
    main()
