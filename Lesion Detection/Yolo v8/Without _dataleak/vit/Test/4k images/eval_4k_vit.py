#!/usr/bin/env python3
"""ViT-Base + YOLOv8 lesion detection on VinDr-Mammo, TRAIN regime 4k: TEST evaluation.

Evaluates the checkpoint of vit/Train/4k images/train_4k_vit.py on the official CSV TEST
split (4,000 images, 452 lesion boxes).

Pipeline: TEST DICOM -> breast_tissue windowing -> breast crop -> 1024x1024
-> ViT-Base (P3/P4/P5) -> YOLOv8 Detect -> NMS -> predictions.csv -> metrics.

The six evaluation scripts ({Swin, ViT} x {full, 4k, positives}) are identical
except for the model and the experiment paths. Preprocessing, splits, dataset
and model are copied from the training script, so the model is evaluated with
the preprocessing it was trained with. Keep them in sync.

Why NMS is applied
------------------
The YOLOv8 Detect head scores every position of its three grids (21,504 per
1024x1024 image), so a single lesion produces many overlapping boxes. Without
NMS every duplicate of a correct box counts as a false positive, which makes
FPPI, precision and AP meaningless. Settings follow the Ultralytics validation
defaults: conf 0.001, NMS IoU 0.70, max 300 detections per image.

Ground-truth denominator
------------------------
TPR uses the number of CSV boxes in TEST (identical for all experiments). Boxes
removed by the breast crop count as missed lesions. Matching is done in the
model's 1024x1024 input space.

Outputs (Test/outputs/vit_4k/):
    predictions.csv    one row per detection after NMS
    ground_truth.csv   TEST boxes after preprocessing
    meta.json          checkpoint, epoch and counts
    metrics_by_iou.csv TPR at fixed FPPI, TPR with all detections and AP, per IoU
    metrics.json       summary
Test/compare_all.py compares the six experiments.

Usage:
    python eval_4k_vit.py                         # inference + metrics with best.pt
    python eval_4k_vit.py --checkpoint last.pt
    python eval_4k_vit.py --metrics-only          # recompute metrics from predictions.csv
    python eval_4k_vit.py --limit 20 --output-dir /tmp/check   # smoke test
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics.nn.modules import Detect
from ultralytics.utils.nms import non_max_suppression

from mammo_prep.io import load_dicom
from mammo_prep.windowing import preprocess_window

# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_IMAGES = Path("/home/enric/Datasets/Original/vindr/images")
ANNOTATIONS_CSV = Path("/home/enric/Datasets/Original/vindr/finding_annotations.csv")
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]  # Without _dataleak/
TRAIN_SCRIPT = EXPERIMENTS_ROOT / "vit" / "Train" / "4k images" / "train_4k_vit.py"
CHECKPOINT_DIR = TRAIN_SCRIPT.parent / "outputs"
OUTPUT_DIR = EXPERIMENTS_ROOT / "Test" / "outputs" / "vit_4k"

EXPERIMENT = "vit_4k"
EXPERIMENT_LABEL = "ViT · 4k"

NUM_CLASSES = 1
IMG_SIZE = 1024
BATCH_SIZE = 8
NUM_WORKERS = 16
SEED = 42
VAL_FRACTION = 0.1  # of the CSV training studies (only to rebuild the same splits)
REGIME = "4k"
DETECT_STRIDES = (8.0, 16.0, 32.0)  # P3, P4, P5

MODEL_DESCRIPTION = "ViT-Base"
MODEL_NAME = "vit_base_patch16_224"
VIT_EMBED_DIM = 768
VIT_BLOCKS = (3, 7, 11)  # transformer blocks whose tokens become P3 / P4 / P5
VIT_FEATURE_CHANNELS = (256, 512, 768)  # channels of P3 / P4 / P5 after the 1x1 adapters

# Detections: Ultralytics validation defaults.
CONF_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 300

# Metrics.
IOU_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55,
                  0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MAP_THRESHOLDS = [t for t in IOU_THRESHOLDS if t >= 0.50]  # 0.50:0.05:0.95
FPPI_POINTS = [0.25, 0.5, 1.0, 2.0, 4.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_dicom(dicom_path):
    """Read a DICOM and window it to a uint8 image with bright tissue."""
    image, ds = load_dicom(dicom_path)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    return preprocess_window(image, dicom_dataset=ds, method="breast_tissue", voi_func="LINEAR",
                             exclude_background=True, output_dtype=np.uint8)


def detect_breast_roi(image):
    """Bounding box (x1, y1, x2, y2) of the breast: the largest bright region.

    Falls back to the whole image when no reasonable region is found.
    """
    h, w = image.shape
    _, mask = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bw < 0.05 * w or bh < 0.05 * h:
        return 0, 0, w, h
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def resize_keep_aspect(image, max_size):
    h, w = image.shape[:2]
    scale = min(max_size / float(w), max_size / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def resize_and_pad(image, img_size):
    """Fit the image inside an img_size square, centred on a black canvas."""
    h, w = image.shape[:2]
    scale = min(img_size / float(w), img_size / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    pad_x, pad_y = (img_size - new_w) // 2, (img_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y

# ============================================================
# BOX TRANSFORMATION
# ============================================================

# Boxes are YOLO labels (cls, cx, cy, w, h), normalised to the image they live in.

def boxes_to_yolo(boxes, width, height):
    """Absolute CSV boxes (x1, y1, x2, y2) -> YOLO labels normalised to the image."""
    labels = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))
        y1, y2 = float(np.clip(y1, 0, height)), float(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append([0.0, (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height,
                       (x2 - x1) / width, (y2 - y1) / height])
    return np.asarray(labels, dtype=np.float32).reshape(-1, 5)


def transform_boxes_to_roi(labels, width, height, roi_x1, roi_y1, roi_x2, roi_y2):
    """Labels of the full image -> labels of the crop. Boxes left with <= 1 px are dropped."""
    roi_w, roi_h = roi_x2 - roi_x1, roi_y2 - roi_y1
    transformed = []
    for cls_id, cx, cy, bw, bh in labels:
        x_center, y_center = cx * width, cy * height
        box_w, box_h = bw * width, bh * height
        x_min = np.clip(x_center - box_w / 2.0 - roi_x1, 0, roi_w)
        x_max = np.clip(x_center + box_w / 2.0 - roi_x1, 0, roi_w)
        y_min = np.clip(y_center - box_h / 2.0 - roi_y1, 0, roi_h)
        y_max = np.clip(y_center + box_h / 2.0 - roi_y1, 0, roi_h)
        new_w, new_h = x_max - x_min, y_max - y_min
        if new_w <= 1 or new_h <= 1:
            continue
        transformed.append([cls_id, (x_min + x_max) / 2.0 / roi_w, (y_min + y_max) / 2.0 / roi_h,
                            new_w / roi_w, new_h / roi_h])
    return np.asarray(transformed, dtype=np.float32).reshape(-1, 5)


def transform_boxes_to_padded_image(labels, crop_width, crop_height, scale, pad_x, pad_y, img_size):
    """Labels of the crop -> labels of the padded img_size square."""
    transformed = []
    for cls_id, cx, cy, bw, bh in labels:
        x_center = np.clip((cx * crop_width * scale + pad_x) / img_size, 0.0, 1.0)
        y_center = np.clip((cy * crop_height * scale + pad_y) / img_size, 0.0, 1.0)
        box_w = np.clip(bw * crop_width * scale / img_size, 0.0, 1.0)
        box_h = np.clip(bh * crop_height * scale / img_size, 0.0, 1.0)
        if box_w <= 0 or box_h <= 0:
            continue
        transformed.append([cls_id, x_center, y_center, box_w, box_h])
    return np.asarray(transformed, dtype=np.float32).reshape(-1, 5)

# ============================================================
# DATA SPLITS
# ============================================================

def read_annotations(annotations_csv):
    """Split of every study ('training' / 'test') and the valid boxes of every image."""
    df = pd.read_csv(annotations_csv)
    df["study_id"] = df["study_id"].astype(str).str.strip()
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    for column in ("xmin", "ymin", "xmax", "ymax"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["split"].isin(["training", "test"])]
    if (df.groupby("study_id")["split"].nunique() > 1).any():
        raise RuntimeError("Some studies appear in both CSV splits.")
    study_splits = df.groupby("study_id")["split"].first().to_dict()

    boxes_by_image = {}
    box_df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    box_df = box_df.drop_duplicates(subset=["study_id", "image_id", "xmin", "ymin", "xmax", "ymax"])
    for row in box_df.itertuples(index=False):
        x1, y1, x2, y2 = float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)
        if np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1:
            boxes_by_image.setdefault((row.study_id, row.image_id), []).append([x1, y1, x2, y2])
    return study_splits, boxes_by_image


def build_dataset_records(annotations_csv=ANNOTATIONS_CSV, network_images=NETWORK_IMAGES,
                          val_fraction=VAL_FRACTION, seed=SEED):
    """TRAIN / VAL / TEST records: one per DICOM of every study listed in the CSV.

    Images without CSV boxes are negatives (empty target). VAL is a random
    val_fraction of the CSV training STUDIES, so no study is split across sets.
    """
    study_splits, boxes_by_image = read_annotations(annotations_csv)
    records = {"training": [], "test": []}
    for study_id, split in study_splits.items():
        study_dir = Path(network_images) / study_id
        if not study_dir.is_dir():
            continue
        for dicom_path in sorted(study_dir.iterdir()):
            if not dicom_path.is_file() or dicom_path.suffix.lower() not in {".dicom", ".dcm"}:
                continue
            records[split].append({
                "study_id": study_id,
                "image_id": dicom_path.stem,
                "dicom_path": dicom_path,
                "boxes": boxes_by_image.get((study_id, dicom_path.stem), []),
            })

    training_studies = sorted({r["study_id"] for r in records["training"]})
    np.random.RandomState(seed).shuffle(training_studies)
    n_val = min(max(1, round(len(training_studies) * val_fraction)), len(training_studies) - 1)
    val_studies = set(training_studies[:n_val])
    train_records = [r for r in records["training"] if r["study_id"] not in val_studies]
    val_records = [r for r in records["training"] if r["study_id"] in val_studies]
    test_records = records["test"]

    for name, split_records in (("TRAIN", train_records), ("VAL", val_records), ("TEST", test_records)):
        positives = sum(bool(r["boxes"]) for r in split_records)
        print(f"[{name}] Images: {len(split_records):,} | Positive: {positives:,} | "
              f"Negative: {len(split_records) - positives:,}", flush=True)
    return train_records, val_records, test_records


# ============================================================
# DATASET
# ============================================================

class VindrDataset(Dataset):
    """Returns (image tensor 3 x 1024 x 1024 in [0, 1], YOLO labels, DICOM path)."""

    def __init__(self, records, img_size=IMG_SIZE):
        self.records = records
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = preprocess_dicom(record["dicom_path"])
        labels = boxes_to_yolo(record["boxes"], image.shape[1], image.shape[0])

        image, _ = resize_keep_aspect(image, self.img_size)
        resized_h, resized_w = image.shape
        x1, y1, x2, y2 = detect_breast_roi(image)
        crop = image[y1:y2, x1:x2]
        labels = transform_boxes_to_roi(labels, resized_w, resized_h, x1, y1, x2, y2)

        final_image, scale, pad_x, pad_y = resize_and_pad(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), self.img_size)
        labels = transform_boxes_to_padded_image(labels, crop.shape[1], crop.shape[0], scale, pad_x, pad_y,
                                                 self.img_size)
        final_image = np.transpose(final_image.astype(np.float32) / 255.0, (2, 0, 1))
        return torch.from_numpy(final_image).float(), torch.from_numpy(labels).float(), str(record["dicom_path"])


def collate_fn(batch):
    """Stack the images and merge the labels into the dict v8DetectionLoss expects."""
    images, batch_idx, classes, boxes, paths = [], [], [], [], []
    for i, (image, labels, path) in enumerate(batch):
        images.append(image)
        paths.append(path)
        batch_idx.append(torch.full((labels.shape[0],), i, dtype=torch.long))
        classes.append(labels[:, 0:1])
        boxes.append(labels[:, 1:5])
    targets = {"batch_idx": torch.cat(batch_idx), "cls": torch.cat(classes), "bboxes": torch.cat(boxes)}
    return torch.stack(images, dim=0), targets, paths

# ============================================================
# MODEL
# ============================================================

class ViTBackbone(nn.Module):
    """ViT-Base run at 1024 px: a single 64 x 64 grid of 16 px patches.

    Returns the patch tokens of VIT_BLOCKS as three (B, 768, 64, 64) maps.
    """

    def __init__(self, pretrained=True, img_size=IMG_SIZE):
        super().__init__()
        self.vit = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.embed_dim = self.vit.embed_dim
        self.feature_adapters = nn.ModuleList([nn.Conv2d(VIT_EMBED_DIM, channels, kernel_size=1)
                                               for channels in VIT_FEATURE_CHANNELS])
        self.p5_downsample = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)

    def _interpolate_pos_embed(self, height, width):
        """Resize the pretrained positional embeddings (14 x 14 grid) to height x width."""
        pos_embed = self.vit.pos_embed
        cls_pos, patch_pos = pos_embed[:, :1], pos_embed[:, 1:]
        old_size = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(height, width), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, height * width, self.embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        if x.ndim == 4:  # some timm versions return (B, H, W, C)
            _, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        else:
            H = W = int(x.shape[1] ** 0.5)
        x = torch.cat((self.vit.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self._interpolate_pos_embed(H, W)
        x = self.vit.pos_drop(x)
        outputs = []
        for block_index, block in enumerate(self.vit.blocks):
            x = block(x)
            if block_index in VIT_BLOCKS:
                patch_tokens = x[:, 1:]  # drop the CLS token
                outputs.append(patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W).contiguous())
        return outputs


class ViTYOLO(nn.Module):
    """ViT-Base blocks 3 / 7 / 11, resized to strides 8 / 16 / 32, feed the YOLOv8 Detect head."""

    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = ViTBackbone(pretrained=pretrained, img_size=img_size)
        self.detect = Detect(nc=num_classes, ch=VIT_FEATURE_CHANNELS)
        self.detect.stride = torch.tensor(DETECT_STRIDES, dtype=torch.float32)
        self.detect.bias_init()

    def forward(self, x):
        # All three ViT maps are at stride 16; build the stride 8 / 16 / 32 pyramid YOLO expects.
        p3, p4, p5 = (adapter(feature) for adapter, feature in zip(self.backbone.feature_adapters, self.backbone(x)))
        height, width = x.shape[-2:]
        p3 = F.interpolate(p3, size=(height // 8, width // 8), mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=(height // 16, width // 16), mode="bilinear", align_corners=False)
        p5 = self.backbone.p5_downsample(p5)
        return self.detect([p3, p4, p5])


MODEL_CLASS = ViTYOLO


def build_model(img_size=IMG_SIZE, pretrained=True):
    model = MODEL_CLASS(img_size=img_size, num_classes=NUM_CLASSES, pretrained=pretrained).to(DEVICE)
    model.detect.stride = torch.tensor(DETECT_STRIDES, dtype=torch.float32, device=DEVICE)
    return model

# ============================================================
# INFERENCE
# ============================================================

def image_key(dicom_path):
    """Image identifier used in the output CSVs: "study_id/image_id"."""
    path = Path(dicom_path)
    return f"{path.parent.name}/{path.stem}"


def load_model(checkpoint_path):
    """The model in eval() mode with the checkpoint weights, and the checkpoint itself."""
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def run_inference(checkpoint_name, batch_size, workers, limit, output_dir):
    """Predictions after NMS and ground truth of every TEST image, both in the 1024x1024 input space."""
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test_records = build_dataset_records()
    if limit:
        # Smoke-test subset with both positives and negatives.
        positives = [r for r in test_records if r["boxes"]]
        negatives = [r for r in test_records if not r["boxes"]]
        test_records = positives[: limit // 2] + negatives[: limit - limit // 2]
    csv_boxes = sum(len(r["boxes"]) for r in test_records)

    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    print(f"[{EXPERIMENT}] Checkpoint: {checkpoint_path}", flush=True)
    print(f"[{EXPERIMENT}] TEST images: {len(test_records):,} | CSV boxes: {csv_boxes:,}", flush=True)
    model, checkpoint = load_model(checkpoint_path)

    loader = DataLoader(VindrDataset(test_records, IMG_SIZE), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)

    prediction_rows, gt_rows = [], []
    start = time.time()
    for images, targets, paths in tqdm(loader, desc=f"[{EXPERIMENT}] TEST", unit="batch", dynamic_ncols=True):
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            output = model(images.to(DEVICE, non_blocking=True))
        # Eval-mode Detect returns (decoded predictions, raw head outputs).
        decoded = output[0] if isinstance(output, (tuple, list)) else output
        detections = non_max_suppression(decoded.float(), conf_thres=CONF_THRESHOLD,
                                         iou_thres=NMS_IOU_THRESHOLD, max_det=MAX_DETECTIONS)
        for i, det in enumerate(detections):
            image_id = image_key(paths[i])
            for x1, y1, x2, y2, conf, _ in det.cpu().numpy().tolist():
                prediction_rows.append((image_id, conf, x1, y1, x2, y2))

        # Labels are normalised (cx, cy, w, h); back to pixels of the input image.
        boxes = targets["bboxes"].numpy() * IMG_SIZE
        for batch_index, (cx, cy, w, h) in zip(targets["batch_idx"].numpy(), boxes):
            gt_rows.append((image_key(paths[int(batch_index)]), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    elapsed = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows, columns=["image_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(
        output_dir / "predictions.csv", index=False)
    pd.DataFrame(gt_rows, columns=["image_id", "x1", "y1", "x2", "y2"]).to_csv(
        output_dir / "ground_truth.csv", index=False)
    meta = {
        "experiment": EXPERIMENT,
        "training_script": str(TRAIN_SCRIPT),
        "evaluation_script": str(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        "img_size": IMG_SIZE,
        "test_images": len(test_records),
        "csv_gt_boxes": csv_boxes,
        "gt_boxes_after_preprocessing": len(gt_rows),
        "predictions_after_nms": len(prediction_rows),
        "inference_seconds": round(elapsed, 1),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{EXPERIMENT}] Done in {elapsed / 60:.1f} min | predictions after NMS: {len(prediction_rows):,} | "
          f"GT boxes kept by preprocessing: {len(gt_rows)}/{csv_boxes}", flush=True)

# ============================================================
# METRICS
# ============================================================

def iou_matrix(boxes_a, boxes_b):
    """IoU of every box in boxes_a (N, 4) with every box in boxes_b (M, 4); boxes are (x1, y1, x2, y2)."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


def match_detections(predictions, ground_truth, iou_threshold):
    """Greedy matching by descending confidence; each GT can be matched once.

    Returns the TP flag of every detection, sorted by descending confidence
    over the whole TEST set.
    """
    confidence = predictions["confidence"].to_numpy()
    is_tp = np.zeros(len(predictions), dtype=bool)
    gt_by_image = {img: g[["x1", "y1", "x2", "y2"]].to_numpy() for img, g in ground_truth.groupby("image_id")}
    for image_id, group in predictions.groupby("image_id"):
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None:
            continue  # Images without GT: every detection is a FP.
        order = group.index.to_numpy()[np.argsort(-group["confidence"].to_numpy(), kind="stable")]
        ious = iou_matrix(predictions.loc[order, ["x1", "y1", "x2", "y2"]].to_numpy(), gt_boxes)
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for row, det_index in enumerate(order):
            candidates = np.where(matched, -1.0, ious[row])
            best = int(np.argmax(candidates))
            if candidates[best] >= iou_threshold:
                matched[best] = True
                is_tp[det_index] = True
    global_order = np.argsort(-confidence, kind="stable")
    return is_tp[global_order]


def average_precision(cum_tp, cum_fp, total_gt):
    """All-point interpolated AP."""
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall_points = np.concatenate([[0.0], recall, [1.0]])
    precision_points = np.concatenate([[1.0], precision, [0.0]])
    precision_points = np.maximum.accumulate(precision_points[::-1])[::-1]
    return float(np.sum(np.diff(recall_points) * precision_points[1:]))


def tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, fppi):
    """TPR when detections are kept down to the confidence giving `fppi` FPs per image."""
    kept = np.searchsorted(cum_fp, fppi * n_images, side="right")
    return float(cum_tp[kept - 1] / total_gt) if kept > 0 else 0.0


def compute_metrics(output_dir):
    """Metrics per IoU threshold (metrics_by_iou.csv) and a summary (metrics.json)."""
    meta = json.loads((output_dir / "meta.json").read_text())
    predictions = pd.read_csv(output_dir / "predictions.csv")
    ground_truth = pd.read_csv(output_dir / "ground_truth.csv")
    total_gt, n_images = meta["csv_gt_boxes"], meta["test_images"]

    rows = []
    for iou_threshold in IOU_THRESHOLDS:
        tp = match_detections(predictions, ground_truth, iou_threshold)
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(~tp)
        row = {
            "experiment": EXPERIMENT,
            "iou": iou_threshold,
            "AP": average_precision(cum_tp, cum_fp, total_gt),
            "TPR_all_detections": float(cum_tp[-1] / total_gt) if len(tp) else 0.0,
            "TP_all_detections": int(cum_tp[-1]) if len(tp) else 0,
            "FP_all_detections": int(cum_fp[-1]) if len(tp) else 0,
        }
        for fppi in FPPI_POINTS:
            row[f"TPR@{fppi:g}FPPI"] = tpr_at_fppi(cum_tp, cum_fp, total_gt, n_images, fppi) if len(tp) else 0.0
        rows.append(row)
    per_iou = pd.DataFrame(rows)
    per_iou.to_csv(output_dir / "metrics_by_iou.csv", index=False)

    at = per_iou.set_index("iou")
    summary = {
        "experiment": EXPERIMENT,
        "label": EXPERIMENT_LABEL,
        "checkpoint_epoch": meta["checkpoint_epoch"],
        "test_images": n_images,
        "csv_gt_boxes": total_gt,
        "gt_lost_by_preprocessing": total_gt - meta["gt_boxes_after_preprocessing"],
        "predictions_after_nms": meta["predictions_after_nms"],
        "AP@0.10": at.loc[0.10, "AP"],
        "AP@0.30": at.loc[0.30, "AP"],
        "AP@0.50": at.loc[0.50, "AP"],
        "mAP@0.50:0.95": float(at.loc[MAP_THRESHOLDS, "AP"].mean()),
    }
    for iou_threshold in (0.30, 0.50):
        for fppi in (0.5, 1.0, 2.0):
            summary[f"TPR@IoU{iou_threshold:.2f}_{fppi:g}FPPI"] = at.loc[iou_threshold, f"TPR@{fppi:g}FPPI"]
        summary[f"TPR@IoU{iou_threshold:.2f}_all_detections"] = at.loc[iou_threshold, "TPR_all_detections"]
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Evaluate {MODEL_DESCRIPTION} + YOLOv8 (TRAIN regime: {REGIME}) on the CSV TEST split.")
    parser.add_argument("--checkpoint", choices=["best.pt", "last.pt"], default="best.pt")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only N TEST images (smoke test).")
    parser.add_argument("--metrics-only", action="store_true", help="Recompute the metrics from predictions.csv.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    if not args.metrics_only:
        run_inference(args.checkpoint, args.batch, args.workers, args.limit, args.output_dir)
    summary = compute_metrics(args.output_dir)

    print()
    for name in ("checkpoint_epoch", "gt_lost_by_preprocessing", "AP@0.50", "mAP@0.50:0.95",
                 "TPR@IoU0.30_1FPPI", "TPR@IoU0.50_1FPPI", "TPR@IoU0.50_all_detections"):
        value = summary[name]
        print(f"{name:<28} {value:.4f}" if isinstance(value, float) else f"{name:<28} {value}")
    print(f"[SAVED] {args.output_dir}")


if __name__ == "__main__":
    main()
