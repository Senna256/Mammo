#!/usr/bin/env python3
"""Shared code for the six Without _dataleak experiments.

Experiments = {Swin-Tiny, ViT-Base} x {full, 4k, positives}. Everything except
the backbone and the TRAIN regime lives here, so preprocessing, splits,
validation, loss and optimisation are identical by construction:

    NETWORK DICOM -> breast_tissue + LINEAR windowing -> resize (keep aspect)
    -> detect_breast_roi crop -> resize + pad to 1024x1024 -> backbone (P3/P4/P5)
    -> YOLOv8 Detect -> v8DetectionLoss

TRAIN regimes (applied to TRAIN only; VAL and TEST are never filtered):
    full       all TRAIN images
    4k         all positive TRAIN images + random negatives up to TRAIN_MAX_IMAGES
    positives  only TRAIN images with at least one CSV box

Each training script calls main(MODEL_CLASS, REGIME, OUTPUT_DIR). The scripts
also re-export this module's names because Test/evaluate_all.py loads them.
"""
import argparse
import random
import time
from pathlib import Path
from types import SimpleNamespace

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
from ultralytics.utils.loss import v8DetectionLoss

from mammo_prep.io import load_dicom
from mammo_prep.windowing import preprocess_window

# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_IMAGES = Path("/home/enric/Datasets/Original/vindr/images")
ANNOTATIONS_CSV = Path("/home/enric/Datasets/Original/vindr/finding_annotations.csv")

NUM_CLASSES = 1
IMG_SIZE = 1024
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16
SEED = 42
VAL_FRACTION = 0.1
TRAIN_MAX_IMAGES = 4000  # "4k" regime
REGIMES = ("full", "4k", "positives")

DETECT_STRIDES = (8.0, 16.0, 32.0)

SWIN_MODEL_NAME = "swin_tiny_patch4_window7_224"
SWIN_CHANNELS = (192, 384, 768)

VIT_MODEL_NAME = "vit_base_patch16_224"
VIT_EMBED_DIM = 768
VIT_FEATURE_CHANNELS = (256, 512, 768)
VIT_BLOCKS = (3, 7, 11)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_dicom(dicom_path):
    """Read a DICOM mammogram and apply breast_tissue + LINEAR windowing to uint8."""
    image, ds = load_dicom(dicom_path)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    return preprocess_window(image, dicom_dataset=ds, method="breast_tissue", voi_func="LINEAR",
                             exclude_background=True, output_dtype=np.uint8)


def detect_breast_roi(image):
    """Bounding box (x1, y1, x2, y2) of the largest bright region; full image if none is found."""
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image, got {image.shape}")
    h, w = image.shape
    _, thresh = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

# YOLO boxes (cls, cx, cy, w, h normalised) across spaces: full image -> ROI -> padded image.

def transform_boxes_to_roi(labels, original_width, original_height, roi_x1, roi_y1, roi_x2, roi_y2):
    roi_w, roi_h = roi_x2 - roi_x1, roi_y2 - roi_y1
    transformed = []
    for cls_id, cx, cy, bw, bh in labels:
        x_center, y_center = cx * original_width, cy * original_height
        box_w, box_h = bw * original_width, bh * original_height
        x_min = np.clip(x_center - box_w / 2.0 - roi_x1, 0, roi_w)
        x_max = np.clip(x_center + box_w / 2.0 - roi_x1, 0, roi_w)
        y_min = np.clip(y_center - box_h / 2.0 - roi_y1, 0, roi_h)
        y_max = np.clip(y_center + box_h / 2.0 - roi_y1, 0, roi_h)
        new_w, new_h = x_max - x_min, y_max - y_min
        if new_w <= 1 or new_h <= 1:
            continue  # Box (almost) entirely outside the crop.
        transformed.append([cls_id, (x_min + x_max) / 2.0 / roi_w, (y_min + y_max) / 2.0 / roi_h,
                            new_w / roi_w, new_h / roi_h])
    if not transformed:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(transformed, dtype=np.float32)


def transform_boxes_to_padded_image(labels, crop_width, crop_height, scale, pad_x, pad_y, img_size):
    transformed = []
    for cls_id, cx, cy, bw, bh in labels:
        x_center = np.clip((cx * crop_width * scale + pad_x) / img_size, 0.0, 1.0)
        y_center = np.clip((cy * crop_height * scale + pad_y) / img_size, 0.0, 1.0)
        box_w = np.clip(bw * crop_width * scale / img_size, 0.0, 1.0)
        box_h = np.clip(bh * crop_height * scale / img_size, 0.0, 1.0)
        if box_w <= 0 or box_h <= 0:
            continue
        transformed.append([cls_id, x_center, y_center, box_w, box_h])
    if not transformed:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(transformed, dtype=np.float32)

# ============================================================
# DATASET RECORDS AND TRAIN REGIMES
# ============================================================

def build_dataset_records(annotations_csv=ANNOTATIONS_CSV, network_images=NETWORK_IMAGES,
                          val_fraction=VAL_FRACTION, seed=SEED):
    """Train/validation/test records from the CSV study split and the DICOMs on disk.

    Every DICOM of a study listed in the CSV is included; images without CSV
    boxes get an empty target (negatives). VAL = val_fraction of the TRAINING
    studies, chosen with a fixed seed. TEST = the official CSV test split.
    """
    annotations_csv, network_images = Path(annotations_csv), Path(network_images)
    if not annotations_csv.is_file():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")
    if not network_images.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {network_images}")
    df = pd.read_csv(annotations_csv)
    missing = {"study_id", "image_id", "split", "xmin", "ymin", "xmax", "ymax"} - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in CSV: {sorted(missing)}")
    df = df.copy()
    for column in ("study_id", "image_id"):
        df[column] = df[column].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    for column in ("xmin", "ymin", "xmax", "ymax"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["split"].isin(["training", "test"])].copy()
    if df.empty:
        raise RuntimeError("CSV must contain split='training' and/or split='test'.")
    split_counts = df.groupby("study_id")["split"].nunique()
    conflicting = split_counts[split_counts > 1].index.tolist()
    if conflicting:
        raise RuntimeError(f"Studies have conflicting split values: {conflicting[:10]}")
    study_splits = df.groupby("study_id")["split"].first().to_dict()

    box_df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    box_df = box_df.drop_duplicates(subset=["study_id", "image_id", "xmin", "ymin", "xmax", "ymax"])
    boxes_by_image = {}
    for row in box_df.itertuples(index=False):
        x1, y1, x2, y2 = float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)
        if np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1:
            boxes_by_image.setdefault((row.study_id, row.image_id), []).append([x1, y1, x2, y2])

    records_by_split = {"training": [], "test": []}
    scanned_studies = set()
    for study_id, split_name in study_splits.items():
        study_dir = network_images / study_id
        if not study_dir.is_dir():
            continue
        scanned_studies.add(study_id)
        for dicom_path in sorted(study_dir.iterdir()):
            if not dicom_path.is_file() or dicom_path.suffix.lower() not in {".dicom", ".dcm"}:
                continue
            records_by_split[split_name].append({
                "study_id": study_id,
                "image_id": dicom_path.stem,
                "dicom_path": dicom_path,
                "boxes": boxes_by_image.get((study_id, dicom_path.stem), []),
            })
    print(f"[DATA] Studies represented in CSV: {len(study_splits):,}", flush=True)
    print(f"[DATA] Study directories missing: {len(set(study_splits) - scanned_studies):,}", flush=True)
    print(f"[DATA] Images with CSV boxes: {len(boxes_by_image):,}", flush=True)

    training_records, test_records = records_by_split["training"], records_by_split["test"]
    if not training_records:
        raise RuntimeError("No training DICOMs found in studies listed in the CSV.")
    if not test_records:
        raise RuntimeError("No test DICOMs found in studies listed in the CSV.")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    training_studies = sorted({record["study_id"] for record in training_records})
    if len(training_studies) < 2:
        raise RuntimeError("At least two training studies are required for train/validation split.")
    rng = np.random.RandomState(seed)
    rng.shuffle(training_studies)
    n_val = min(max(1, round(len(training_studies) * val_fraction)), len(training_studies) - 1)
    val_studies = set(training_studies[:n_val])
    train_records = [r for r in training_records if r["study_id"] not in val_studies]
    val_records = [r for r in training_records if r["study_id"] in val_studies]

    for name, records in (("TRAIN", train_records), ("VAL", val_records), ("TEST", test_records)):
        positives = sum(bool(r["boxes"]) for r in records)
        print(f"[{name}] Images: {len(records):,} | Positive: {positives:,} | "
              f"Negative: {len(records) - positives:,}", flush=True)
    study_sets = [{r["study_id"] for r in records} for records in (train_records, val_records, test_records)]
    if any(not a.isdisjoint(b) for i, a in enumerate(study_sets) for b in study_sets[i + 1:]):
        raise RuntimeError("Study leakage detected between train, validation and test.")
    return train_records, val_records, test_records


def downsample_negative_records(records, max_images=TRAIN_MAX_IMAGES, seed=SEED):
    """Keep every positive TRAIN image and random negatives up to max_images in total."""
    records = list(records)
    positives = [r for r in records if r["boxes"]]
    negatives = [r for r in records if not r["boxes"]]
    print(f"[TRAIN DOWNSAMPLING] Before: {len(records):,} images | {len(positives):,} positive | "
          f"{len(negatives):,} negative", flush=True)
    if max_images <= 0:
        raise ValueError("TRAIN_MAX_IMAGES must be greater than zero.")
    if len(records) <= max_images:
        print(f"[TRAIN DOWNSAMPLING] Skipped: TRAIN has {len(records):,} images, "
              f"not more than the limit of {max_images:,}.", flush=True)
        return records
    if len(positives) > max_images:
        raise RuntimeError(f"TRAIN contains {len(positives):,} positive images, more than "
                           f"TRAIN_MAX_IMAGES={max_images:,}.")
    rng = random.Random(seed)
    selected_records = positives + rng.sample(negatives, k=max_images - len(positives))
    rng.shuffle(selected_records)
    print(f"[TRAIN DOWNSAMPLING] After: {len(selected_records):,} images | {len(positives):,} positive | "
          f"{max_images - len(positives):,} negative", flush=True)
    return selected_records


def filter_positive_records(records):
    """Keep only TRAIN images with at least one CSV box."""
    records = list(records)
    positives = [r for r in records if r["boxes"]]
    if not positives:
        raise RuntimeError("TRAIN contains no positive images.")
    print(f"[TRAIN POSITIVES] {len(records):,} -> {len(positives):,} images "
          f"(removed {len(records) - len(positives):,} negatives from TRAIN only)", flush=True)
    return positives


def apply_train_regime(train_records, regime):
    """Apply the experiment's TRAIN regime. VAL and TEST are never passed here."""
    if regime == "full":
        return train_records
    if regime == "4k":
        return downsample_negative_records(train_records, TRAIN_MAX_IMAGES, SEED)
    if regime == "positives":
        return filter_positive_records(train_records)
    raise ValueError(f"Unknown regime {regime!r}; expected one of {REGIMES}")

# ============================================================
# DATASET
# ============================================================

class VindrDataset(Dataset):
    """DICOM -> windowing -> breast crop -> 1024x1024 tensor, with YOLO targets in that space."""

    def __init__(self, records, img_size=IMG_SIZE):
        self.records = records
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_labels(boxes, original_width, original_height):
        """Absolute CSV xyxy boxes -> normalised YOLO (cls, cx, cy, w, h)."""
        labels = []
        for x1, y1, x2, y2 in boxes:
            x1, x2 = float(np.clip(x1, 0, original_width)), float(np.clip(x2, 0, original_width))
            y1, y2 = float(np.clip(y1, 0, original_height)), float(np.clip(y2, 0, original_height))
            if x2 <= x1 or y2 <= y1:
                continue
            labels.append([0.0, (x1 + x2) / 2.0 / original_width, (y1 + y2) / 2.0 / original_height,
                           (x2 - x1) / original_width, (y2 - y1) / original_height])
        return np.asarray(labels, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, index):
        record = self.records[index]
        image = preprocess_dicom(record["dicom_path"])
        original_h, original_w = image.shape[:2]
        labels = self._load_labels(record["boxes"], original_w, original_h)
        image, _ = resize_keep_aspect(image, self.img_size)
        resized_h, resized_w = image.shape[:2]
        x1, y1, x2, y2 = detect_breast_roi(image)
        crop = image[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]
        labels_roi = transform_boxes_to_roi(labels, resized_w, resized_h, x1, y1, x2, y2)
        final_image, scale, pad_x, pad_y = resize_and_pad(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), self.img_size)
        labels_final = transform_boxes_to_padded_image(labels_roi, crop_w, crop_h, scale, pad_x, pad_y,
                                                       self.img_size)
        final_image = np.transpose(final_image.astype(np.float32) / 255.0, (2, 0, 1))
        return torch.from_numpy(final_image).float(), torch.from_numpy(labels_final).float(), str(record["dicom_path"])


def collate_fn(batch):
    """Stack images and merge variable-length labels into the dict v8DetectionLoss expects."""
    images, batch_idx, classes, boxes, paths = [], [], [], [], []
    for i, (image, labels, path) in enumerate(batch):
        images.append(image)
        paths.append(path)
        if labels.numel() == 0:
            continue
        batch_idx.append(torch.full((labels.shape[0],), i, dtype=torch.long))
        classes.append(labels[:, 0:1])
        boxes.append(labels[:, 1:5])
    targets = {
        "batch_idx": torch.cat(batch_idx) if batch_idx else torch.zeros((0,), dtype=torch.long),
        "cls": torch.cat(classes) if classes else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32),
    }
    return torch.stack(images, dim=0), targets, paths

# ============================================================
# MODELS
# ============================================================

class SwinYOLO(nn.Module):
    """Swin-Tiny stages 1-3 (strides 8/16/32) -> YOLOv8 Detect."""

    description = "Swin-Tiny"

    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        print(f"[INIT] Loading {SWIN_MODEL_NAME}...", flush=True)
        self.backbone = timm.create_model(SWIN_MODEL_NAME, pretrained=pretrained, features_only=True,
                                          out_indices=(1, 2, 3), img_size=img_size)
        print("[INIT] Creating YOLOv8 Detect head...", flush=True)
        self.detect = Detect(nc=num_classes, ch=SWIN_CHANNELS)
        self.detect.stride = torch.tensor(DETECT_STRIDES, dtype=torch.float32)
        self.detect.bias_init()

    def forward(self, x):
        features = []
        for feat in self.backbone(x):
            if feat.ndim != 4:
                raise RuntimeError(f"Unexpected Swin feature shape: {feat.shape}")
            features.append(feat.permute(0, 3, 1, 2).contiguous())  # NHWC -> NCHW
        return self.detect(features)


class ViTBackbone(nn.Module):
    """ViT-Base at 1024 px (64x64 patch grid); returns the tokens of VIT_BLOCKS as feature maps."""

    def __init__(self, model_name=VIT_MODEL_NAME, pretrained=True, block_indices=VIT_BLOCKS, img_size=IMG_SIZE):
        super().__init__()
        print(f"[INIT] Loading {model_name}...", flush=True)
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.block_indices = tuple(block_indices)
        self.embed_dim = self.vit.embed_dim
        if self.embed_dim != VIT_EMBED_DIM:
            raise RuntimeError(f"Unexpected ViT embedding dimension: {self.embed_dim}. Expected {VIT_EMBED_DIM}.")
        patch_size = self.vit.patch_embed.patch_size
        self.patch_size = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
        self.feature_adapters = nn.ModuleList([nn.Conv2d(VIT_EMBED_DIM, channels, kernel_size=1)
                                               for channels in VIT_FEATURE_CHANNELS])
        self.p5_downsample = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)

    def _interpolate_pos_embed(self, pos_embed, height, width):
        num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)
        if pos_embed.shape[1] <= num_prefix_tokens:
            return pos_embed
        prefix, patch_pos = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
        old_size = int(patch_pos.shape[1] ** 0.5)
        if old_size * old_size != patch_pos.shape[1]:
            raise RuntimeError(f"Cannot infer square positional embedding grid from shape {patch_pos.shape}.")
        patch_pos = patch_pos.reshape(1, old_size, old_size, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(height, width), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, height * width, self.embed_dim)
        return torch.cat([prefix, patch_pos], dim=1)

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        if x.ndim == 4:
            _, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        elif x.ndim == 3:
            H = W = int(x.shape[1] ** 0.5)
            if H * W != x.shape[1]:
                raise RuntimeError(f"Patch tokens do not form a square grid: {x.shape[1]} tokens.")
        else:
            raise RuntimeError(f"Unexpected patch embedding shape: {x.shape}")
        cls_token = getattr(self.vit, "cls_token", None)
        if cls_token is not None:
            x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)
        reg_token = getattr(self.vit, "reg_token", None)
        if reg_token is not None:
            prefix_count = 1 if cls_token is not None else 0
            x = torch.cat((x[:, :prefix_count], reg_token.expand(B, -1, -1), x[:, prefix_count:]), dim=1)
        if getattr(self.vit, "pos_embed", None) is not None:
            pos_embed = self._interpolate_pos_embed(self.vit.pos_embed, H, W)
            if pos_embed.shape[1] != x.shape[1]:
                raise RuntimeError(f"ViT positional embedding shape {pos_embed.shape} does not match tokens {x.shape}.")
            x = x + pos_embed
        if hasattr(self.vit, "pos_drop"):
            x = self.vit.pos_drop(x)
        prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)
        outputs = []
        for block_index, block in enumerate(self.vit.blocks):
            x = block(x)
            if block_index in self.block_indices:
                tokens = x[:, prefix_tokens:]
                outputs.append(tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W).contiguous())
        if len(outputs) != 3:
            raise RuntimeError(f"ViT returned {len(outputs)} intermediate feature maps, expected 3.")
        return outputs


class ViTYOLO(nn.Module):
    """ViT-Base blocks 3/7/11 -> 1x1 adapters, resized to strides 8/16/32 -> YOLOv8 Detect."""

    description = "ViT-Base"

    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        print("[INIT] Creating ViT backbone...", flush=True)
        self.backbone = ViTBackbone(VIT_MODEL_NAME, pretrained=pretrained, block_indices=VIT_BLOCKS,
                                    img_size=img_size)
        print("[INIT] Creating YOLOv8 Detect head...", flush=True)
        self.detect = Detect(nc=num_classes, ch=VIT_FEATURE_CHANNELS)
        self.detect.stride = torch.tensor(DETECT_STRIDES, dtype=torch.float32)
        self.detect.bias_init()

    def forward(self, x):
        p3, p4, p5 = (adapter(feature) for adapter, feature in zip(self.backbone.feature_adapters, self.backbone(x)))
        height, width = x.shape[-2:]
        p3 = F.interpolate(p3, size=(height // 8, width // 8), mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=(height // 16, width // 16), mode="bilinear", align_corners=False)
        p5 = self.backbone.p5_downsample(p5)
        if p5.shape[-2:] != (height // 32, width // 32):
            raise RuntimeError(f"Unexpected P5 shape: {tuple(p5.shape[-2:])}")
        return self.detect([p3, p4, p5])


def build_model(model_cls, img_size=IMG_SIZE, pretrained=True):
    model = model_cls(img_size=img_size, num_classes=NUM_CLASSES, pretrained=pretrained).to(DEVICE)
    model.detect.stride = torch.tensor(DETECT_STRIDES, dtype=torch.float32, device=DEVICE)
    return model

# ============================================================
# LOSS
# ============================================================

class DetectionLossModel(nn.Module):
    """Minimal adapter so Ultralytics v8DetectionLoss can consume a custom model."""

    def __init__(self, detect):
        super().__init__()
        self.model = nn.ModuleList([detect])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


def create_loss(model):
    return v8DetectionLoss(DetectionLossModel(model.detect))


def compute_loss(criterion, predictions, targets):
    raw_loss, loss_items = criterion(predictions, targets)
    if not isinstance(raw_loss, torch.Tensor):
        raise RuntimeError(f"Ultralytics loss returned unexpected type: {type(raw_loss)}")
    total_loss = raw_loss.reshape(()) if raw_loss.numel() == 1 else raw_loss.sum()
    return total_loss, loss_items, raw_loss

# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    print(f"[RESUME] Loading checkpoint: {path}", flush=True)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"[RESUME] Starting from epoch {start_epoch}", flush=True)
    return start_epoch, checkpoint.get("best_val_loss", float("inf"))

# ============================================================
# TRAIN / VALIDATION LOOPS
# ============================================================

def train_one_epoch(model, criterion, loader, optimizer, scaler, epoch, accumulation_steps):
    """One epoch with AMP, gradient accumulation and gradient clipping."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = DEVICE.type == "cuda"
    progress = tqdm(loader, total=len(loader), desc=f"Epoch {epoch:03d} [TRAIN]", unit="batch",
                    dynamic_ncols=True, leave=True)
    for step, (images, targets, _) in enumerate(progress):
        images = images.to(DEVICE, non_blocking=True)
        targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            total_loss, _, _ = compute_loss(criterion, model(images), targets)
            loss_backward = total_loss / accumulation_steps
        scaler.scale(loss_backward).backward()
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += total_loss.item()
        progress.set_postfix(loss=f"{running_loss / (step + 1):.5f}", batch_loss=f"{total_loss.item():.5f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, criterion, loader):
    """Mean detection loss on VAL, deterministic for every backbone.

    - Backbone in eval(): disables stochastic layers. Swin-Tiny has DropPath up
      to 0.1, which in train() mode made its VAL loss random; ViT-Base has none,
      so for ViT this is identical to keeping it in train().
    - Detect head in train(): in eval() it returns decoded boxes instead of the
      raw per-level outputs that v8DetectionLoss expects.
    - BatchNorm in eval(): VAL batches do not update running statistics.
    no_grad() prevents any weight update.
    """
    model.train()
    model.backbone.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    running_loss = 0.0
    amp_enabled = DEVICE.type == "cuda"
    progress = tqdm(loader, total=len(loader), desc="Validation", unit="batch", dynamic_ncols=True, leave=True)
    for step, (images, targets, _) in enumerate(progress):
        images = images.to(DEVICE, non_blocking=True)
        targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            total_loss, _, _ = compute_loss(criterion, model(images), targets)
        running_loss += total_loss.item()
        progress.set_postfix(loss=f"{running_loss / (step + 1):.5f}")
    return running_loss / len(loader)

# ============================================================
# INTEGRATION TEST
# ============================================================

def inspect_prediction(obj, prefix=""):
    """Recursively print the structure and tensor shapes of a model output."""
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}dict keys={list(obj.keys())}")
        for key, value in obj.items():
            print(f"{prefix}  [{key}]")
            inspect_prediction(value, prefix + "    ")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} length={len(obj)}")
        for i, value in enumerate(obj):
            print(f"{prefix}  [{i}]")
            inspect_prediction(value, prefix + "    ")
    else:
        print(f"{prefix}type={type(obj)} value={obj}")


def run_test(model_cls, regime):
    """End-to-end smoke test: one positive TRAIN image -> model -> loss -> backward, with gradient checks."""
    print("=" * 70)
    print(f"{model_cls.description} + YOLOv8 INTEGRATION TEST ({regime})")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    train_records, _, _ = build_dataset_records()
    train_records = apply_train_regime(train_records, regime)
    positive = next(r for r in train_records if r["boxes"])
    images, targets, paths = collate_fn([VindrDataset([positive])[0]])
    print(f"Image tensor: {tuple(images.shape)} | Targets: {targets['bboxes'].shape[0]} boxes | DICOM: {paths[0]}")

    model = build_model(model_cls)
    criterion = create_loss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    amp_enabled = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    images = images.to(DEVICE)
    targets = {k: v.to(DEVICE) for k, v in targets.items()}
    model.train()
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        predictions = model(images)
        print("[TEST] YOLO Detect output:")
        inspect_prediction(predictions)
        total_loss, _, raw_loss = compute_loss(criterion, predictions, targets)
    print(f"Raw loss: {raw_loss.detach().float().cpu().numpy()} | Total loss: {total_loss.item():.6f}")
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    backbone_gradients = [p.grad.abs().mean().item() for p in model.backbone.parameters() if p.grad is not None]
    if not backbone_gradients:
        raise RuntimeError("No gradients found in the backbone. The backbone is not being trained.")
    print(f"Gradient norm: {float(grad_norm):.6f} | Backbone parameters with gradients: "
          f"{len(backbone_gradients)} | Mean |grad|: {np.mean(backbone_gradients):.8e}")
    print("TEST PASSED")

# ============================================================
# TRAINING
# ============================================================

def run_training(model_cls, regime, output_dir, resume=None, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 workers=NUM_WORKERS, img_size=IMG_SIZE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"{model_cls.description.upper()} + YOLOv8 TRAINING · TRAIN regime: {regime}")
    print("=" * 70)
    print("[INIT] Configuration")
    print(f"  Network images: {NETWORK_IMAGES}")
    print(f"  Annotations: {ANNOTATIONS_CSV}")
    print(f"  Output: {output_dir}")
    print(f"  Image size: {img_size} × {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {batch_size * ACCUMULATION_STEPS}")
    print(f"  Epochs: {epochs}")
    print(f"  Workers: {workers}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("[INIT] Pipeline: DICOM -> breast_tissue + LINEAR -> detect_breast_roi -> resize + pad "
          f"{img_size} -> {model_cls.description} P3/P4/P5 -> YOLOv8 Detect -> v8DetectionLoss", flush=True)

    train_records, val_records, test_records = build_dataset_records()
    train_records = apply_train_regime(train_records, regime)
    train_dataset = VindrDataset(train_records, img_size)
    val_dataset = VindrDataset(val_records, img_size)
    print(f"[INIT] Train images ({regime}): {len(train_dataset):,}", flush=True)
    print(f"[INIT] Validation images (unfiltered): {len(val_dataset):,}", flush=True)
    print(f"[INIT] Test images reserved for evaluation: {len(test_records):,}", flush=True)
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=workers > 0, collate_fn=collate_fn, drop_last=False)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(model_cls, img_size)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INIT] Parameters: {total_params / 1e6:.2f} M total | {trainable_params / 1e6:.2f} M trainable",
          flush=True)

    criterion = create_loss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LEARNING_RATE * 0.01)
    amp_enabled = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"[INIT] AdamW + cosine schedule | AMP enabled: {amp_enabled}", flush=True)

    start_epoch, best_val_loss = 1, float("inf")
    if resume is not None:
        start_epoch, best_val_loss = load_checkpoint(resume, model, optimizer, scheduler, scaler)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        print("=" * 70)
        print(f"EPOCH {epoch}/{epochs} | Learning rate: {optimizer.param_groups[0]['lr']:.8f}", flush=True)
        print("=" * 70)
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scaler, epoch, ACCUMULATION_STEPS)
        val_loss = validate(model, criterion, val_loader)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_val_loss)
            print("[CHECKPOINT] New best model saved", flush=True)
        print("-" * 70)
        print(f"EPOCH {epoch}/{epochs} SUMMARY")
        print(f"Train loss: {train_loss:.6f}")
        print(f"Val loss:   {val_loss:.6f}")
        print(f"Best val:   {best_val_loss:.6f}")
        print(f"Time:       {epoch_time / 60:.2f} min")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.8f}")
        print("-" * 70, flush=True)

    print("=" * 70)
    print(f"TRAINING FINISHED | Best validation loss: {best_val_loss:.6f} | Output: {output_dir}")
    print("=" * 70)

# ============================================================
# COMMAND LINE
# ============================================================

def main(model_cls, regime, output_dir):
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {REGIMES}")
    parser = argparse.ArgumentParser(
        description=f"Train {model_cls.description} + YOLOv8 on VinDr mammography (TRAIN regime: {regime}).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="Run one real forward/loss/backward test.")
    mode.add_argument("--train", action="store_true", help="Train the model.")
    mode.add_argument("--resume", type=str, metavar="CHECKPOINT", help="Resume training from a checkpoint.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    set_seed(SEED)
    if args.test:
        run_test(model_cls, regime)
        return
    run_training(model_cls, regime, output_dir, resume=args.resume, epochs=args.epochs, batch_size=args.batch,
                 workers=args.workers, img_size=args.img_size)
