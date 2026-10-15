#!/usr/bin/env python3
import argparse
import random
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss

# ============================================================
# CONFIGURATION
# ============================================================

# Centralized project paths, model settings, and training hyperparameters.
NETWORK_IMAGES = Path(
    "/home/enric/Datasets/Original/vindr/images"
)
ANNOTATIONS_CSV = Path(
    "/home/enric/Datasets/Original/vindr/finding_annotations.csv"
)
OUTPUT_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/Without _dataleak/swin/Train/positives/outputs"
)
MODEL_NAME = "swin_tiny_patch4_window7_224"
NUM_CLASSES = 1
IMG_SIZE = 1024
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16
SWIN_CHANNELS = (
    192,
    384,
    768,
)
DETECT_STRIDES = (
    8.0,
    16.0,
    32.0,
)
SEED = 42
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

# Seed all relevant RNGs to make experiments as repeatable as possible.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ============================================================
# BREAST ROI
# ============================================================

# Find the breast region and return a robust bounding box for cropping.

def detect_breast_roi(image):
    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image, got {image.shape}"
        )
    h, w = image.shape
    _, thresh = cv2.threshold(
        image,
        5,
        255,
        cv2.THRESH_BINARY,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
    )
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0, 0, w, h
    contour = max(
        contours,
        key=cv2.contourArea,
    )
    x, y, bw, bh = cv2.boundingRect(
        contour
    )
    if (
        bw < 0.05 * w
        or bh < 0.05 * h
    ):
        return 0, 0, w, h
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2

# ============================================================
# DICOM PREPROCESSING
# ============================================================

# Read a DICOM mammogram and apply windowing to produce an 8-bit image.

def preprocess_dicom(dicom_path):
    from mammo_prep.windowing import preprocess_window
    ds = pydicom.dcmread(
        str(dicom_path)
    )
    image = ds.pixel_array.astype(
        np.float32
    )
    photometric = getattr(
        ds,
        "PhotometricInterpretation",
        "",
    )
    if photometric == "MONOCHROME1":
        image = image.max() - image
    image = preprocess_window(
        image,
        dicom_dataset=ds,
        method="breast_tissue",
        voi_func="LINEAR",
        exclude_background=True,
        output_dtype=np.uint8,
    )
    return image

# ============================================================
# RESIZE
# ============================================================

# Helpers to resize while keeping aspect ratio and to pad into a fixed square.

def resize_keep_aspect(
    image,
    max_size,
):
    h, w = image.shape[:2]
    scale = min(
        max_size / float(w),
        max_size / float(h),
    )
    new_w = max(
        1,
        int(round(w * scale)),
    )
    new_h = max(
        1,
        int(round(h * scale)),
    )
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, scale

def resize_and_pad(
    image,
    img_size,
):
    h, w = image.shape[:2]
    scale = min(
        img_size / float(w),
        img_size / float(h),
    )
    new_w = max(
        1,
        int(round(w * scale)),
    )
    new_h = max(
        1,
        int(round(h * scale)),
    )
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros(
        (
            img_size,
            img_size,
            3,
        ),
        dtype=np.uint8,
    )
    pad_x = (
        img_size - new_w
    ) // 2
    pad_y = (
        img_size - new_h
    ) // 2
    canvas[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ] = resized
    return (
        canvas,
        scale,
        pad_x,
        pad_y,
    )

# ============================================================
# BOX TRANSFORMATION
# ============================================================

# Convert YOLO boxes across coordinate spaces (full image -> ROI -> padded image).

def transform_boxes_to_roi(
    labels,
    original_width,
    original_height,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2,
):
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1
    transformed = []
    for label in labels:
        cls_id, cx, cy, bw, bh = label
        x_center = (
            cx * original_width
        )
        y_center = (
            cy * original_height
        )
        box_w = (
            bw * original_width
        )
        box_h = (
            bh * original_height
        )
        x_min = (
            x_center - box_w / 2.0
        )
        y_min = (
            y_center - box_h / 2.0
        )
        x_max = (
            x_center + box_w / 2.0
        )
        y_max = (
            y_center + box_h / 2.0
        )
        x_min -= roi_x1
        x_max -= roi_x1
        y_min -= roi_y1
        y_max -= roi_y1
        x_min = np.clip(
            x_min,
            0,
            roi_w,
        )
        x_max = np.clip(
            x_max,
            0,
            roi_w,
        )
        y_min = np.clip(
            y_min,
            0,
            roi_h,
        )
        y_max = np.clip(
            y_max,
            0,
            roi_h,
        )
        new_w = x_max - x_min
        new_h = y_max - y_min
        if (
            new_w <= 1
            or new_h <= 1
        ):
            continue
        new_cx = (
            x_min + x_max
        ) / 2.0
        new_cy = (
            y_min + y_max
        ) / 2.0
        transformed.append(
            [
                cls_id,
                new_cx / roi_w,
                new_cy / roi_h,
                new_w / roi_w,
                new_h / roi_h,
            ]
        )
    if not transformed:
        return np.zeros(
            (0, 5),
            dtype=np.float32,
        )
    return np.asarray(
        transformed,
        dtype=np.float32,
    )

def transform_boxes_to_padded_image(
    labels,
    crop_width,
    crop_height,
    scale,
    pad_x,
    pad_y,
    img_size,
):
    transformed = []
    for label in labels:
        cls_id, cx, cy, bw, bh = label
        x_center = (
            cx * crop_width
        )
        y_center = (
            cy * crop_height
        )
        box_w = (
            bw * crop_width
        )
        box_h = (
            bh * crop_height
        )
        x_center = (
            x_center * scale
            + pad_x
        )
        y_center = (
            y_center * scale
            + pad_y
        )
        box_w *= scale
        box_h *= scale
        x_center /= img_size
        y_center /= img_size
        box_w /= img_size
        box_h /= img_size
        x_center = np.clip(
            x_center,
            0.0,
            1.0,
        )
        y_center = np.clip(
            y_center,
            0.0,
            1.0,
        )
        box_w = np.clip(
            box_w,
            0.0,
            1.0,
        )
        box_h = np.clip(
            box_h,
            0.0,
            1.0,
        )
        if (
            box_w <= 0
            or box_h <= 0
        ):
            continue
        transformed.append(
            [
                cls_id,
                x_center,
                y_center,
                box_w,
                box_h,
            ]
        )
    if not transformed:
        return np.zeros(
            (0, 5),
            dtype=np.float32,
        )
    return np.asarray(
        transformed,
        dtype=np.float32,
    )

# ============================================================
# DATASET
# ============================================================

# Dataset that builds valid DICOM/label pairs and applies full preprocessing.

def build_dataset_records(annotations_csv, network_images, val_fraction=0.1, seed=SEED):
    """Build train/validation/test records from CSV splits and DICOM files."""
    annotations_csv = Path(annotations_csv)
    network_images = Path(network_images)
    if not annotations_csv.is_file():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")
    if not network_images.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {network_images}")
    df = pd.read_csv(annotations_csv)
    required = {"study_id", "image_id", "split", "xmin", "ymin", "xmax", "ymax"}
    missing = required - set(df.columns)
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
    box_df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"]).copy()
    box_df = box_df.drop_duplicates(subset=["study_id", "image_id", "xmin", "ymin", "xmax", "ymax"])
    boxes_by_image = {}
    for row in box_df.itertuples(index=False):
        key = (row.study_id, row.image_id)
        x1, y1, x2, y2 = float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)
        if np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1:
            boxes_by_image.setdefault(key, []).append([x1, y1, x2, y2])
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
    training_records = records_by_split["training"]
    test_records = records_by_split["test"]
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
        print(f"[{name}] Images: {len(records):,} | Positive: {positives:,} | Negative: {len(records) - positives:,}", flush=True)
    train_studies = {r["study_id"] for r in train_records}
    val_studies = {r["study_id"] for r in val_records}
    test_studies = {r["study_id"] for r in test_records}
    if not train_studies.isdisjoint(val_studies) or not train_studies.isdisjoint(test_studies) or not val_studies.isdisjoint(test_studies):
        raise RuntimeError("Study leakage detected between train, validation and test.")
    return train_records, val_records, test_records

# TRAIN-ONLY POSITIVE FILTERING
def filter_positive_records(records):
    """Keep only TRAIN images that have at least one CSV bounding box.

    Call this function only on the training records. Validation and official
    test records are deliberately left unchanged.
    """
    records = list(records)
    positives = [record for record in records if record.get("boxes")]
    print(
        f"[TRAIN POSITIVES] Before: {len(records):,} images | "
        f"{len(positives):,} positive | {len(records) - len(positives):,} negative",
        flush=True,
    )
    if not positives:
        raise RuntimeError("TRAIN contains no positive images.")
    print(
        f"[TRAIN POSITIVES] After: {len(positives):,} images (positive only) | "
        f"removed {len(records) - len(positives):,} negative images from TRAIN only.",
        flush=True,
    )
    return positives

class VindrSwinDataset(Dataset):
    def __init__(self, records, img_size=1024):
        self.records = records
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_labels(boxes, original_width, original_height):
        labels = []
        for x1, y1, x2, y2 in boxes:
            x1 = float(np.clip(x1, 0, original_width))
            x2 = float(np.clip(x2, 0, original_width))
            y1 = float(np.clip(y1, 0, original_height))
            y2 = float(np.clip(y2, 0, original_height))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2.0) / original_width
            cy = ((y1 + y2) / 2.0) / original_height
            bw = (x2 - x1) / original_width
            bh = (y2 - y1) / original_height
            labels.append([0.0, cx, cy, bw, bh])
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
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        final_image, scale, pad_x, pad_y = resize_and_pad(crop_bgr, self.img_size)
        labels_final = transform_boxes_to_padded_image(labels_roi, crop_w, crop_h, scale, pad_x, pad_y, self.img_size)
        final_image = np.transpose(final_image.astype(np.float32) / 255.0, (2, 0, 1))
        return torch.from_numpy(final_image).float(), torch.from_numpy(labels_final).float(), str(record["dicom_path"])

# ============================================================
# COLLATE
# ============================================================

# Merge variable-length box annotations into a YOLO-compatible batch dict.

def collate_fn(batch):
    images = []
    batch_idx = []
    classes = []
    boxes = []
    paths = []
    for i, (
        image,
        labels,
        path,
    ) in enumerate(batch):
        images.append(image)
        paths.append(path)
        if labels.numel() == 0:
            continue
        n = labels.shape[0]
        batch_idx.append(
            torch.full(
                (n,),
                i,
                dtype=torch.long,
            )
        )
        classes.append(
            labels[:, 0:1]
        )
        boxes.append(
            labels[:, 1:5]
        )
    images = torch.stack(
        images,
        dim=0,
    )
    if batch_idx:
        batch_idx = torch.cat(
            batch_idx,
            dim=0,
        )
        classes = torch.cat(
            classes,
            dim=0,
        )
        boxes = torch.cat(
            boxes,
            dim=0,
        )
    else:
        batch_idx = torch.zeros(
            (0,),
            dtype=torch.long,
        )
        classes = torch.zeros(
            (0, 1),
            dtype=torch.float32,
        )
        boxes = torch.zeros(
            (0, 4),
            dtype=torch.float32,
        )
    targets = {
        "batch_idx": batch_idx,
        "cls": classes,
        "bboxes": boxes,
    }
    return (
        images,
        targets,
        paths,
    )

# ============================================================
# SWIN + YOLOv8
# ============================================================

# Model that uses Swin features and feeds them into a YOLOv8 Detect head.

class SwinYOLO(nn.Module):
    def __init__(
        self,
        img_size=1024,
        num_classes=1,
        pretrained=True,
    ):
        super().__init__()
        print(
            f"[INIT] Loading {MODEL_NAME}...",
            flush=True,
        )
        self.backbone = (
            timm.create_model(
                MODEL_NAME,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3),
                img_size=img_size,
            )
        )
        print(
            "[INIT] Creating YOLOv8 Detect head...",
            flush=True,
        )
        self.detect = Detect(
            nc=num_classes,
            ch=SWIN_CHANNELS,
        )
        self.detect.stride = (
            torch.tensor(
                DETECT_STRIDES,
                dtype=torch.float32,
            )
        )
        self.detect.bias_init()
    def forward(self, x):
        features = self.backbone(x)
        swin_features = []
        for feat in features:
            if feat.ndim != 4:
                raise RuntimeError(
                    "Unexpected Swin feature "
                    f"shape: {feat.shape}"
                )
            feat = feat.permute(
                0,
                3,
                1,
                2,
            ).contiguous()
            swin_features.append(
                feat
            )
        predictions = self.detect(
            swin_features
        )
        return predictions

# ============================================================
# ULTRALYTICS LOSS WRAPPER
# ============================================================

# Minimal adapter so Ultralytics v8DetectionLoss can consume this custom model.

class DetectionLossModel(
    nn.Module
):
    def __init__(
        self,
        detect,
    ):
        super().__init__()
        self.model = nn.ModuleList(
            [detect]
        )
        self.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )

def create_loss(model):
    loss_model = DetectionLossModel(
        model.detect
    )
    criterion = v8DetectionLoss(
        loss_model
    )
    return criterion

# ============================================================
# LOSS HANDLING
# ============================================================

# Normalize the returned loss structure into a single scalar for backprop.

def compute_loss(
    criterion,
    predictions,
    targets,
):
    raw_loss, loss_items = criterion(
        predictions,
        targets,
    )
    if not isinstance(
        raw_loss,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Ultralytics loss returned "
            f"unexpected type: "
            f"{type(raw_loss)}"
        )
    if raw_loss.numel() == 1:
        total_loss = raw_loss.reshape(
            ()
        )
    else:
        total_loss = raw_loss.sum()
    return (
        total_loss,
        loss_items,
        raw_loss,
    )

# ============================================================
# CHECKPOINT
# ============================================================

# Save and restore model/optimizer/scheduler/scaler state for resume training.

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(
        checkpoint,
        path,
    )

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    print(
        f"[RESUME] Loading checkpoint: "
        f"{path}",
        flush=True,
    )
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )
    model.load_state_dict(
        checkpoint["model"]
    )
    start_epoch = (
        checkpoint.get(
            "epoch",
            0,
        )
        + 1
    )
    best_val_loss = (
        checkpoint.get(
            "best_val_loss",
            float("inf"),
        )
    )
    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )
    if (
        scaler is not None
        and "scaler" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )
    print(
        f"[RESUME] Starting from epoch "
        f"{start_epoch}",
        flush=True,
    )
    return (
        start_epoch,
        best_val_loss,
    )

# ============================================================
# OUTPUT INSPECTION
# ============================================================

# Debug utility to recursively print prediction structure and tensor shapes.

def inspect_prediction(
    obj,
    prefix="",
):
    if isinstance(
        obj,
        torch.Tensor,
    ):
        print(
            f"{prefix}"
            f"Tensor "
            f"shape={tuple(obj.shape)} "
            f"dtype={obj.dtype}"
        )
        return
    if isinstance(
        obj,
        dict,
    ):
        print(
            f"{prefix}"
            f"dict keys={list(obj.keys())}"
        )
        for key, value in obj.items():
            print(
                f"{prefix}  [{key}]"
            )
            inspect_prediction(
                value,
                prefix + "    ",
            )
        return
    if isinstance(
        obj,
        (list, tuple),
    ):
        print(
            f"{prefix}"
            f"{type(obj).__name__} "
            f"length={len(obj)}"
        )
        for i, value in enumerate(obj):
            print(
                f"{prefix}  [{i}]"
            )
            inspect_prediction(
                value,
                prefix + "    ",
            )
        return
    print(
        f"{prefix}"
        f"type={type(obj)} "
        f"value={obj}"
    )

# ============================================================
# TEST
# ============================================================

# End-to-end smoke test: data -> model -> loss -> backward with gradient checks.

def run_test():
    print()
    print("=" * 70)
    print("SWIN + YOLO INTEGRATION TEST")
    print("=" * 70)
    print(
        f"Device: {DEVICE}"
    )
    if torch.cuda.is_available():
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            "VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    print()
    print(
        "[TEST] Loading dataset..."
    )
    train_records, _, _ = build_dataset_records(ANNOTATIONS_CSV, NETWORK_IMAGES)
    train_records = filter_positive_records(train_records)
    dataset = VindrSwinDataset(train_records, IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    print(
        "[TEST] Dataset ready."
    )
    print()
    print(
        "[TEST] Creating model..."
    )
    model = SwinYOLO(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        pretrained=True,
    )
    model = model.to(
        DEVICE
    )
    model.detect.stride = (
        torch.tensor(
            DETECT_STRIDES,
            dtype=torch.float32,
            device=DEVICE,
        )
    )
    print(
        "[TEST] Creating loss..."
    )
    criterion = create_loss(
        model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    amp_enabled = (
        DEVICE.type == "cuda"
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print()
    print(
        "Trainable parameters: "
        f"{total_params / 1e6:.2f} M"
    )
    print()
    print(
        "[TEST] Searching for positive sample..."
    )
    found = False
    for (
        images,
        targets,
        paths,
    ) in loader:
        if (
            targets["bboxes"].shape[0]
            > 0
        ):
            found = True
            break
    if not found:
        raise RuntimeError(
            "No training image with "
            "bounding boxes was found."
        )
    print(
        "[TEST] Positive sample found."
    )
    print(
        f"Image tensor: "
        f"{tuple(images.shape)}"
    )
    print(
        "Targets: "
        f"{targets['bboxes'].shape[0]} boxes"
    )
    print(
        f"DICOM: {paths[0]}"
    )
    print()
    print(
        "Target boxes:"
    )
    print(
        targets["bboxes"]
    )
    print()
    print(
        "Target classes:"
    )
    print(
        targets["cls"]
    )
    images = images.to(
        DEVICE,
        non_blocking=True,
    )
    targets = {
        k: v.to(
            DEVICE,
            non_blocking=True,
        )
        for k, v in targets.items()
    }
    print()
    print(
        "[TEST] Forward pass..."
    )
    model.train()
    with torch.amp.autocast(
        device_type="cuda",
        enabled=amp_enabled,
    ):
        predictions = model(
            images
        )
    print(
        "[TEST] YOLO Detect output:"
    )
    inspect_prediction(
        predictions
    )
    print()
    print(
        "[TEST] Computing YOLOv8 loss..."
    )
    total_loss, loss_items, raw_loss = (
        compute_loss(
            criterion,
            predictions,
            targets,
        )
    )
    print()
    print(
        "Raw loss:"
    )
    print(
        raw_loss.detach()
        .float()
        .cpu()
        .numpy()
    )
    print()
    print(
        f"Total loss: "
        f"{total_loss.item():.6f}"
    )
    print()
    print(
        "[TEST] Backward pass..."
    )
    scaler.scale(
        total_loss
    ).backward()
    scaler.unscale_(
        optimizer
    )
    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )
    )
    print(
        "Gradient norm: "
        f"{float(grad_norm):.6f}"
    )
    swin_gradients = []
    for (
        name,
        parameter,
    ) in model.backbone.named_parameters():
        if parameter.grad is not None:
            swin_gradients.append(
                parameter.grad.detach()
                .abs()
                .mean()
                .item()
            )
    print()
    if not swin_gradients:
        raise RuntimeError(
            "No gradients found in Swin. "
            "The Swin backbone is not "
            "being trained."
        )
    print(
        "Swin parameters with gradients: "
        f"{len(swin_gradients)}"
    )
    print(
        "Mean absolute Swin gradient: "
        f"{np.mean(swin_gradients):.8e}"
    )
    print()
    print("=" * 70)
    print(
        "TEST PASSED"
    )
    print("=" * 70)
    print()
    print(
        "NETWORK DICOM"
        " -> breast_tissue + LINEAR"
        " -> breast ROI"
        " -> Swin-Tiny"
        " -> YOLOv8 Detect"
        " -> YOLOv8 loss"
        " -> backward"
    )
    print()
    print(
        "Swin is TRAINABLE."
    )
    print()

# ============================================================
# TRAIN ONE EPOCH
# ============================================================

# One training epoch with AMP, gradient accumulation, and gradient clipping.

def train_one_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    epoch,
    accumulation_steps,
):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(
        set_to_none=True
    )
    amp_enabled = (
        DEVICE.type == "cuda"
    )
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:03d} [TRAIN]",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    for step, (
        images,
        targets,
        paths,
    ) in enumerate(progress):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )
        targets = {
            k: v.to(
                DEVICE,
                non_blocking=True,
            )
            for k, v in targets.items()
        }
        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            predictions = model(
                images
            )
            total_loss, loss_items, raw_loss = (
                compute_loss(
                    criterion,
                    predictions,
                    targets,
                )
            )
            loss_backward = (
                total_loss
                / accumulation_steps
            )
        scaler.scale(
            loss_backward
        ).backward()
        should_step = (
            (
                (step + 1)
                % accumulation_steps
            ) == 0
            or
            (step + 1)
            == len(loader)
        )
        if should_step:
            scaler.unscale_(
                optimizer
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )
            scaler.step(
                optimizer
            )
            scaler.update()
            optimizer.zero_grad(
                set_to_none=True
            )
        running_loss += (
            total_loss.item()
        )
        avg_loss = (
            running_loss
            / (step + 1)
        )
        progress.set_postfix(
            loss=f"{avg_loss:.5f}",
            batch_loss=f"{total_loss.item():.5f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
    return (
        running_loss
        / len(loader)
    )

# ============================================================
# VALIDATION
# ============================================================

# Validation pass that computes mean detection loss without gradient updates.
@torch.no_grad()

def validate(
    model,
    criterion,
    loader,
):
    model.train()
    for module in model.modules():
        if isinstance(
            module,
            nn.BatchNorm2d,
        ):
            module.eval()
    running_loss = 0.0
    amp_enabled = (
        DEVICE.type == "cuda"
    )
    progress = tqdm(
        loader,
        total=len(loader),
        desc="Validation",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    for step, (
        images,
        targets,
        paths,
    ) in enumerate(progress):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )
        targets = {
            k: v.to(
                DEVICE,
                non_blocking=True,
            )
            for k, v in targets.items()
        }
        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            predictions = model(
                images
            )
            total_loss, loss_items, raw_loss = (
                compute_loss(
                    criterion,
                    predictions,
                    targets,
                )
            )
        running_loss += (
            total_loss.item()
        )
        avg_loss = (
            running_loss
            / (step + 1)
        )
        progress.set_postfix(
            loss=f"{avg_loss:.5f}",
        )
    return (
        running_loss
        / len(loader)
    )

# ============================================================
# FULL TRAINING
# ============================================================

# Full training loop orchestration: loaders, model, optimization, checkpoints.

def run_training(
    resume=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    workers=NUM_WORKERS,
    img_size=IMG_SIZE,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    print()
    print("=" * 70)
    print(
        "SWIN-TINY + YOLOv8 TRAINING (POSITIVE-ONLY TRAIN)"
    )
    print("=" * 70)
    print()
    print("[INIT] Configuration")
    print(
        f"  Network images: {NETWORK_IMAGES}"
    )
    print(
        f"  Annotations: {ANNOTATIONS_CSV}"
    )
    print(
        f"  Output: {OUTPUT_DIR}"
    )
    print(
        f"  Image size: {img_size} × {img_size}"
    )
    print(
        f"  Batch size: {batch_size}"
    )
    print(
        "  Gradient accumulation: "
        f"{ACCUMULATION_STEPS}"
    )
    print(
        "  Effective batch size: "
        f"{batch_size * ACCUMULATION_STEPS}"
    )
    print(
        f"  Epochs: {epochs}"
    )
    print(
        f"  Workers: {workers}"
    )
    print(
        f"  Learning rate: {LEARNING_RATE}"
    )
    print(
        f"  Weight decay: {WEIGHT_DECAY}"
    )
    print(
        f"  Device: {DEVICE}"
    )
    if torch.cuda.is_available():
        print(
            f"  GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"  VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    print()
    print("[INIT] Pipeline")
    print(
        "  Network DICOM"
    )
    print(
        "    ↓"
    )
    print(
        "  breast_tissue + LINEAR"
    )
    print(
        "    ↓"
    )
    print(
        "  Breast ROI extraction"
    )
    print(
        "    ↓"
    )
    print(
        "  Resize + padding → 1024 × 1024"
    )
    print(
        "    ↓"
    )
    print(
        "  Swin-Tiny pretrained + trainable"
    )
    print(
        "    ↓"
    )
    print(
        "  P3 / P4 / P5"
    )
    print(
        "    ↓"
    )
    print(
        "  YOLOv8 Detect"
    )
    print(
        "    ↓"
    )
    print(
        "  v8DetectionLoss"
    )
    print()
    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    train_records, val_records, test_records = build_dataset_records(ANNOTATIONS_CSV, NETWORK_IMAGES)
    # Restrict TRAIN to positives. Validation and the complete official TEST split stay untouched.
    train_records = filter_positive_records(train_records)
    train_dataset = VindrSwinDataset(train_records, img_size)
    val_dataset = VindrSwinDataset(val_records, img_size)
    print(f"[INIT] Train images: {len(train_dataset):,}", flush=True)
    print(f"[INIT] Validation images: {len(val_dataset):,}", flush=True)
    print(f"[INIT] Full test set reserved for evaluation: {len(test_records):,} images", flush=True)
    print(
        "[INIT] Creating DataLoaders...",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            workers > 0
        ),
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            workers > 0
        ),
        collate_fn=collate_fn,
        drop_last=False,
    )
    print(
        "[INIT] DataLoaders ready.",
        flush=True,
    )
    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    print()
    print(
        "[INIT] Building model...",
        flush=True,
    )
    model = SwinYOLO(
        img_size=img_size,
        num_classes=NUM_CLASSES,
        pretrained=True,
    )
    model = model.to(
        DEVICE
    )
    model.detect.stride = (
        torch.tensor(
            DETECT_STRIDES,
            dtype=torch.float32,
            device=DEVICE,
        )
    )
    print(
        "[INIT] Model ready.",
        flush=True,
    )
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(
        f"[INIT] Parameters: "
        f"{total_params / 1e6:.2f} M total"
    )
    print(
        f"[INIT] Trainable: "
        f"{trainable_params / 1e6:.2f} M"
    )
    print()
    print(
        "[INIT] Swin features:"
    )
    print(
        "  P3: 192 × 128 × 128 | stride 8"
    )
    print(
        "  P4: 384 × 64 × 64   | stride 16"
    )
    print(
        "  P5: 768 × 32 × 32   | stride 32"
    )
    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------
    print()
    print(
        "[INIT] Creating YOLOv8 detection loss...",
        flush=True,
    )
    criterion = create_loss(
        model
    )
    print(
        "[INIT] Detection loss ready.",
        flush=True,
    )
    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------
    print()
    print(
        "[INIT] Creating AdamW optimizer...",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    print(
        "[INIT] Creating cosine scheduler...",
        flush=True,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=LEARNING_RATE * 0.01,
        )
    )
    amp_enabled = (
        DEVICE.type == "cuda"
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )
    print(
        f"[INIT] AMP enabled: "
        f"{amp_enabled}"
    )
    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------
    start_epoch = 1
    best_val_loss = float(
        "inf"
    )
    if resume is not None:
        print()
        print(
            "[INIT] Resume requested."
        )
        (
            start_epoch,
            best_val_loss,
        ) = load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            scaler,
        )
    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print(
        "READY TO TRAIN"
    )
    print("=" * 70)
    print(
        f"Starting epoch: {start_epoch}"
    )
    print(
        f"Final epoch: {epochs}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print()
    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------
    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        epoch_start = time.time()
        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch}/{epochs}"
        )
        print("=" * 70)
        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )
        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )
        print()
        print(
            "Training..."
        )
        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------
        train_loss = (
            train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                scaler,
                epoch,
                ACCUMULATION_STEPS,
            )
        )
        print()
        print(
            f"Train loss: "
            f"{train_loss:.6f}"
        )
        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------
        print()
        print(
            "Validation..."
        )
        val_loss = validate(
            model,
            criterion,
            val_loader,
        )
        print()
        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )
        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------
        scheduler.step()
        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------
        epoch_time = (
            time.time()
            - epoch_start
        )
        print()
        print(
            f"Epoch time: "
            f"{epoch_time / 60:.2f} min"
        )
        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------
        last_path = (
            OUTPUT_DIR
            / "last.pt"
        )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
        )
        print(
            "[CHECKPOINT] Saved last.pt"
        )
        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = (
                OUTPUT_DIR
                / "best.pt"
            )
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
            )
            print(
                "[CHECKPOINT] New best model saved"
            )
        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------
        print()
        print("-" * 70)
        print(
            f"EPOCH {epoch}/{epochs} SUMMARY"
        )
        print(
            f"Train loss: {train_loss:.6f}"
        )
        print(
            f"Val loss:   {val_loss:.6f}"
        )
        print(
            f"Best val:   {best_val_loss:.6f}"
        )
        print(
            f"Time:       {epoch_time / 60:.2f} min"
        )
        print(
            f"LR:         "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )
        print("-" * 70)
    print()
    print("=" * 70)
    print(
        "TRAINING FINISHED"
    )
    print("=" * 70)
    print(
        "Best validation loss: "
        f"{best_val_loss:.6f}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )

# ============================================================
# ARGUMENTS
# ============================================================

# CLI options to run a quick test, train from scratch, or resume from checkpoint.

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Swin-Tiny + YOLOv8 "
            "on ViNDR mammography (positive-only TRAIN)."
        )
    )
    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )
    mode.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run one real "
            "forward/loss/backward test."
        ),
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
    )
    mode.add_argument(
        "--resume",
        type=str,
        metavar="CHECKPOINT",
        help=(
            "Resume training from "
            "a checkpoint."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE,
    )
    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================

# Entry point that dispatches execution mode based on CLI arguments.

def main():
    args = parse_args()
    set_seed(
        SEED
    )
    if args.test:
        run_test()
        return
    if args.resume:
        run_training(
            resume=args.resume,
            epochs=args.epochs,
            batch_size=args.batch,
            workers=args.workers,
            img_size=args.img_size,
        )
        return
    if args.train:
        run_training(
            resume=None,
            epochs=args.epochs,
            batch_size=args.batch,
            workers=args.workers,
            img_size=args.img_size,
        )
        return

if __name__ == "__main__":
    main()
