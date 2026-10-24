#!/usr/bin/env python3
"""ViT-Base + YOLOv8 lesion detection on VinDr-Mammo. TRAIN regime: positives.

Trains only on the 1,270 TRAIN images with lesions.

Pipeline: DICOM -> breast_tissue windowing -> breast crop -> 1024x1024
-> ViT-Base (P3/P4/P5) -> YOLOv8 Detect -> v8DetectionLoss.

The six experiments ({Swin, ViT} x {full, 4k, positives}) are identical except
for the model and select_train_records(). Keep the rest of the code in sync.

Usage:
    python train_vit_positives.py --test                    # one forward/backward check
    python train_vit_positives.py --train
    python train_vit_positives.py --resume outputs/last.pt
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
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

NUM_CLASSES = 1
IMG_SIZE = 1024
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4  # effective batch = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16
SEED = 42
VAL_FRACTION = 0.1  # of the CSV training studies
REGIME = "positives"
DETECT_STRIDES = (8.0, 16.0, 32.0)  # P3, P4, P5

MODEL_DESCRIPTION = "ViT-Base"
MODEL_NAME = "vit_base_patch16_224"
VIT_EMBED_DIM = 768
VIT_BLOCKS = (3, 7, 11)  # transformer blocks whose tokens become P3 / P4 / P5
VIT_FEATURE_CHANNELS = (256, 512, 768)  # channels of P3 / P4 / P5 after the 1x1 adapters

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


def select_train_records(train_records):
    """TRAIN regime "positives": only TRAIN images with at least one lesion box."""
    positives = [r for r in train_records if r["boxes"]]
    print(f"[TRAIN positives] {len(positives):,} of {len(train_records):,} images", flush=True)
    return positives

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
# LOSS
# ============================================================

class DetectionLossModel(nn.Module):
    """The minimum v8DetectionLoss needs from a model: its Detect head and the loss gains."""

    def __init__(self, detect):
        super().__init__()
        self.model = nn.ModuleList([detect])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


def create_loss(model):
    return v8DetectionLoss(DetectionLossModel(model.detect))


def compute_loss(criterion, predictions, targets):
    """Total loss as a scalar (v8DetectionLoss returns the box / cls / dfl terms)."""
    loss, _ = criterion(predictions, targets)
    return loss.sum()

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


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Restore a training state; returns (next epoch, best validation loss so far)."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    print(f"[RESUME] {path}: continuing from epoch {checkpoint['epoch'] + 1}", flush=True)
    return checkpoint["epoch"] + 1, checkpoint["best_val_loss"]

# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def to_device(images, targets):
    images = images.to(DEVICE, non_blocking=True)
    targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}
    return images, targets


def train_one_epoch(model, criterion, loader, optimizer, scaler, epoch):
    """One epoch with mixed precision, gradient accumulation and gradient clipping."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = DEVICE.type == "cuda"
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [TRAIN]", unit="batch", dynamic_ncols=True)
    for step, (images, targets, _) in enumerate(progress):
        images, targets = to_device(images, targets)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            loss = compute_loss(criterion, model(images), targets)
        scaler.scale(loss / ACCUMULATION_STEPS).backward()
        # Update every ACCUMULATION_STEPS batches, and with whatever is left at the end of the epoch.
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()
        progress.set_postfix(loss=f"{running_loss / (step + 1):.5f}", batch_loss=f"{loss.item():.5f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, criterion, loader):
    """Mean detection loss on VAL, computed exactly like the training loss.

    The Detect head must stay in train() mode: in eval() it returns decoded
    boxes, not the raw outputs v8DetectionLoss expects. The backbone goes to
    eval() so its stochastic layers (Swin's DropPath) are off and the loss is
    deterministic, and BatchNorm goes to eval() so VAL does not update it.
    """
    model.train()
    model.backbone.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    amp_enabled = DEVICE.type == "cuda"
    running_loss = 0.0
    progress = tqdm(loader, desc="Validation", unit="batch", dynamic_ncols=True)
    for step, (images, targets, _) in enumerate(progress):
        images, targets = to_device(images, targets)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            loss = compute_loss(criterion, model(images), targets)
        running_loss += loss.item()
        progress.set_postfix(loss=f"{running_loss / (step + 1):.5f}")
    return running_loss / len(loader)


def make_loaders(batch_size, workers, img_size):
    train_records, val_records, test_records = build_dataset_records()
    train_records = select_train_records(train_records)
    print(f"[INIT] Train images ({REGIME}): {len(train_records):,} | Validation images (unfiltered): "
          f"{len(val_records):,} | Test images (not used here): {len(test_records):,}", flush=True)
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=workers > 0, collate_fn=collate_fn)
    train_loader = DataLoader(VindrDataset(train_records, img_size), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(VindrDataset(val_records, img_size), shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def run_training(resume=None, epochs=EPOCHS, batch_size=BATCH_SIZE, workers=NUM_WORKERS, img_size=IMG_SIZE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"{MODEL_DESCRIPTION.upper()} + YOLOv8 TRAINING · TRAIN regime: {REGIME}")
    print("=" * 70)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Image size: {img_size} × {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {batch_size * ACCUMULATION_STEPS}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_loader, val_loader = make_loaders(batch_size, workers, img_size)
    model = build_model(img_size)
    print(f"[INIT] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    criterion = create_loss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Cosine decay from LEARNING_RATE to 1% of it over all the epochs (stepped once per epoch).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LEARNING_RATE * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    start_epoch, best_val_loss = 1, float("inf")
    if resume is not None:
        start_epoch, best_val_loss = load_checkpoint(resume, model, optimizer, scheduler, scaler)

    # Log lines below are parsed by training_status.py and misc/export_evolucio.py.
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        print("=" * 70)
        print(f"EPOCH {epoch}/{epochs} | Learning rate: {optimizer.param_groups[0]['lr']:.8f}", flush=True)
        print("=" * 70)
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scaler, epoch)
        val_loss = validate(model, criterion, val_loader)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # best.pt = lowest VAL loss; it is the checkpoint used for evaluation.
        save_checkpoint(OUTPUT_DIR / "last.pt", model, optimizer, scheduler, scaler, epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(OUTPUT_DIR / "best.pt", model, optimizer, scheduler, scaler, epoch, best_val_loss)
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
    print(f"TRAINING FINISHED | Best validation loss: {best_val_loss:.6f} | Output: {OUTPUT_DIR}")
    print("=" * 70)

# ============================================================
# SMOKE TEST
# ============================================================

def run_test():
    """One positive TRAIN image through model -> loss -> backward, checking the backbone gets gradients."""
    train_records, _, _ = build_dataset_records()
    record = next(r for r in select_train_records(train_records) if r["boxes"])
    images, targets, paths = collate_fn([VindrDataset([record])[0]])
    print(f"[TEST] {paths[0]} | image {tuple(images.shape)} | {targets['bboxes'].shape[0]} boxes")

    model = build_model()
    model.train()
    images, targets = to_device(images, targets)
    loss = compute_loss(create_loss(model), model(images), targets)
    loss.backward()
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    if not backbone_grads:
        raise RuntimeError("The backbone received no gradients.")
    print(f"[TEST] Loss {loss.item():.4f} | backbone tensors with gradients: {len(backbone_grads)}")
    print("TEST PASSED")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=f"Train {MODEL_DESCRIPTION} + YOLOv8 (TRAIN regime: {REGIME}).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="Run one forward/loss/backward check.")
    mode.add_argument("--train", action="store_true", help="Train from scratch.")
    mode.add_argument("--resume", type=str, metavar="CHECKPOINT", help="Resume training from a checkpoint.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    set_seed(SEED)
    if args.test:
        run_test()
    else:
        run_training(resume=args.resume, epochs=args.epochs, batch_size=args.batch, workers=args.workers,
                     img_size=args.img_size)


if __name__ == "__main__":
    main()
