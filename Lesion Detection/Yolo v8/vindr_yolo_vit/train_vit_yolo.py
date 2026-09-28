#!/usr/bin/env python3

"""
ViT + YOLOv8 detector for VinDr-Mammo.

Pipeline:
    DICOM
      -> breast_tissue + LINEAR
      -> breast ROI
      -> resize + square padding
      -> ViT-Base patch16
      -> YOLOv8 Detect head
      -> lesion bounding boxes

The ViT backbone can be initialized from the previously trained
binary ViT classifier checkpoint. The classifier head itself is discarded.

This is an independent pipeline from the Swin + YOLO model.
"""

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import ViTModel

from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss

from mammo_prep.windowing import preprocess_window


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

OUTPUT_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/train_v1"
)

# Previously trained ViT classifier.
VIT_CLASSIFIER_CHECKPOINT = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_vit/vit_splits/vit_checkpoints/vit_vindr_final.pt"
)

MODEL_NAME = "google/vit-base-patch16-224"

NUM_CLASSES = 1

# 512 is deliberate:
# ViT-B/16 at 1024 would create 4096 tokens and is extremely
# expensive because self-attention scales quadratically.
IMG_SIZE = 512

BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 16

DETECT_CHANNELS = (
    256,
    256,
    256,
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

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# BREAST ROI
# Same ROI procedure used by the existing Swin + YOLO code.
# ============================================================

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

def preprocess_dicom(dicom_path):

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
# RESIZE + PAD
# ============================================================

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

        x_center = cx * original_width
        y_center = cy * original_height

        box_w = bw * original_width
        box_h = bh * original_height

        x_min = x_center - box_w / 2.0
        y_min = y_center - box_h / 2.0

        x_max = x_center + box_w / 2.0
        y_max = y_center + box_h / 2.0

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

        x_center = cx * crop_width
        y_center = cy * crop_height

        box_w = bw * crop_width
        box_h = bh * crop_height

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

class VindrViTYOLODataset(Dataset):

    def __init__(
        self,
        network_images,
        annotations_csv,
        local_dataset,
        split,
        img_size=512,
    ):

        self.network_images = Path(
            network_images
        )

        self.annotations_csv = Path(
            annotations_csv
        )

        self.local_dataset = Path(
            local_dataset
        )

        self.split = split
        self.img_size = img_size

        self.label_dir = (
            self.local_dataset
            / "labels"
            / split
        )

        if not self.network_images.exists():
            raise FileNotFoundError(
                f"Network image directory not found: "
                f"{self.network_images}"
            )

        if not self.annotations_csv.exists():
            raise FileNotFoundError(
                f"Annotations CSV not found: "
                f"{self.annotations_csv}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: "
                f"{self.label_dir}"
            )

        df = pd.read_csv(
            self.annotations_csv
        )

        required_columns = [
            "study_id",
            "image_id",
            "split",
        ]

        missing = [
            c
            for c in required_columns
            if c not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing CSV columns: {missing}"
            )

        if split in ("train", "val"):
            csv_split = "training"
        elif split == "test":
            csv_split = "test"
        else:
            raise ValueError(
                f"Unknown split: {split}"
            )

        df = df[
            df["split"].astype(str)
            == csv_split
        ].copy()

        label_ids = {
            p.stem
            for p in self.label_dir.glob(
                "*.txt"
            )
        }

        df = df[
            df["image_id"].astype(str)
            .isin(label_ids)
        ].copy()

        # The ViT split CSVs define the exact train/val partition.
        if split == "val":

            vit_val_csv = (
                self.local_dataset.parent
                / "vindr_yolo_vit"
                / "vit_splits"
                / "vindr_val.csv"
            )

            if vit_val_csv.exists():

                val_df = pd.read_csv(
                    vit_val_csv
                )

                val_ids = set(
                    val_df["image_id"]
                    .astype(str)
                )

                df = df[
                    df["image_id"].astype(str)
                    .isin(val_ids)
                ].copy()

        df = df.drop_duplicates(
            subset=["image_id"]
        )

        self.records = []

        missing_dicoms = 0

        for _, row in df.iterrows():

            study_id = str(
                row["study_id"]
            )

            image_id = str(
                row["image_id"]
            )

            dicom_path = (
                self.network_images
                / study_id
                / f"{image_id}.dicom"
            )

            label_path = (
                self.label_dir
                / f"{image_id}.txt"
            )

            if not dicom_path.exists():
                missing_dicoms += 1
                continue

            if not label_path.exists():
                continue

            self.records.append(
                {
                    "study_id": study_id,
                    "image_id": image_id,
                    "dicom_path": dicom_path,
                    "label_path": label_path,
                }
            )

        print(
            f"[{split}] samples: "
            f"{len(self.records):,}",
            flush=True,
        )

        print(
            f"[{split}] missing DICOMs: "
            f"{missing_dicoms:,}",
            flush=True,
        )

        if not self.records:
            raise RuntimeError(
                f"No valid samples for {split}"
            )

    def __len__(self):
        return len(self.records)

    def _load_labels(self, path):

        labels = []

        with open(path, "r") as f:

            for line in f:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                labels.append(
                    [float(v) for v in values]
                )

        if not labels:
            return np.zeros(
                (0, 5),
                dtype=np.float32,
            )

        return np.asarray(
            labels,
            dtype=np.float32,
        )

    def __getitem__(self, index):

        record = self.records[index]

        image_path = record[
            "dicom_path"
        ]

        label_path = record[
            "label_path"
        ]

        image = preprocess_dicom(
            image_path
        )

        original_h, original_w = (
            image.shape
        )

        labels = self._load_labels(
            label_path
        )

        # First resize exactly as the Swin pipeline.
        image, _ = resize_keep_aspect(
            image,
            self.img_size,
        )

        resized_h, resized_w = (
            image.shape
        )

        # Breast ROI.
        x1, y1, x2, y2 = (
            detect_breast_roi(image)
        )

        crop = image[
            y1:y2,
            x1:x2,
        ]

        crop_h, crop_w = (
            crop.shape
        )

        labels_roi = (
            transform_boxes_to_roi(
                labels,
                resized_w,
                resized_h,
                x1,
                y1,
                x2,
                y2,
            )
        )

        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_GRAY2RGB,
        )

        (
            final_image,
            scale,
            pad_x,
            pad_y,
        ) = resize_and_pad(
            crop_rgb,
            self.img_size,
        )

        labels_final = (
            transform_boxes_to_padded_image(
                labels_roi,
                crop_w,
                crop_h,
                scale,
                pad_x,
                pad_y,
                self.img_size,
            )
        )

        # ViT ImageNet normalization.
        image_float = (
            final_image.astype(
                np.float32
            ) / 255.0
        )

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        ).reshape(1, 1, 3)

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        ).reshape(1, 1, 3)

        image_float = (
            image_float - mean
        ) / std

        image_tensor = torch.from_numpy(
            image_float.transpose(2, 0, 1)
        ).float()

        target_tensor = torch.from_numpy(
            labels_final
        ).float()

        return (
            image_tensor,
            target_tensor,
            str(image_path),
        )


# ============================================================
# COLLATE
# ============================================================

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
# ViT + YOLO
# ============================================================

class ViTYOLO(nn.Module):

    def __init__(
        self,
        pretrained_classifier=None,
        num_classes=1,
    ):

        super().__init__()

        print(
            f"[INIT] Loading {MODEL_NAME}",
            flush=True,
        )

        # Load the ViT backbone directly. We do not instantiate
        # ViTForImageClassification here because its original
        # ImageNet classifier has 1000 outputs, while the previously
        # trained VinDr classifier has 2 outputs. The classifier head
        # is not needed for the YOLO detector.
        self.backbone = ViTModel.from_pretrained(
            MODEL_NAME,
            add_pooling_layer=False,
        )

        if (
            pretrained_classifier is not None
            and Path(pretrained_classifier).exists()
        ):

            print(
                "[INIT] Loading previously trained "
                "ViT backbone:",
                pretrained_classifier,
                flush=True,
            )

            state = torch.load(
                pretrained_classifier,
                map_location="cpu",
            )

            if "state_dict" in state:
                state = state["state_dict"]

            vit_state = {
                key.replace("vit.", "", 1): value
                for key, value in state.items()
                if key.startswith("vit.")
            }

            if not vit_state:
                raise RuntimeError(
                    "No 'vit.*' weights found in the checkpoint: "
                    f"{pretrained_classifier}"
                )

            missing, unexpected = (
                self.backbone.load_state_dict(
                    vit_state,
                    strict=False,
                )
            )

            print(
                f"[INIT] ViT backbone loaded. "
                f"Missing={len(missing)}, "
                f"Unexpected={len(unexpected)}",
                flush=True,
            )

            if missing:
                print(
                    f"[INIT] Missing keys: {missing}",
                    flush=True,
                )

            if unexpected:
                print(
                    f"[INIT] Unexpected keys: {unexpected}",
                    flush=True,
                )

        else:

            print(
                "[INIT] Using HuggingFace pretrained ViT.",
                flush=True,
            )

        hidden = (
            self.backbone.config.hidden_size
        )

        print(
            f"[INIT] ViT hidden size: {hidden}",
            flush=True,
        )

        # Feature pyramid.
        #
        # At 512 input with 16x16 patches:
        #
        # tokens = 32 x 32
        #
        # P3 = 64 x 64  -> stride 8
        # P4 = 32 x 32  -> stride 16
        # P5 = 16 x 16  -> stride 32

        self.p4 = nn.Sequential(
            nn.Conv2d(
                hidden,
                256,
                kernel_size=1,
            ),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        self.p5 = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        print(
            "[INIT] Creating YOLOv8 Detect head...",
            flush=True,
        )

        self.detect = Detect(
            nc=num_classes,
            ch=DETECT_CHANNELS,
        )

        self.detect.stride = torch.tensor(
            DETECT_STRIDES,
            dtype=torch.float32,
        )

        self.detect.bias_init()

    def forward(self, x):

        outputs = self.backbone(
            pixel_values=x,
            interpolate_pos_encoding=True,
            return_dict=True,
        )

        tokens = outputs.last_hidden_state

        # Remove CLS token.
        tokens = tokens[:, 1:, :]

        b, n, c = tokens.shape

        side = int(
            n ** 0.5
        )

        if side * side != n:
            raise RuntimeError(
                f"ViT token count {n} is not square."
            )

        features = tokens.transpose(
            1,
            2,
        ).reshape(
            b,
            c,
            side,
            side,
        ).contiguous()

        p4 = self.p4(
            features
        )

        p3 = F.interpolate(
            p4,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        p3 = self.p3(
            p3
        )

        p5 = self.p5(
            p4
        )

        predictions = self.detect(
            [
                p3,
                p4,
                p5,
            ]
        )

        return predictions


# ============================================================
# ULTRALYTICS LOSS
# ============================================================

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
# CHECKPOINTS
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    if scaler is not None:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    return (
        checkpoint["epoch"],
        checkpoint.get(
            "best_val_loss",
            float("inf"),
        ),
    )


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    accumulation_steps=1,
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    num_batches = 0

    if training:
        optimizer.zero_grad(
            set_to_none=True
        )

    pbar = tqdm(
        loader,
        leave=False,
        desc="train" if training else "val",
    )

    for step, (
        images,
        targets,
        _,
    ) in enumerate(pbar):

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

        with torch.set_grad_enabled(
            training
        ):

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

                predictions = model(
                    images
                )

                loss, _ = criterion(
                    predictions,
                    targets,
                )

                if loss.numel() > 1:
                    loss = loss.sum()

                loss_for_backward = (
                    loss / accumulation_steps
                )

            if training:

                scaler.scale(
                    loss_for_backward
                ).backward()

                if (
                    (step + 1)
                    % accumulation_steps
                    == 0
                    or step + 1
                    == len(loader)
                ):

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
            loss.detach().item()
        )

        num_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        running_loss
        / max(1, num_batches)
    )


# ============================================================
# DATA CHECK
# ============================================================

def check_dataset():

    print("=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    for split in (
        "train",
        "val",
    ):

        dataset = VindrViTYOLODataset(
            NETWORK_IMAGES,
            ANNOTATIONS_CSV,
            LOCAL_DATASET,
            split,
            IMG_SIZE,
        )

        print(
            f"{split}: "
            f"{len(dataset):,} samples"
        )

        image, labels, path = dataset[0]

        print(
            f"  image shape: {tuple(image.shape)}"
        )

        print(
            f"  labels: {labels.shape[0]}"
        )

        print(
            f"  example: {path}"
        )

    print()
    print("Dataset check completed.")


# ============================================================
# TRAIN
# ============================================================

def train(
    resume=None,
):

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("ViT + YOLOv8 TRAINING")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"Image size: {IMG_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch: "
        f"{BATCH_SIZE * ACCUMULATION_STEPS}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    train_dataset = (
        VindrViTYOLODataset(
            NETWORK_IMAGES,
            ANNOTATIONS_CSV,
            LOCAL_DATASET,
            "train",
            IMG_SIZE,
        )
    )

    val_dataset = (
        VindrViTYOLODataset(
            NETWORK_IMAGES,
            ANNOTATIONS_CSV,
            LOCAL_DATASET,
            "val",
            IMG_SIZE,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        collate_fn=collate_fn,
    )

    model = ViTYOLO(
        pretrained_classifier=(
            VIT_CLASSIFIER_CHECKPOINT
        ),
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = create_loss(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.01,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume is not None:

        print(
            f"[RESUME] Loading {resume}",
            flush=True,
        )

        (
            last_epoch,
            best_val_loss,
        ) = load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            scaler,
        )

        start_epoch = (
            last_epoch + 1
        )

        print(
            f"[RESUME] Starting at epoch "
            f"{start_epoch}",
            flush=True,
        )

    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        epoch_start = time.time()

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )
        print("=" * 70)

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            ACCUMULATION_STEPS,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            scaler=None,
            accumulation_steps=1,
        )

        scheduler.step()

        elapsed = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"time={elapsed / 60:.2f} min",
            flush=True,
        )

        last_path = (
            OUTPUT_DIR / "last.pt"
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

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = (
                OUTPUT_DIR / "best.pt"
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
                f"[BEST] Saved: {best_path}",
                flush=True,
            )

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best checkpoint: "
        f"{OUTPUT_DIR / 'best.pt'}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train ViT + YOLOv8 on VinDr"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dataset and preprocessing.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train ViT + YOLO.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint.",
    )

    args = parser.parse_args()

    if args.check:

        check_dataset()
        return

    if args.train:

        train(
            resume=args.resume
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
