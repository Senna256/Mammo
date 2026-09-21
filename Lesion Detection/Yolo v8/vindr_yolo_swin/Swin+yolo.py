#!/usr/bin/env python3

"""
Swin-Tiny + YOLOv8 lesion detector
==================================

VinDr-Mammo lesion detection.

Pipeline:

    DICOM
      |
      v
    breast_tissue + LINEAR
      |
      v
    breast ROI
      |
      v
    1024x1024 padded image
      |
      v
    Swin-Tiny (TRAINABLE)
      |
      v
    multi-scale features
      |
      v
    YOLOv8 Detect head (TRAINABLE)
      |
      v
    lesion bounding boxes

IMPORTANT:
- Images in vindr_yolo/images/... are symlinks to DICOM files.
- They are read with pydicom, NOT PIL/OpenCV.
- Swin is fine-tuned jointly with the YOLO detection head.
- One class: lesion.
"""

from pathlib import Path
import random
import time
import argparse

import cv2
import numpy as np
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss


# =============================================================================
# CONFIG
# =============================================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

IMAGE_ROOT = DATASET / "images"

LABEL_ROOT = DATASET / "labels"

TRAIN_IMAGES = IMAGE_ROOT / "train"
VAL_IMAGES = IMAGE_ROOT / "val"

TRAIN_LABELS = LABEL_ROOT / "train"
VAL_LABELS = LABEL_ROOT / "val"

OUTPUT_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/train"
)

MODEL_NAME = "swin_tiny_patch4_window7_224"

NUM_CLASSES = 1

IMG_SIZE = 1024

BATCH_SIZE = 2

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

# Gradient accumulation.
# Effective batch = BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 2

# AMP
USE_AMP = True


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# =============================================================================
# WINDOWING
# =============================================================================

from mammo_prep.windowing import preprocess_window


def load_dicom(path):

    ds = pydicom.dcmread(
        str(path)
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    if getattr(
        ds,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":

        image = image.max() - image

    return image, ds


def preprocess_dicom(
    image,
    ds,
):
    """
    Same windowing pipeline validated previously.
    """

    image = preprocess_window(
        image,
        dicom_dataset=ds,
        method="breast_tissue",
        voi_func="LINEAR",
        exclude_background=True,
        output_dtype=np.uint8,
    )

    return image


# =============================================================================
# BREAST ROI
# =============================================================================

def detect_breast_roi(
    image,
):
    """
    Detect largest breast region.
    """

    gray = image

    threshold = np.percentile(
        gray,
        5,
    )

    binary = (
        gray > threshold
    ).astype(
        np.uint8
    ) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:

        h, w = image.shape

        return (
            0,
            0,
            w,
            h,
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    return (
        max(0, x),
        max(0, y),
        min(image.shape[1], x + w),
        min(image.shape[0], y + h),
    )


# =============================================================================
# IMAGE RESIZE + PADDING
# =============================================================================

def resize_pad(
    image,
    target_size,
):
    """
    Preserve aspect ratio and pad to square.
    """

    h, w = image.shape[:2]

    scale = min(
        target_size / h,
        target_size / w,
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            target_size,
            target_size,
        ),
        dtype=np.uint8,
    )

    left = (
        target_size - new_w
    ) // 2

    top = (
        target_size - new_h
    ) // 2

    canvas[
        top:top + new_h,
        left:left + new_w,
    ] = resized

    return canvas


# =============================================================================
# YOLO LABELS
# =============================================================================

def load_yolo_labels(
    label_path,
):
    """
    Read YOLO labels:

        class x_center y_center width height

    normalized [0,1].
    """

    if not label_path.exists():

        return np.zeros(
            (0, 5),
            dtype=np.float32,
        )

    labels = []

    with open(
        label_path,
        "r",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                continue

            labels.append(
                [
                    float(v)
                    for v in values
                ]
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


# =============================================================================
# DATASET
# =============================================================================

class VinDrSwinDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        image_dir,
        label_dir,
        img_size=1024,
    ):

        self.image_dir = Path(
            image_dir
        )

        self.label_dir = Path(
            label_dir
        )

        self.img_size = img_size

        self.images = sorted(
            self.image_dir.glob(
                "*.jpg"
            )
        )

        if not self.images:

            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        print(
            f"[DATASET] {self.image_dir}"
        )

        print(
            f"[DATASET] images={len(self.images)}"
        )

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index,
    ):

        image_path = self.images[
            index
        ]

        label_path = (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

        # ---------------------------------------------------------------------
        # DICOM
        # ---------------------------------------------------------------------

        image, ds = load_dicom(
            image_path
        )

        # ---------------------------------------------------------------------
        # WINDOWING
        # ---------------------------------------------------------------------

        image = preprocess_dicom(
            image,
            ds,
        )

        # ---------------------------------------------------------------------
        # ROI
        # ---------------------------------------------------------------------

        x1, y1, x2, y2 = detect_breast_roi(
            image
        )

        image = image[
            y1:y2,
            x1:x2,
        ]

        crop_h, crop_w = image.shape[:2]

        # ---------------------------------------------------------------------
        # LABELS
        # ---------------------------------------------------------------------

        labels = load_yolo_labels(
            label_path
        )

        # ---------------------------------------------------------------------
        # Transform bounding boxes into ROI coordinates
        # ---------------------------------------------------------------------

        transformed = []

        for label in labels:

            cls, xc, yc, bw, bh = label

            # Original coordinates
            box_xc = xc * (
                ds.Columns
                if hasattr(ds, "Columns")
                else image.shape[1]
            )

            box_yc = yc * (
                ds.Rows
                if hasattr(ds, "Rows")
                else image.shape[0]
            )

            box_w = bw * (
                ds.Columns
                if hasattr(ds, "Columns")
                else image.shape[1]
            )

            box_h = bh * (
                ds.Rows
                if hasattr(ds, "Rows")
                else image.shape[0]
            )

            # Move to ROI coordinates
            box_xc -= x1
            box_yc -= y1

            # Check whether center is inside ROI
            if (
                box_xc < 0
                or box_xc >= crop_w
                or box_yc < 0
                or box_yc >= crop_h
            ):
                continue

            # Convert to ROI normalized coordinates
            xc_new = box_xc / crop_w
            yc_new = box_yc / crop_h
            bw_new = box_w / crop_w
            bh_new = box_h / crop_h

            transformed.append(
                [
                    cls,
                    xc_new,
                    yc_new,
                    bw_new,
                    bh_new,
                ]
            )

        # ---------------------------------------------------------------------
        # Resize + pad
        # ---------------------------------------------------------------------

        original_h, original_w = image.shape[:2]

        scale = min(
            self.img_size / original_h,
            self.img_size / original_w,
        )

        new_h = int(
            round(original_h * scale)
        )

        new_w = int(
            round(original_w * scale)
        )

        image = cv2.resize(
            image,
            (
                new_w,
                new_h,
            ),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.zeros(
            (
                self.img_size,
                self.img_size,
            ),
            dtype=np.uint8,
        )

        left = (
            self.img_size - new_w
        ) // 2

        top = (
            self.img_size - new_h
        ) // 2

        canvas[
            top:top + new_h,
            left:left + new_w,
        ] = image

        # ---------------------------------------------------------------------
        # Transform boxes for padding
        # ---------------------------------------------------------------------

        final_labels = []

        for (
            cls,
            xc,
            yc,
            bw,
            bh,
        ) in transformed:

            xc_px = (
                xc * new_w
                + left
            )

            yc_px = (
                yc * new_h
                + top
            )

            bw_px = (
                bw * new_w
            )

            bh_px = (
                bh * new_h
            )

            final_labels.append(
                [
                    cls,
                    xc_px / self.img_size,
                    yc_px / self.img_size,
                    bw_px / self.img_size,
                    bh_px / self.img_size,
                ]
            )

        # ---------------------------------------------------------------------
        # Tensor
        # ---------------------------------------------------------------------

        image_tensor = torch.from_numpy(
            canvas
        ).float() / 255.0

        image_tensor = image_tensor.unsqueeze(
            0
        )

        # 1 channel -> 3 channels
        image_tensor = image_tensor.repeat(
            3,
            1,
            1,
        )

        if final_labels:

            target = torch.tensor(
                final_labels,
                dtype=torch.float32,
            )

        else:

            target = torch.zeros(
                (0, 5),
                dtype=torch.float32,
            )

        return (
            image_tensor,
            target,
            str(image_path),
        )


# =============================================================================
# COLLATE
# =============================================================================

def collate_fn(
    batch,
):

    images = torch.stack(
        [
            item[0]
            for item in batch
        ]
    )

    targets = [
        item[1]
        for item in batch
    ]

    paths = [
        item[2]
        for item in batch
    ]

    return (
        images,
        targets,
        paths,
    )


# =============================================================================
# SWIN BACKBONE
# =============================================================================

class SwinBackbone(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        print()
        print(
            "=" * 75
        )

        print(
            "CREATING SWIN BACKBONE"
        )

        print(
            "=" * 75
        )

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(
                0,
                1,
                2,
            ),
            img_size=IMG_SIZE,
        )

        print(
            "[SWIN] Trainable."
        )

        # Channels of Swin Tiny
        self.out_channels = (
            96,
            192,
            384,
        )

    def forward(
        self,
        x,
    ):

        features = self.model(
            x
        )

        outputs = []

        for feat in features:

            # timm Swin returns NHWC
            if feat.shape[-1] > feat.shape[1]:

                feat = feat.permute(
                    0,
                    3,
                    1,
                    2,
                ).contiguous()

            outputs.append(
                feat
            )

        return outputs


# =============================================================================
# YOLO DETECTOR
# =============================================================================

class SwinYOLO(
    nn.Module
):

    def __init__(
        self,
        num_classes=1,
    ):

        super().__init__()

        self.backbone = SwinBackbone()

        # YOLOv8 Detect head
        self.detect = Detect(
            nc=num_classes,
            ch=self.backbone.out_channels,
        )

        # Swin stages correspond to:
        #
        # 1024 -> 256 -> stride 4
        # 1024 -> 128 -> stride 8
        # 1024 ->  64 -> stride 16
        #
        self.detect.stride = torch.tensor(
            [4.0, 8.0, 16.0]
        )

        print(
            "[YOLO] Detection head created."
        )

        print(
            f"[YOLO] classes={num_classes}"
        )

    def forward(
        self,
        x,
    ):

        features = self.backbone(
            x
        )

        return self.detect(
            features
        )


# =============================================================================
# ULTRALYTICS LOSS WRAPPER
# =============================================================================

class LossModel:

    def __init__(
        self,
        model,
    ):

        self.model = model

        # Dummy configuration object required by
        # Ultralytics v8DetectionLoss.
        class Args:
            box = 7.5
            cls = 0.5
            dfl = 1.5

        self.args = Args()

        # Attach attributes expected by loss
        self.model.model = [
            None,
            None,
            self.model.detect,
        ]

        self.model.args = self.args

        self.criterion = v8DetectionLoss(
            self.model
        )

    def __call__(
        self,
        preds,
        targets,
    ):

        return self.criterion(
            preds,
            targets,
        )


# =============================================================================
# TARGET CONVERSION
# =============================================================================

def build_targets(
    targets,
    device,
):
    """
    Convert list of per-image YOLO labels into the format expected
    by Ultralytics v8DetectionLoss.

    Output:

        batch_idx
        cls
        bboxes
    """

    batch_indices = []
    classes = []
    boxes = []

    for batch_idx, target in enumerate(
        targets
    ):

        if target.numel() == 0:
            continue

        target = target.to(
            device
        )

        cls = target[
            :,
            0:1,
        ]

        bbox = target[
            :,
            1:5,
        ]

        batch = torch.full(
            (
                target.shape[0],
                1,
            ),
            batch_idx,
            dtype=torch.float32,
            device=device,
        )

        batch_indices.append(
            batch
        )

        classes.append(
            cls
        )

        boxes.append(
            bbox
        )

    if not batch_indices:

        return {
            "batch_idx": torch.zeros(
                (0, 1),
                device=device,
            ),
            "cls": torch.zeros(
                (0, 1),
                device=device,
            ),
            "bboxes": torch.zeros(
                (0, 4),
                device=device,
            ),
        }

    return {
        "batch_idx": torch.cat(
            batch_indices,
            dim=0,
        ),
        "cls": torch.cat(
            classes,
            dim=0,
        ),
        "bboxes": torch.cat(
            boxes,
            dim=0,
        ),
    }


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    epoch,
    criterion,
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    start_time = time.time()

    for batch_idx, (
        images,
        targets,
        paths,
    ) in enumerate(
        loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # ---------------------------------------------------------------------
        # AMP
        # ---------------------------------------------------------------------

        with torch.cuda.amp.autocast(
            enabled=(
                USE_AMP
                and DEVICE == "cuda"
            )
        ):

            preds = model(
                images
            )

            batch_targets = build_targets(
                targets,
                DEVICE,
            )

            loss, loss_items = criterion(
                preds,
                batch_targets,
            )

            loss = (
                loss
                / ACCUMULATION_STEPS
            )

        scaler.scale(
            loss
        ).backward()

        if (
            (batch_idx + 1)
            % ACCUMULATION_STEPS
            == 0
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

        total_loss += (
            loss.item()
            * ACCUMULATION_STEPS
        )

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        if (
            batch_idx % 10 == 0
            or batch_idx
            == len(loader) - 1
        ):

            elapsed = (
                time.time()
                - start_time
            )

            avg_loss = (
                total_loss
                / (batch_idx + 1)
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"Batch "
                f"{batch_idx + 1}/{len(loader)} | "
                f"Loss "
                f"{avg_loss:.4f} | "
                f"Time "
                f"{elapsed / 60:.1f} min"
            )

    return (
        total_loss
        / len(loader)
    )


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    loss,
    path,
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "loss": loss,
        },
        path,
    )

    print(
        f"[CHECKPOINT] {path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    seed_everything(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SWIN + YOLOv8 LESION DETECTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Dataset : {DATASET}"
    )

    print(
        f"Model   : {MODEL_NAME}"
    )

    print(
        f"Input   : {IMG_SIZE}x{IMG_SIZE}"
    )

    print(
        f"Epochs  : {EPOCHS}"
    )

    print(
        f"Batch   : {BATCH_SIZE}"
    )

    print(
        f"Accum.  : {ACCUMULATION_STEPS}"
    )

    print(
        f"Device  : {DEVICE}"
    )

    print(
        "=" * 80
    )

    # =========================================================================
    # DATASETS
    # =========================================================================

    train_dataset = VinDrSwinDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        IMG_SIZE,
    )

    val_dataset = VinDrSwinDataset(
        VAL_IMAGES,
        VAL_LABELS,
        IMG_SIZE,
    )

    train_loader = torch.utils.data.DataLoader(
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

    val_loader = torch.utils.data.DataLoader(
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

    # =========================================================================
    # MODEL
    # =========================================================================

    model = SwinYOLO(
        NUM_CLASSES
    )

    model = model.to(
        DEVICE
    )

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            USE_AMP
            and DEVICE == "cuda"
        )
    )

    # =========================================================================
    # LOSS
    # =========================================================================

    criterion = LossModel(
        model
    )

    # =========================================================================
    # RESUME
    # =========================================================================

    start_epoch = 0

    if args.resume is not None:

        print(
            f"[RESUME] Loading {args.resume}"
        )

        checkpoint = torch.load(
            args.resume,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scaler.load_state_dict(
            checkpoint["scaler"]
        )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        print(
            f"[RESUME] Starting epoch "
            f"{start_epoch + 1}"
        )

    # =========================================================================
    # TRAINING
    # =========================================================================

    best_loss = float(
        "inf"
    )

    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 80
        )

        epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            criterion,
        )

        scheduler.step()

        print()
        print(
            f"[EPOCH {epoch + 1}] "
            f"loss={epoch_loss:.6f}"
        )

        print(
            f"[LR] "
            f"{optimizer.param_groups[0]['lr']:.8e}"
        )

        # ---------------------------------------------------------------------
        # Save latest
        # ---------------------------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scaler,
            epoch,
            epoch_loss,
            OUTPUT_DIR / "last.pt",
        )

        # ---------------------------------------------------------------------
        # Save best training-loss checkpoint
        # ---------------------------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                epoch_loss,
                OUTPUT_DIR / "best.pt",
            )

    print()
    print(
        "=" * 80
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"Best loss: {best_loss:.6f}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()