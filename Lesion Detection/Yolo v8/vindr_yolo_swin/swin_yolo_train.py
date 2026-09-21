#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm

from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

OUTPUT_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/train"
)

MODEL_NAME = "swin_tiny_patch4_window7_224"

NUM_CLASSES = 1

IMG_SIZE = 1024

BATCH_SIZE = 1
ACCUMULATION_STEPS = 4

EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

# Swin-Tiny stages used as YOLO P3/P4/P5
#
# Stage 1 -> stride 8  -> 192 channels
# Stage 2 -> stride 16 -> 384 channels
# Stage 3 -> stride 32 -> 768 channels
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

        # Translate to ROI
        x_min -= roi_x1
        x_max -= roi_x1

        y_min -= roi_y1
        y_max -= roi_y1

        # Clip
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

class VindrSwinDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        img_size=1024,
    ):

        self.root = Path(root)
        self.split = split
        self.img_size = img_size

        self.image_dir = (
            self.root
            / "images"
            / split
        )

        self.label_dir = (
            self.root
            / "labels"
            / split
        )

        # IMPORTANT:
        # These .jpg files are symlinks to DICOM files.
        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

        if not self.images:

            raise RuntimeError(
                f"No images found in "
                f"{self.image_dir}"
            )

        print(
            f"[{split}] images: "
            f"{len(self.images)}"
        )

    def __len__(self):

        return len(self.images)

    def _load_labels(
        self,
        image_path,
    ):

        label_path = (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

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

    def __getitem__(
        self,
        index,
    ):

        image_path = self.images[index]

        # ----------------------------------------------------
        # DICOM
        # ----------------------------------------------------

        image = preprocess_dicom(
            image_path
        )

        # Original DICOM dimensions
        original_h, original_w = (
            image.shape
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = self._load_labels(
            image_path
        )

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        image, resize_scale = (
            resize_keep_aspect(
                image,
                self.img_size,
            )
        )

        resized_h, resized_w = (
            image.shape
        )

        # ----------------------------------------------------
        # Breast ROI
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            detect_breast_roi(
                image
            )
        )

        crop = image[
            y1:y2,
            x1:x2,
        ]

        crop_h, crop_w = (
            crop.shape
        )

        # ----------------------------------------------------
        # Labels -> ROI
        #
        # YOLO coordinates are normalized, therefore
        # the uniform resize above does not change them.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Grayscale -> 3 channels
        # ----------------------------------------------------

        crop_bgr = cv2.cvtColor(
            crop,
            cv2.COLOR_GRAY2BGR,
        )

        # ----------------------------------------------------
        # Resize + padding
        # ----------------------------------------------------

        (
            final_image,
            scale,
            pad_x,
            pad_y,
        ) = resize_and_pad(
            crop_bgr,
            self.img_size,
        )

        # ----------------------------------------------------
        # Labels -> final image
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        final_image = (
            final_image.astype(
                np.float32
            ) / 255.0
        )

        final_image = np.transpose(
            final_image,
            (2, 0, 1),
        )

        image_tensor = (
            torch.from_numpy(
                final_image
            ).float()
        )

        target_tensor = (
            torch.from_numpy(
                labels_final
            ).float()
        )

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
# SWIN + YOLOv8
# ============================================================

class SwinYOLO(nn.Module):

    def __init__(
        self,
        img_size=1024,
        num_classes=1,
        pretrained=True,
    ):

        super().__init__()

        print(
            f"Loading {MODEL_NAME}..."
        )

        # ----------------------------------------------------
        # Swin-Tiny
        # ----------------------------------------------------

        self.backbone = (
            timm.create_model(
                MODEL_NAME,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3),
                img_size=img_size,
            )
        )

        # ----------------------------------------------------
        # Official Ultralytics Detect head
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Swin
        # ----------------------------------------------------

        features = self.backbone(x)

        swin_features = []

        for feat in features:

            # timm Swin:
            #
            # [B,H,W,C]
            #
            # YOLO:
            #
            # [B,C,H,W]

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

        # ----------------------------------------------------
        # YOLO Detect
        # ----------------------------------------------------

        predictions = self.detect(
            swin_features
        )

        return predictions


# ============================================================
# ULTRALYTICS LOSS WRAPPER
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
# LOSS HANDLING
# ============================================================

def compute_loss(
    criterion,
    predictions,
    targets,
):
    """
    Handles both possible Ultralytics behaviours:

    scalar:
        loss = tensor(...)

    vector:
        loss = tensor([box, cls, dfl])

    The vector must be summed before backward().
    """

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
        f"Loading checkpoint: "
        f"{path}"
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
        f"Resuming from epoch "
        f"{start_epoch}"
    )

    return (
        start_epoch,
        best_val_loss,
    )


# ============================================================
# OUTPUT INSPECTION
# ============================================================

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

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = VindrSwinDataset(
        DATASET,
        "train",
        IMG_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Creating model..."
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

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FIND POSITIVE SAMPLE
    # --------------------------------------------------------

    print()
    print(
        "Searching for a training image "
        "with at least one lesion..."
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
        "Positive sample found."
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

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    print()
    print(
        "Forward pass..."
    )

    model.train()

    with torch.amp.autocast(
        device_type="cuda",
        enabled=amp_enabled,
    ):

        predictions = model(
            images
        )

    print()
    print(
        "YOLO Detect output:"
    )

    inspect_prediction(
        predictions
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    print()
    print(
        "Computing YOLOv8 loss..."
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

    if isinstance(
        loss_items,
        torch.Tensor,
    ):

        print(
            "Detached loss components:"
        )

        print(
            loss_items.detach()
            .float()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    print()
    print(
        "Backward pass..."
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

    # --------------------------------------------------------
    # SWIN GRADIENTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEST PASSED"
    )
    print("=" * 70)

    print()
    print(
        "DICOM"
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

    for step, (
        images,
        targets,
        paths,
    ) in enumerate(loader):

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

        if step % 20 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"step {step:05d}/{len(loader):05d} | "
                f"loss {total_loss.item():.5f}"
            )

    return (
        running_loss
        / len(loader)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    criterion,
    loader,
):

    model.train()

    # Freeze BatchNorm statistics during validation
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

    for (
        images,
        targets,
        paths,
    ) in loader:

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

    return (
        running_loss
        / len(loader)
    )


# ============================================================
# FULL TRAINING
# ============================================================

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
        "SWIN-TINY + YOLOv8 TRAINING"
    )
    print("=" * 70)

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Image size: {img_size}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Gradient accumulation: "
        f"{ACCUMULATION_STEPS}"
    )

    print(
        "Effective batch size: "
        f"{batch_size * ACCUMULATION_STEPS}"
    )

    print(
        f"Epochs: {epochs}"
    )

    print(
        f"Workers: {workers}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = VindrSwinDataset(
        DATASET,
        "train",
        img_size,
    )

    val_dataset = VindrSwinDataset(
        DATASET,
        "val",
        img_size,
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

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

    criterion = create_loss(
        model
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_epoch = 1

    best_val_loss = float(
        "inf"
    )

    if resume is not None:

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
    # EPOCHS
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

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
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step()

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
            f"Saved: {last_path}"
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
                "New best model: "
                f"{best_path}"
            )

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

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train Swin-Tiny + YOLOv8 "
            "on ViNDR mammography."
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