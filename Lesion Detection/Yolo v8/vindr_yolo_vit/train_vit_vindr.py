
import os
import time
import copy
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tqdm import tqdm

from mammo_prep.io import load_dicom
from mammo_prep.windowing import preprocess_window
from mammo_prep.artifacts import (
    flip_to_standard,
    crop_breast,
    resize_long_side,
    pad_to_square,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"

SPLIT_ROOT = "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/vit_splits"

TRAIN_CSV = os.path.join(
    SPLIT_ROOT,
    "vindr_train.csv",
)

VAL_CSV = os.path.join(
    SPLIT_ROOT,
    "vindr_val.csv",
)

TEST_CSV = os.path.join(
    SPLIT_ROOT,
    "vindr_test.csv",
)

OUTPUT_DIR = os.path.join(
    SPLIT_ROOT,
    "vit_checkpoints",
)

MODEL_NAME = "google/vit-base-patch16-224"

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS_PHASE1 = 5
NUM_EPOCHS_PHASE2 = 10

LR_PHASE1 = 1e-4
LR_PHASE2 = 5e-5

WEIGHT_DECAY = 0.01

NUM_WORKERS = 8

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ImageNet normalization used by the pretrained ViT
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

# ============================================================
# FIND IMAGE
# ============================================================
def find_image(image_id, image_root):

    matches = list(
        image_root.glob(f"*/{image_id}.jpg")
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No s'ha trobat la imatge {image_id} a {image_root}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"S'han trobat múltiples imatges per {image_id}: {matches}"
        )

    return matches[0]

# ============================================================
# DATASET
# ============================================================

class VinDrViTDataset(Dataset):

    def __init__(
        self,
        df,
        image_dir,
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.image_dir = Path(image_dir)

    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["image_id"]

        label = int(
            row["is_positive"]
        )

        image_path = find_image(
            image_id,
            self.image_dir
        )

        # ----------------------------------------------------
        # DICOM
        # ----------------------------------------------------

        image, ds = load_dicom(
            image_path
        )

        # ----------------------------------------------------
        # WINDOWING
        # breast_tissue + LINEAR
        # ----------------------------------------------------

        try:
            image = preprocess_window(
                image,
                dicom_dataset=ds,
                method="breast_tissue",
                voi_func="LINEAR",
                exclude_background=True,
                output_dtype=np.uint8,
            )

        except ValueError as e:
            print(
                f"\n[WINDOWING ERROR] "
                f"image_id={image_id}: {e}"
            )
            raise

        # ----------------------------------------------------
        # STANDARD ORIENTATION
        # ----------------------------------------------------

        image = flip_to_standard(
            image,
            ds,
        )

        # ----------------------------------------------------
        # BREAST CROP
        # ----------------------------------------------------

        image, _ = crop_breast(
            image
        )

        # ----------------------------------------------------
        # RESIZE LONG SIDE
        # ----------------------------------------------------

        image = resize_long_side(
            image,
            target_size=1536,
        )

        # ----------------------------------------------------
        # PAD TO SQUARE
        # ----------------------------------------------------

        image = pad_to_square(
            image
        )

        # ----------------------------------------------------
        # FINAL ViT SIZE
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------
        # 0-1
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # GRAYSCALE -> RGB
        # ----------------------------------------------------

        image = np.stack(
            [
                image,
                image,
                image,
            ],
            axis=0,
        )

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        mean = (
            IMAGENET_MEAN
            .reshape(3, 1, 1)
        )

        std = (
            IMAGENET_STD
            .reshape(3, 1, 1)
        )

        image = (
            image - mean
        ) / std

        # ----------------------------------------------------
        # TORCH
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return (
            image,
            label,
        )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true,
    y_probs,
    y_pred,
):

    metrics = {}

    metrics["auc"] = roc_auc_score(
        y_true,
        y_probs,
    )

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return metrics


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch,
    total_epochs,
):

    model.train()

    running_loss = 0.0
    samples_seen = 0

    progress = tqdm(
        dataloader,
        desc=(
            f"Epoch {epoch}/{total_epochs} "
            "[TRAIN]"
        ),
        unit="batch",
        dynamic_ncols=True,
    )

    for step, batch in enumerate(
        progress
    ):

        pixel_values, labels = batch

        pixel_values = pixel_values.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda"),
        ):

            outputs = model(
                pixel_values=pixel_values
            )

            logits = outputs.logits

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size_actual = pixel_values.size(0)

        running_loss += (
            loss.item()
            * batch_size_actual
        )

        samples_seen += batch_size_actual

        epoch_loss = (
            running_loss
            / samples_seen
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{epoch_loss:.4f}",
            lr=(
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ),
        )

    epoch_loss = (
        running_loss
        / len(dataloader.dataset)
    )

    return epoch_loss


# ============================================================
# VALIDATION / TEST
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    description="VAL",
):

    model.eval()

    running_loss = 0.0

    y_true = []
    y_probs = []
    y_pred = []

    progress = tqdm(
        dataloader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress:

        pixel_values, labels = batch

        pixel_values = pixel_values.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda"),
        ):

            outputs = model(
                pixel_values=pixel_values
            )

            logits = outputs.logits

            loss = criterion(
                logits,
                labels,
            )

        running_loss += (
            loss.item()
            * pixel_values.size(0)
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        preds = torch.argmax(
            logits,
            dim=1,
        )

        y_true.extend(
            labels.cpu().numpy()
        )

        y_probs.extend(
            probs.cpu().numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(dataloader.dataset)
    )

    metrics = compute_metrics(
        y_true,
        y_probs,
        y_pred,
    )

    return (
        epoch_loss,
        metrics,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_auc,
    phase,
):

    checkpoint = {

        "epoch": epoch,

        "phase": phase,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_auc":
            best_val_auc,

    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    batch_size = args.batch

    workers = args.workers

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 75)
    print("ViT ViNDR BINARY CLASSIFICATION")
    print("=" * 75)

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

    print()
    print("Configuration:")
    print(
        f"  Model: {MODEL_NAME}"
    )
    print(
        f"  Image size: {IMAGE_SIZE}"
    )
    print(
        f"  Batch size: {batch_size}"
    )
    print(
        f"  Workers: {workers}"
    )
    print(
        f"  Phase 1 epochs: {NUM_EPOCHS_PHASE1}"
    )
    print(
        f"  Phase 2 epochs: {NUM_EPOCHS_PHASE2}"
    )
    print(
        f"  Phase 1 LR: {LR_PHASE1}"
    )
    print(
        f"  Phase 2 LR: {LR_PHASE2}"
    )

    print()
    print("Preprocessing:")
    print(
        "  DICOM"
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
        "  Flip standard"
    )
    print(
        "    ↓"
    )
    print(
        "  Breast crop"
    )
    print(
        "    ↓"
    )
    print(
        "  Resize long side → 1536"
    )
    print(
        "    ↓"
    )
    print(
        "  Pad square"
    )
    print(
        "    ↓"
    )
    print(
        "  Resize → 224 × 224"
    )
    print(
        "    ↓"
    )
    print(
        "  Grayscale → RGB"
    )
    print(
        "    ↓"
    )
    print(
        "  ImageNet normalization"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("LOADING DATA")
    print("=" * 75)

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"Train images: "
        f"{len(train_df):,}"
    )

    print(
        f"Val images:   "
        f"{len(val_df):,}"
    )

    print(
        f"Test images:  "
        f"{len(test_df):,}"
    )

    print()
    print("Train labels:")

    print(
        train_df[
            "is_positive"
        ].value_counts()
        .sort_index()
    )

    print()
    print("Validation labels:")

    print(
        val_df[
            "is_positive"
        ].value_counts()
        .sort_index()
    )

    print()
    print("Test labels:")

    print(
        test_df[
            "is_positive"
        ].value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # IMAGE DIRECTORIES
    # --------------------------------------------------------

    # The ViT CSV splits do not necessarily match the physical YOLO folders.
    # Use the common images root; find_image() searches train/val/test.
    image_root = os.path.join(
        DATASET_ROOT,
        "images",
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print()
    print(
        "Creating datasets..."
    )

    train_dataset = VinDrViTDataset(
        train_df,
        image_root,
    )

    val_dataset = VinDrViTDataset(
        val_df,
        image_root,
    )

    test_dataset = VinDrViTDataset(
        test_df,
        image_root,
    )

    print(
        "Datasets ready."
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    print()
    print(
        "Creating DataLoaders..."
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            workers > 0
        ),
        drop_last=False,
    )

    print(
        "DataLoaders ready."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("LOADING ViT")
    print("=" * 75)

    model = (
        ViTForImageClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={
                0: "no_finding",
                1: "finding",
            },
            label2id={
                "no_finding": 0,
                "finding": 1,
            },
            ignore_mismatched_sizes=True,
        )
    )

    model = model.to(
        DEVICE
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_params / 1e6:.2f} M"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    train_counts = (
        train_df[
            "is_positive"
        ]
        .value_counts()
        .sort_index()
    )

    n_negative = int(
        train_counts.get(
            0,
            0,
        )
    )

    n_positive = int(
        train_counts.get(
            1,
            0,
        )
    )

    # Balanced weights:
    # N / (2 * class_count)

    weight_negative = (
        len(train_df)
        / (
            2
            * n_negative
        )
    )

    weight_positive = (
        len(train_df)
        / (
            2
            * n_positive
        )
    )

    class_weights = torch.tensor(
        [
            weight_negative,
            weight_positive,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("Class weights:")

    print(
        f"  Negative: "
        f"{weight_negative:.4f}"
    )

    print(
        f"  Positive: "
        f"{weight_positive:.4f}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    )

    # ========================================================
    # PHASE 1
    # ========================================================

    print()
    print("=" * 75)
    print("PHASE 1: CLASSIFIER ONLY")
    print("=" * 75)

    for param in model.vit.parameters():

        param.requires_grad = False

    for param in model.classifier.parameters():

        param.requires_grad = True

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params / 1e6:.4f} M"
    )

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LR_PHASE1,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = 0.0

    best_model_wts = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(
        1,
        NUM_EPOCHS_PHASE1 + 1,
    ):

        epoch_start = time.time()

        print()
        print(
            f"PHASE 1 — EPOCH "
            f"{epoch}/{NUM_EPOCHS_PHASE1}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            scaler,
            epoch,
            NUM_EPOCHS_PHASE1,
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            description="Validation",
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            "-" * 75
        )

        print(
            f"Train Loss: "
            f"{train_loss:.5f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.5f}"
        )

        print(
            f"Val AUC:    "
            f"{val_metrics['auc']:.5f}"
        )

        print(
            f"Val Recall: "
            f"{val_metrics['recall']:.5f}"
        )

        print(
            f"Val Prec.:  "
            f"{val_metrics['precision']:.5f}"
        )

        print(
            f"Val F1:     "
            f"{val_metrics['f1']:.5f}"
        )

        print(
            f"Time:       "
            f"{epoch_time / 60:.2f} min"
        )

        print(
            "-" * 75
        )

        if (
            val_metrics["auc"]
            > best_val_auc
        ):

            best_val_auc = (
                val_metrics["auc"]
            )

            best_model_wts = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                os.path.join(
                    OUTPUT_DIR,
                    "best_phase1.pt",
                ),
                model,
                optimizer,
                epoch,
                best_val_auc,
                "phase1",
            )

            print(
                f"[CHECKPOINT] "
                f"New best Phase 1 "
                f"AUC={best_val_auc:.5f}"
            )

    model.load_state_dict(
        best_model_wts
    )

    print()
    print(
        f"Best Phase 1 Val AUC: "
        f"{best_val_auc:.5f}"
    )

    # ========================================================
    # PHASE 2
    # ========================================================

    print()
    print("=" * 75)
    print("PHASE 2: FULL FINE-TUNING")
    print("=" * 75)

    for param in model.parameters():

        param.requires_grad = True

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params / 1e6:.4f} M"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PHASE2,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = 0.0

    best_model_wts = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(
        1,
        NUM_EPOCHS_PHASE2 + 1,
    ):

        epoch_start = time.time()

        print()
        print(
            f"PHASE 2 — EPOCH "
            f"{epoch}/{NUM_EPOCHS_PHASE2}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            scaler,
            epoch,
            NUM_EPOCHS_PHASE2,
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            description="Validation",
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            "-" * 75
        )

        print(
            f"Train Loss: "
            f"{train_loss:.5f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.5f}"
        )

        print(
            f"Val AUC:    "
            f"{val_metrics['auc']:.5f}"
        )

        print(
            f"Val Recall: "
            f"{val_metrics['recall']:.5f}"
        )

        print(
            f"Val Prec.:  "
            f"{val_metrics['precision']:.5f}"
        )

        print(
            f"Val F1:     "
            f"{val_metrics['f1']:.5f}"
        )

        print(
            f"Time:       "
            f"{epoch_time / 60:.2f} min"
        )

        print(
            "-" * 75
        )

        if (
            val_metrics["auc"]
            > best_val_auc
        ):

            best_val_auc = (
                val_metrics["auc"]
            )

            best_model_wts = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                os.path.join(
                    OUTPUT_DIR,
                    "best_phase2.pt",
                ),
                model,
                optimizer,
                epoch,
                best_val_auc,
                "phase2",
            )

            print(
                f"[CHECKPOINT] "
                f"New best Phase 2 "
                f"AUC={best_val_auc:.5f}"
            )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_model_wts
    )

    print()
    print(
        "=" * 75
    )

    print(
        f"Best Phase 2 Val AUC: "
        f"{best_val_auc:.5f}"
    )

    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL TEST")
    print("=" * 75)

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        description="Test",
    )

    print()
    print(
        "FINAL TEST RESULTS"
    )

    print(
        f"Test Loss:    "
        f"{test_loss:.5f}"
    )

    print(
        f"Test AUC:     "
        f"{test_metrics['auc']:.5f}"
    )

    print(
        f"Test Accuracy:"
        f" {test_metrics['accuracy']:.5f}"
    )

    print(
        f"Test Recall:  "
        f"{test_metrics['recall']:.5f}"
    )

    print(
        f"Test Precision:"
        f" {test_metrics['precision']:.5f}"
    )

    print(
        f"Test F1:       "
        f"{test_metrics['f1']:.5f}"
    )

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    final_path = os.path.join(
        OUTPUT_DIR,
        "vit_vindr_final.pt",
    )

    torch.save(
        model.state_dict(),
        final_path,
    )

    print()
    print(
        f"Final model saved:"
    )

    print(
        final_path
    )

    print()
    print("=" * 75)
    print("TRAINING FINISHED")
    print("=" * 75)


if __name__ == "__main__":

    main()