#!/usr/bin/env python3

import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification

# ============================================================
# CONFIG
# ============================================================

NETWORK_IMAGES = "/home/enric/Datasets/Original/vindr/images"

SPLIT_ROOT = "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit/vit_splits"

TEST_CSV = os.path.join(SPLIT_ROOT, "vindr_test.csv")

OUTPUT_DIR = os.path.join(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/vindr_yolo_vit",
    "evaluation_10"
)

MODEL_NAME = "google/vit-base-patch16-224"

CHECKPOINT_FINAL = os.path.join(
    SPLIT_ROOT,
    "vit_checkpoints",
    "vit_vindr_final.pt"
)

CHECKPOINT_PHASE2 = os.path.join(
    SPLIT_ROOT,
    "vit_checkpoints",
    "best_phase2.pt"
)

IMAGE_SIZE = 224

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# IMPORT MAMMO PREPROCESSING
# ============================================================

sys.path.insert(0, "/home/enric/Mammo")

from mammo_prep.io import load_dicom
from mammo_prep.windowing import preprocess_window
from mammo_prep.orientation import flip_to_standard
from mammo_prep.crop import crop_breast
from mammo_prep.resize import resize_long_side
from mammo_prep.padding import pad_to_square


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class VinDrViTDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        study_id = row["study_id"]
        image_id = row["image_id"]
        label = int(row["is_positive"])

        dicom_path = (
            Path(NETWORK_IMAGES)
            / str(study_id)
            / f"{image_id}.dicom"
        )

        # ----------------------------------------------------
        # LOAD DICOM
        # ----------------------------------------------------

        image = load_dicom(str(dicom_path))

        image = image.astype(np.float32)

        # ----------------------------------------------------
        # PIXEL PADDING
        # ----------------------------------------------------

        if hasattr(image, "dtype"):
            pass

        # ----------------------------------------------------
        # WINDOWING
        # Same preprocessing as training script
        # ----------------------------------------------------

        try:

            image = preprocess_window(
                image,
                method="breast_tissue",
                voi_func="LINEAR",
                exclude_background=True,
                output_dtype=np.uint8,
            )

        except Exception as e:

            raise RuntimeError(
                f"Windowing failed for {dicom_path}\n"
                f"Error: {e}"
            )

        # ----------------------------------------------------
        # FLIP TO STANDARD
        # ----------------------------------------------------

        image = flip_to_standard(image)

        # ----------------------------------------------------
        # CROP BREAST
        # ----------------------------------------------------

        image = crop_breast(image)

        # ----------------------------------------------------
        # RESIZE LONG SIDE
        # ----------------------------------------------------

        image = resize_long_side(
            image,
            target_size=1536
        )

        # ----------------------------------------------------
        # PAD TO SQUARE
        # ----------------------------------------------------

        image = pad_to_square(image)

        # ----------------------------------------------------
        # RESIZE TO ViT INPUT
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------
        # UINT8 -> FLOAT
        # ----------------------------------------------------

        image = image.astype(np.float32) / 255.0

        # ----------------------------------------------------
        # GRAYSCALE -> RGB
        # ----------------------------------------------------

        if image.ndim == 2:

            image = np.stack(
                [image, image, image],
                axis=-1
            )

        elif image.ndim == 3 and image.shape[-1] == 1:

            image = np.repeat(
                image,
                3,
                axis=-1
            )

        # ----------------------------------------------------
        # IMAGENET NORMALIZATION
        # ----------------------------------------------------

        image = (image - MEAN) / STD

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image_tensor = torch.from_numpy(
            image.astype(np.float32)
        )

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.long),
            str(study_id),
            str(image_id),
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model():

    model = ViTForImageClassification.from_pretrained(
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

    # Prefer final model
    if os.path.exists(CHECKPOINT_FINAL):

        checkpoint_path = CHECKPOINT_FINAL

    elif os.path.exists(CHECKPOINT_PHASE2):

        checkpoint_path = CHECKPOINT_PHASE2

    else:

        raise FileNotFoundError(
            "No ViT checkpoint found.\n\n"
            f"Checked:\n"
            f"  {CHECKPOINT_FINAL}\n"
            f"  {CHECKPOINT_PHASE2}"
        )

    print()
    print("=" * 70)
    print("CHECKPOINT")
    print("=" * 70)
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Final checkpoint:
    # model.state_dict()
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        print(f"Missing keys: {len(missing)}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    return model, checkpoint_path


# ============================================================
# UNNORMALIZE IMAGE FOR DISPLAY
# ============================================================

def tensor_to_display_image(tensor):

    image = tensor.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = image * STD + MEAN

    image = np.clip(
        image,
        0,
        1,
    )

    return image


# ============================================================
# EVALUATE 10 IMAGES
# ============================================================

def evaluate_10():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD TEST CSV
    # --------------------------------------------------------

    if not os.path.exists(TEST_CSV):

        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    df = pd.read_csv(TEST_CSV)

    required_columns = [
        "study_id",
        "image_id",
        "is_positive",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in test CSV: {missing_columns}"
        )

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    print(f"Total test images: {len(df)}")

    print()
    print("Class distribution:")

    print(
        df["is_positive"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # FIRST 10 IMAGES
    # --------------------------------------------------------

    df_10 = df.iloc[:10].copy()

    print()
    print("=" * 70)
    print("10-IMAGE TEST")
    print("=" * 70)

    print(
        df_10[
            [
                "study_id",
                "image_id",
                "is_positive",
            ]
        ].to_string(index=False)
    )

    dataset = VinDrViTDataset(
        df_10
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, checkpoint_path = load_model()

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("INFERENCE")
    print("=" * 70)

    with torch.no_grad():

        for i, batch in enumerate(loader):

            images, labels, study_ids, image_ids = batch

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(DEVICE)

            outputs = model(
                pixel_values=images
            )

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            positive_probability = (
                probabilities[:, 1]
                .item()
            )

            prediction = torch.argmax(
                logits,
                dim=1,
            ).item()

            ground_truth = labels.item()

            correct = (
                prediction == ground_truth
            )

            result = {
                "index": i,
                "study_id": study_ids[0],
                "image_id": image_ids[0],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "probability_finding": positive_probability,
                "correct": correct,
            }

            results.append(result)

            gt_name = (
                "finding"
                if ground_truth == 1
                else "no_finding"
            )

            pred_name = (
                "finding"
                if prediction == 1
                else "no_finding"
            )

            print()
            print(
                f"[{i + 1:02d}/10] "
                f"{image_ids[0]}"
            )

            print(
                f"    GT:       {gt_name}"
            )

            print(
                f"    Pred:     {pred_name}"
            )

            print(
                f"    P(finding): "
                f"{positive_probability:.6f}"
            )

            print(
                f"    Correct:  "
                f"{'YES' if correct else 'NO'}"
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_10_results.csv",
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    y_true = results_df[
        "ground_truth"
    ].values

    y_pred = results_df[
        "prediction"
    ].values

    accuracy = (
        np.mean(
            y_true == y_pred
        )
    )

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    print()
    print("=" * 70)
    print("RESULTS — 10 IMAGES")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"TP: {tp}"
    )

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 9),
    )

    axes = axes.flatten()

    for i in range(10):

        image_tensor = dataset[i][0]

        image = tensor_to_display_image(
            image_tensor
        )

        row = results_df.iloc[i]

        gt = int(
            row["ground_truth"]
        )

        pred = int(
            row["prediction"]
        )

        prob = float(
            row["probability_finding"]
        )

        gt_name = (
            "FINDING"
            if gt == 1
            else "NO FINDING"
        )

        pred_name = (
            "FINDING"
            if pred == 1
            else "NO FINDING"
        )

        axes[i].imshow(
            image,
            cmap="gray",
        )

        axes[i].set_title(
            f"GT: {gt_name}\n"
            f"Pred: {pred_name}\n"
            f"P(finding): {prob:.3f}",
            fontsize=10,
        )

        axes[i].axis("off")

    plt.tight_layout()

    figure_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_10_visualization.png",
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # SUMMARY FILE
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_10_summary.txt",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ViT — 10 IMAGE EVALUATION\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Checkpoint: {checkpoint_path}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Images evaluated: 10\n\n"
        )

        f.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"TP: {tp}\n"
        )

        f.write(
            f"TN: {tn}\n"
        )

        f.write(
            f"FP: {fp}\n"
        )

        f.write(
            f"FN: {fn}\n"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Results:       {csv_path}"
    )

    print(
        f"Visualization: {figure_path}"
    )

    print(
        f"Summary:       {summary_path}"
    )

    print()
    print(
        "Si aquestes 10 imatges són correctes, "
        "el següent pas serà l'avaluació completa del test."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed(SEED)

    evaluate_10()