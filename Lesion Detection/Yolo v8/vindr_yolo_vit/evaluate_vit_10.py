import os
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

NETWORK_IMAGES = Path(
    "/home/enric/Datasets/Original/vindr/images"
)

SPLIT_ROOT = (
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_vit/vit_splits"
)

TEST_CSV = os.path.join(
    SPLIT_ROOT,
    "vindr_test.csv",
)

OUTPUT_DIR = os.path.join(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_vit",
    "evaluation_10",
)

MODEL_NAME = "google/vit-base-patch16-224"

IMAGE_SIZE = 224

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# MAMMO PREPROCESSING
# EXACT SAME IMPORTS AS ORIGINAL ViT SCRIPT
# ============================================================

from mammo_prep.io import load_dicom
from mammo_prep.windowing import preprocess_window

from mammo_prep.artifacts import (
    flip_to_standard,
    crop_breast,
    resize_long_side,
    pad_to_square,
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
# FIND DICOM
# ============================================================

def find_image(
    study_id,
    image_id,
    image_root,
):

    image_path = (
        Path(image_root)
        / str(study_id)
        / f"{image_id}.dicom"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"No s'ha trobat el DICOM:\n"
            f"{image_path}"
        )

    return image_path


# ============================================================
# DATASET
# EXACT SAME PREPROCESSING AS TRAINING SCRIPT
# ============================================================

class VinDrViTDataset(Dataset):

    def __init__(
        self,
        df,
        network_images,
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.network_images = Path(
            network_images
        )

        if not self.network_images.exists():

            raise FileNotFoundError(
                "Network image directory not found:\n"
                f"{self.network_images}"
            )

        required_columns = [
            "study_id",
            "image_id",
            "is_positive",
        ]

        for column in required_columns:

            if column not in self.df.columns:

                raise RuntimeError(
                    f"Missing column in CSV: {column}"
                )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = str(
            row["image_id"]
        )

        study_id = str(
            row["study_id"]
        )

        label = int(
            row["is_positive"]
        )

        image_path = find_image(
            study_id,
            image_id,
            self.network_images,
        )

        # ----------------------------------------------------
        # DICOM
        # EXACT ORIGINAL
        # ----------------------------------------------------

        image, ds = load_dicom(
            image_path
        )

        # ----------------------------------------------------
        # PIXEL PADDING
        # EXACT ORIGINAL
        # ----------------------------------------------------

        if hasattr(
            ds,
            "PixelPaddingValue",
        ):

            padding_value = float(
                ds.PixelPaddingValue
            )

            valid_pixels = image[
                image < padding_value
            ]

            if valid_pixels.size > 0:

                image = image.copy()

                image[
                    image >= padding_value
                ] = 0

        # ----------------------------------------------------
        # FIRST WINDOWING
        # EXACT ORIGINAL
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

            if (
                "window_width must be > 0"
                not in str(e)
                and
                "No valid pixels available "
                "for window calculation"
                not in str(e)
            ):

                raise

            print(
                f"\n[WINDOWING FALLBACK] "
                f"image_id={image_id}"
            )

            valid_pixels = image[
                image > 0
            ]

            if valid_pixels.size == 0:

                raise ValueError(
                    "No hi ha píxels vàlids "
                    "després d'eliminar el padding: "
                    f"{image_id}"
                )

            low, high = np.percentile(
                valid_pixels,
                [1, 99],
            )

            if high <= low:

                low = float(
                    valid_pixels.min()
                )

                high = float(
                    valid_pixels.max()
                )

            if high <= low:

                raise ValueError(
                    "Imatge constant després "
                    f"del preprocessing: {image_id}"
                )

            image = np.clip(
                (
                    (image - low)
                    / (high - low)
                    * 255.0
                ),
                0,
                255,
            ).astype(
                np.uint8
            )

        # ----------------------------------------------------
        # SECOND WINDOWING
        # EXACT ORIGINAL
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
        # EXACT ORIGINAL
        # ----------------------------------------------------

        image = flip_to_standard(
            image,
            ds,
        )

        # ----------------------------------------------------
        # BREAST CROP
        # EXACT ORIGINAL
        # ----------------------------------------------------

        image, _ = crop_breast(
            image
        )

        # ----------------------------------------------------
        # RESIZE LONG SIDE
        # EXACT ORIGINAL
        # ----------------------------------------------------

        image = resize_long_side(
            image,
            target_size=1536,
        )

        # ----------------------------------------------------
        # PAD TO SQUARE
        # EXACT ORIGINAL
        # ----------------------------------------------------

        image = pad_to_square(
            image
        )

        # ----------------------------------------------------
        # FINAL ViT SIZE
        # EXACT ORIGINAL
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
        # NORMALIZE 0-1
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # GRAYSCALE -> RGB
        # EXACT ORIGINAL
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
        # IMAGENET NORMALIZATION
        # EXACT ORIGINAL
        # ----------------------------------------------------

        mean = IMAGENET_MEAN.reshape(
            3,
            1,
            1,
        )

        std = IMAGENET_STD.reshape(
            3,
            1,
            1,
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
            study_id,
            image_id,
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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

    final_checkpoint = os.path.join(
        SPLIT_ROOT,
        "vit_checkpoints",
        "vit_vindr_final.pt",
    )

    phase2_checkpoint = os.path.join(
        SPLIT_ROOT,
        "vit_checkpoints",
        "best_phase2.pt",
    )

    if os.path.exists(
        final_checkpoint
    ):

        checkpoint_path = (
            final_checkpoint
        )

    elif os.path.exists(
        phase2_checkpoint
    ):

        checkpoint_path = (
            phase2_checkpoint
        )

    else:

        raise FileNotFoundError(
            "No ViT checkpoint found.\n\n"
            f"Checked:\n"
            f"  {final_checkpoint}\n"
            f"  {phase2_checkpoint}"
        )

    print()
    print(
        f"Checkpoint:\n"
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # FINAL CHECKPOINT
    # model.state_dict()
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    # Remove DataParallel prefix if present

    cleaned_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing:

        print(
            f"Missing keys: "
            f"{len(missing)}"
        )

    if unexpected:

        print(
            f"Unexpected keys: "
            f"{len(unexpected)}"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    return model, checkpoint_path


# ============================================================
# UNNORMALIZE FOR DISPLAY
# ============================================================

def tensor_to_display(
    tensor
):

    image = (
        tensor
        .detach()
        .cpu()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    image = (
        image
        * IMAGENET_STD
        + IMAGENET_MEAN
    )

    image = np.clip(
        image,
        0,
        1,
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 75)
    print("ViT — 10 IMAGE EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # TEST CSV
    # --------------------------------------------------------

    if not os.path.exists(
        TEST_CSV
    ):

        raise FileNotFoundError(
            f"Test CSV not found:\n"
            f"{TEST_CSV}"
        )

    test_df = pd.read_csv(
        TEST_CSV
    )

    print()
    print(
        f"Total test images: "
        f"{len(test_df):,}"
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        test_df[
            "is_positive"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # FIRST 10
    # --------------------------------------------------------

    test_10 = (
        test_df
        .iloc[:10]
        .copy()
    )

    print()
    print("=" * 75)
    print("10-IMAGE TEST")
    print("=" * 75)

    print(
        test_10[
            [
                "study_id",
                "image_id",
                "is_positive",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = VinDrViTDataset(
        test_10,
        NETWORK_IMAGES,
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

    model, checkpoint_path = (
        load_model()
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("INFERENCE")
    print("=" * 75)

    results = []

    with torch.no_grad():

        for i, batch in enumerate(
            loader
        ):

            (
                images,
                labels,
                study_ids,
                image_ids,
            ) = batch

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                pixel_values=images
            )

            logits = (
                outputs.logits
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            probability_finding = (
                probabilities[
                    0,
                    1,
                ].item()
            )

            prediction = (
                torch.argmax(
                    logits,
                    dim=1,
                )[0]
                .item()
            )

            ground_truth = (
                labels[0]
                .item()
            )

            correct = (
                prediction
                == ground_truth
            )

            result = {

                "index": i,

                "study_id":
                    study_ids[0],

                "image_id":
                    image_ids[0],

                "ground_truth":
                    ground_truth,

                "prediction":
                    prediction,

                "probability_finding":
                    probability_finding,

                "correct":
                    correct,
            }

            results.append(
                result
            )

            gt_text = (
                "FINDING"
                if ground_truth == 1
                else "NO_FINDING"
            )

            pred_text = (
                "FINDING"
                if prediction == 1
                else "NO_FINDING"
            )

            print()
            print(
                f"[{i + 1:02d}/10] "
                f"{image_ids[0]}"
            )

            print(
                f"    GT: "
                f"{gt_text}"
            )

            print(
                f"    Pred: "
                f"{pred_text}"
            )

            print(
                f"    P(finding): "
                f"{probability_finding:.6f}"
            )

            print(
                f"    Correct: "
                f"{'YES' if correct else 'NO'}"
            )

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # CONFUSION COUNTS
    # --------------------------------------------------------

    y_true = (
        results_df[
            "ground_truth"
        ].values
    )

    y_pred = (
        results_df[
            "prediction"
        ].values
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )

    accuracy = (
        np.mean(
            y_true == y_pred
        )
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_10_results.csv",
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CREATING VISUALIZATION")
    print("=" * 75)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 9),
    )

    axes = axes.flatten()

    for i in range(10):

        image_tensor = (
            dataset[i][0]
        )

        image = (
            tensor_to_display(
                image_tensor
            )
        )

        row = (
            results_df.iloc[i]
        )

        gt = int(
            row["ground_truth"]
        )

        pred = int(
            row["prediction"]
        )

        prob = float(
            row[
                "probability_finding"
            ]
        )

        gt_text = (
            "FINDING"
            if gt == 1
            else "NO FINDING"
        )

        pred_text = (
            "FINDING"
            if pred == 1
            else "NO FINDING"
        )

        axes[i].imshow(
            image
        )

        axes[i].set_title(
            f"GT: {gt_text}\n"
            f"Pred: {pred_text}\n"
            f"P(finding): {prob:.3f}",
            fontsize=10,
        )

        axes[i].axis(
            "off"
        )

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
    # SUMMARY
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
            "=" * 75
            + "\n\n"
        )

        f.write(
            f"Checkpoint: "
            f"{checkpoint_path}\n"
        )

        f.write(
            f"Device: "
            f"{DEVICE}\n"
        )

        f.write(
            "Images evaluated: 10\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy:.6f}\n"
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
    print("=" * 75)
    print("RESULTS — 10 IMAGES")
    print("=" * 75)

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

    print()
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Visualization: "
        f"{figure_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()