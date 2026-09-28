import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from transformers import ViTForImageClassification

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

MODEL_NAME = "google/vit-base-patch16-224"

MODEL_PATH = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_vit/vit_splits/vit_checkpoints/"
    "vit_vindr_final.pt"
)

# IMPORTANT:
# Never write anything inside Datasets/Original.
PREDICTIONS_DIR = Path(
    "/home/enric/Mammo/Lesion Detection/Yolo v8/"
    "vindr_yolo_vit/vit_predictions"
)

IMAGE_SIZE = 224

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Exact same preprocessing used during ViT training.
    """

    print("\n[1/8] Loading DICOM...")

    image, ds = load_dicom(image_path)

    # --------------------------------------------------------
    # Pixel padding
    # --------------------------------------------------------

    if hasattr(ds, "PixelPaddingValue"):

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

    # --------------------------------------------------------
    # First windowing
    # --------------------------------------------------------

    print(
        "[2/8] Windowing: "
        "breast_tissue + LINEAR..."
    )

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
            "[WARNING] Windowing fallback: "
            "percentile 1-99"
        )

        valid_pixels = image[
            image > 0
        ]

        if valid_pixels.size == 0:
            raise ValueError(
                "No valid pixels after "
                "removing padding."
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
                "Image is constant after "
                "preprocessing."
            )

        image = np.clip(
            (
                (image - low)
                / (high - low)
                * 255.0
            ),
            0,
            255,
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Second windowing
    #
    # This is intentionally kept because the original
    # training code applies breast_tissue + LINEAR again.
    # --------------------------------------------------------

    print(
        "[3/8] Second windowing pass..."
    )

    image = preprocess_window(
        image,
        dicom_dataset=ds,
        method="breast_tissue",
        voi_func="LINEAR",
        exclude_background=True,
        output_dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Standard orientation
    # --------------------------------------------------------

    print(
        "[4/8] Standard orientation..."
    )

    image = flip_to_standard(
        image,
        ds,
    )

    # --------------------------------------------------------
    # Breast crop
    # --------------------------------------------------------

    print(
        "[5/8] Breast crop..."
    )

    image, _ = crop_breast(
        image
    )

    # --------------------------------------------------------
    # Resize long side
    # --------------------------------------------------------

    print(
        "[6/8] Resize long side → 1536..."
    )

    image = resize_long_side(
        image,
        target_size=1536,
    )

    # --------------------------------------------------------
    # Pad + final resize
    # --------------------------------------------------------

    print(
        "[7/8] Pad to square + "
        "resize → 224..."
    )

    image = pad_to_square(
        image
    )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # Normalize 0-1
    # --------------------------------------------------------

    image = (
        image.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Grayscale → RGB
    # --------------------------------------------------------

    image = np.stack(
        [
            image,
            image,
            image,
        ],
        axis=0,
    )

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    mean = IMAGENET_MEAN.reshape(
        3, 1, 1
    )

    std = IMAGENET_STD.reshape(
        3, 1, 1
    )

    image = (
        image - mean
    ) / std

    # --------------------------------------------------------
    # Torch tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image
    ).float()

    return tensor, image


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("\nLoading ViT...")

    # --------------------------------------------------------
    # Load original pretrained ViT
    #
    # This loads the official ImageNet model with its
    # original 1000-class classifier.
    # --------------------------------------------------------

    model = (
        ViTForImageClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Replace classifier with the 2-class classifier used
    # during our ViNDR training.
    # --------------------------------------------------------

    model.classifier = nn.Linear(
        model.config.hidden_size,
        2,
    )

    model.config.num_labels = 2

    model.config.id2label = {
        0: "no_finding",
        1: "finding",
    }

    model.config.label2id = {
        "no_finding": 0,
        "finding": 1,
    }

    # --------------------------------------------------------
    # Load our trained ViNDR weights
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Model loaded:\n{MODEL_PATH}"
    )

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

    return model


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    image_tensor,
):

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.amp.autocast(
        device_type="cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    ):

        outputs = model(
            pixel_values=image_tensor
        )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    return (
        prediction,
        probabilities.cpu().numpy(),
        logits.cpu().numpy()[0],
    )


# ============================================================
# SAVE PREPROCESSED IMAGE
# ============================================================

def save_preprocessed_image(
    processed,
    image_path,
):

    # Create output directory.
    # NEVER create anything inside the original dataset.

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Undo ImageNet normalization
    # --------------------------------------------------------

    image = (
        processed
        * IMAGENET_STD.reshape(
            3, 1, 1
        )
        + IMAGENET_MEAN.reshape(
            3, 1, 1
        )
    )

    image = np.clip(
        image,
        0,
        1,
    )

    # Take first channel because
    # all three RGB channels are identical.

    image = (
        image[0] * 255
    ).astype(np.uint8)

    output_path = (
        PREDICTIONS_DIR
        / f"{image_path.stem}_vit_input.png"
    )

    cv2.imwrite(
        str(output_path),
        image,
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ViT ViNDR single-image "
            "prediction"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to DICOM image",
    )

    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help=(
            "Save the 224x224 image "
            "fed to the ViT"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"DICOM not found:\n"
            f"{image_path}"
        )

    if image_path.suffix.lower() != ".dicom":

        print(
            "[WARNING] Input file does not "
            "have .dicom extension."
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "ViT ViNDR SINGLE IMAGE PREDICTION"
    )
    print("=" * 75)

    print(
        f"\nInput image:\n"
        f"{image_path}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_tensor, processed = (
        preprocess_image(
            image_path
        )
    )

    # --------------------------------------------------------
    # SAVE PREPROCESSED
    # --------------------------------------------------------

    if args.save_preprocessed:

        output_path = (
            save_preprocessed_image(
                processed,
                image_path,
            )
        )

        print(
            "\nPreprocessed image saved:"
        )

        print(
            output_path
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print(
        "\n[8/8] Running prediction..."
    )

    (
        prediction,
        probabilities,
        logits,
    ) = predict(
        model,
        image_tensor,
    )

    p_no_finding = (
        probabilities[0]
    )

    p_finding = (
        probabilities[1]
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("PREDICTION")
    print("=" * 75)

    print(
        f"\nP(no_finding): "
        f"{p_no_finding:.4f}"
    )

    print(
        f"P(finding):    "
        f"{p_finding:.4f}"
    )

    if prediction == 1:

        prediction_text = "FINDING"

    else:

        prediction_text = "NO FINDING"

    print(
        f"\nPrediction:     "
        f"{prediction_text}"
    )

    print(
        f"Confidence:     "
        f"{probabilities[prediction] * 100:.2f}%"
    )

    print(
        f"\nLogit no_finding: "
        f"{logits[0]:.4f}"
    )

    print(
        f"Logit finding:    "
        f"{logits[1]:.4f}"
    )

    print()
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()