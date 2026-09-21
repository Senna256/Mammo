#!/usr/bin/env python3

from pathlib import Path
import argparse

import cv2
import numpy as np
import pydicom
import torch
import timm
import matplotlib.pyplot as plt

from mammo_prep.windowing import preprocess_window


# =============================================================================
# CONFIG
# =============================================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

DICOM_DIR = DATASET / "images" / "train"

OUTPUT_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/top_channels"
)

IMAGE_NAME = "000611f8c6a44659a1813f4019241829.jpg"

MODEL_NAME = "swin_tiny_patch4_window7_224"

IMG_SIZE = 1024
SWIN_SIZE = 224

METHOD = "breast_tissue"
VOI_FUNC = "LINEAR"

DEFAULT_TOP_K = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# UTILS
# =============================================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_bgr(img):
    """
    Convert grayscale / RGB / BGR image to BGR.
    """

    img = np.asarray(img)

    if img.ndim == 2:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR,
        )

    elif img.ndim == 3:

        if img.shape[2] == 1:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR,
            )

        elif img.shape[2] == 3:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR,
            )

        else:

            raise ValueError(
                f"Unsupported number of channels: {img.shape}"
            )

    else:

        raise ValueError(
            f"Unsupported image shape: {img.shape}"
        )

    return img


def normalize_uint8(img):
    """
    Robust percentile normalization to uint8.
    """

    img = np.asarray(
        img,
        dtype=np.float32,
    )

    p1 = np.percentile(
        img,
        1,
    )

    p99 = np.percentile(
        img,
        99,
    )

    if p99 <= p1:

        p1 = float(img.min())
        p99 = float(img.max())

    if p99 <= p1:

        return np.zeros_like(
            img,
            dtype=np.uint8,
        )

    out = (
        img - p1
    ) / (
        p99 - p1
    )

    out = np.clip(
        out,
        0.0,
        1.0,
    )

    return (
        out * 255.0
    ).astype(np.uint8)


def normalize_channel(feature):
    """
    Normalize a feature map to uint8.
    """

    feature = np.asarray(
        feature,
        dtype=np.float32,
    )

    p1 = np.percentile(
        feature,
        1,
    )

    p99 = np.percentile(
        feature,
        99,
    )

    if p99 <= p1:

        p1 = float(feature.min())
        p99 = float(feature.max())

    if p99 <= p1:

        return np.zeros(
            feature.shape,
            dtype=np.uint8,
        )

    feature = (
        feature - p1
    ) / (
        p99 - p1
    )

    feature = np.clip(
        feature,
        0.0,
        1.0,
    )

    return (
        feature * 255.0
    ).astype(np.uint8)


def make_overlay(
    base,
    feature,
    alpha=0.50,
):
    """
    Create a heatmap overlay.

    Feature maps from Swin have different spatial resolutions:

        Stage 0 -> 56x56
        Stage 1 -> 28x28
        Stage 2 -> 14x14
        Stage 3 -> 7x7

    They are resized to the base image resolution before blending.
    """

    base = to_bgr(base)

    if base.dtype != np.uint8:

        base = np.clip(
            base,
            0,
            255,
        ).astype(np.uint8)

    base_h, base_w = base.shape[:2]

    feature = np.asarray(
        feature,
        dtype=np.float32,
    )

    feature = np.squeeze(feature)

    if feature.ndim != 2:

        raise ValueError(
            "Feature map must be 2D after squeeze. "
            f"Got shape={feature.shape}"
        )

    feature_resized = cv2.resize(
        feature,
        (
            base_w,
            base_h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    feature_norm = normalize_channel(
        feature_resized
    )

    heatmap = cv2.applyColorMap(
        feature_norm,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        base,
        1.0 - alpha,
        heatmap,
        alpha,
        0,
    )

    return overlay


# =============================================================================
# DICOM
# =============================================================================

def load_dicom(image_path):
    """
    Load the DICOM through the .jpg symlink.

    IMPORTANT:
    vindr_yolo/images/... contains symlinks to DICOM files.
    Therefore we use pydicom directly.
    """

    print(
        f"[IMAGE] {image_path}"
    )

    ds = pydicom.dcmread(
        str(image_path)
    )

    img = ds.pixel_array.astype(
        np.float32
    )

    print(
        f"[RAW] shape={img.shape} "
        f"min={img.min():.2f} "
        f"max={img.max():.2f}"
    )

    photometric = getattr(
        ds,
        "PhotometricInterpretation",
        "",
    )

    if photometric == "MONOCHROME1":

        img = (
            img.max()
            - img
        )

        print(
            "[DICOM] MONOCHROME1 -> inverted"
        )

    return img, ds


# =============================================================================
# WINDOWING
# =============================================================================

def apply_windowing(
    img,
    ds,
):
    """
    Apply the existing mammo_prep windowing pipeline.

    The installed preprocess_window() API is:

        preprocess_window(
            image,
            dicom_dataset=None,
            method="dicom",
            voi_func=None,
            exclude_background=True,
            output_dtype=np.uint8,
        )

    Therefore there is NO calc_window argument.

    For method='breast_tissue', mammo_prep calculates the window internally.
    """

    print(
        f"[WINDOW] method={METHOD}, "
        f"VOI={VOI_FUNC}"
    )

    windowed = preprocess_window(
        img,
        dicom_dataset=ds,
        method=METHOD,
        voi_func=VOI_FUNC,
        exclude_background=True,
        output_dtype=np.uint8,
    )

    windowed = np.asarray(
        windowed
    )

    print(
        f"[WINDOW] dtype={windowed.dtype} "
        f"min={windowed.min()} "
        f"max={windowed.max()}"
    )

    return windowed


# =============================================================================
# RESIZE
# =============================================================================

def resize_max(
    img,
    max_size=1024,
):
    """
    Resize while preserving aspect ratio.
    """

    h, w = img.shape[:2]

    scale = min(
        max_size / h,
        max_size / w,
    )

    if scale >= 1.0:

        return img.copy()

    new_h = int(
        round(h * scale)
    )

    new_w = int(
        round(w * scale)
    )

    resized = cv2.resize(
        img,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized


# =============================================================================
# BREAST ROI
# =============================================================================

def detect_breast_roi(
    img,
):
    """
    Detect breast ROI using thresholding,
    morphology and largest contour.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )

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
        (
            31,
            31,
        ),
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

        print(
            "[ROI] No contour found -> using full image"
        )

        h, w = gray.shape

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

    x1 = max(
        0,
        x,
    )

    y1 = max(
        0,
        y,
    )

    x2 = min(
        gray.shape[1],
        x + w,
    )

    y2 = min(
        gray.shape[0],
        y + h,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# =============================================================================
# RESIZE + PADDING
# =============================================================================

def resize_and_pad(
    img,
    target_size=224,
):
    """
    Resize preserving aspect ratio and pad to square.
    """

    h, w = img.shape[:2]

    scale = min(
        target_size / w,
        target_size / h,
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
        img,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    if img.ndim == 2:

        padded = np.zeros(
            (
                target_size,
                target_size,
            ),
            dtype=img.dtype,
        )

    else:

        padded = np.zeros(
            (
                target_size,
                target_size,
                img.shape[2],
            ),
            dtype=img.dtype,
        )

    pad_left = (
        target_size - new_w
    ) // 2

    pad_right = (
        target_size
        - new_w
        - pad_left
    )

    pad_top = (
        target_size - new_h
    ) // 2

    pad_bottom = (
        target_size
        - new_h
        - pad_top
    )

    padded[
        pad_top:
        pad_top + new_h,
        pad_left:
        pad_left + new_w,
    ] = resized

    valid_mask = np.zeros(
        (
            target_size,
            target_size,
        ),
        dtype=np.uint8,
    )

    valid_mask[
        pad_top:
        pad_top + new_h,
        pad_left:
        pad_left + new_w,
    ] = 1

    print(
        f"[PAD] crop={h}x{w} "
        f"-> {new_h}x{new_w}"
    )

    print(
        f"[PAD] L={pad_left}, "
        f"T={pad_top}, "
        f"R={pad_right}, "
        f"B={pad_bottom}"
    )

    return (
        padded,
        valid_mask,
    )


# =============================================================================
# ROI DEBUG
# =============================================================================

def create_roi_debug_figure(
    original,
    resized,
    roi,
    padded,
    output_path,
):
    """
    Save preprocessing debug figure.
    """

    x1, y1, x2, y2 = roi

    original_display = normalize_uint8(
        original
    )

    if original_display.ndim == 2:

        original_display = cv2.cvtColor(
            original_display,
            cv2.COLOR_GRAY2RGB,
        )

    else:

        original_display = cv2.cvtColor(
            original_display,
            cv2.COLOR_BGR2RGB,
        )

    resized_display = resized.copy()

    if resized_display.dtype != np.uint8:

        resized_display = np.clip(
            resized_display,
            0,
            255,
        ).astype(np.uint8)

    resized_rgb = cv2.cvtColor(
        resized_display,
        cv2.COLOR_BGR2RGB,
    )

    roi_debug = resized_display.copy()

    cv2.rectangle(
        roi_debug,
        (
            x1,
            y1,
        ),
        (
            x2 - 1,
            y2 - 1,
        ),
        (0, 255, 0),
        3,
    )

    roi_debug = cv2.cvtColor(
        roi_debug,
        cv2.COLOR_BGR2RGB,
    )

    padded_display = padded.copy()

    if padded_display.dtype != np.uint8:

        padded_display = np.clip(
            padded_display,
            0,
            255,
        ).astype(np.uint8)

    padded_rgb = cv2.cvtColor(
        padded_display,
        cv2.COLOR_BGR2RGB,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
    )

    axes[0].imshow(
        original_display,
        cmap="gray",
    )

    axes[0].set_title(
        "RAW"
    )

    axes[0].axis("off")

    axes[1].imshow(
        resized_rgb
    )

    axes[1].set_title(
        "RESIZED"
    )

    axes[1].axis("off")

    axes[2].imshow(
        roi_debug
    )

    axes[2].set_title(
        "BREAST ROI"
    )

    axes[2].axis("off")

    axes[3].imshow(
        padded_rgb
    )

    axes[3].set_title(
        "224x224 + PADDING"
    )

    axes[3].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# SWIN
# =============================================================================

def load_swin():

    print()
    print(
        "=" * 75
    )

    print(
        "CARREGANT SWIN"
    )

    print(
        "=" * 75
    )

    print(
        f"Model : {MODEL_NAME}"
    )

    print(
        f"Input : {SWIN_SIZE}x{SWIN_SIZE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "=" * 75
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        features_only=True,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "[SWIN] Model carregat."
    )

    return model


def prepare_swin_input(
    padded,
):
    """
    BGR uint8 -> ImageNet normalized tensor.
    """

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB,
    )

    rgb = (
        rgb.astype(
            np.float32
        ) / 255.0
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=np.float32,
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=np.float32,
    )

    rgb = (
        rgb - mean
    ) / std

    tensor = torch.from_numpy(
        rgb.transpose(
            2,
            0,
            1,
        )
    ).float()

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    return tensor


@torch.no_grad()
def extract_swin_features(
    model,
    padded,
):
    """
    Extract all four Swin feature stages.

    timm returns NHWC for this configuration.
    """

    tensor = prepare_swin_input(
        padded
    )

    features = model(
        tensor
    )

    output = []

    for stage_idx, feat in enumerate(
        features
    ):

        print(
            f"[FORMAT] Stage {stage_idx}: "
            f"{tuple(feat.shape)}"
        )

        if feat.ndim != 4:

            raise ValueError(
                f"Unexpected Swin feature shape: "
                f"{feat.shape}"
            )

        # NHWC -> NCHW
        if feat.shape[-1] > feat.shape[1]:

            feat = feat.permute(
                0,
                3,
                1,
                2,
            ).contiguous()

            print(
                f"[FORMAT] Stage {stage_idx}: "
                f"NHWC -> NCHW"
            )

        feat = feat.squeeze(
            0
        )

        print(
            f"[STAGE {stage_idx}] "
            f"C={feat.shape[0]}, "
            f"H={feat.shape[1]}, "
            f"W={feat.shape[2]}"
        )

        output.append(
            feat.detach()
            .cpu()
            .numpy()
        )

    return output


# =============================================================================
# VALID MASK
# =============================================================================

def resize_valid_mask(
    valid_mask,
    height,
    width,
):
    """
    Resize valid 224x224 region mask to feature-map resolution.
    """

    mask = cv2.resize(
        valid_mask.astype(
            np.uint8
        ),
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask.astype(
        bool
    )


# =============================================================================
# CHANNEL RANKING
# =============================================================================

def rank_channels(
    features,
    valid_mask,
    top_k=8,
):
    """
    Rank channels independently inside every Swin stage.

    Score:

        mean(abs(feature)) * spatial_std

    Only valid non-padding pixels are used.

    This is exploratory and is NOT a clinical relevance score.
    """

    results = []

    for stage_idx, feat in enumerate(
        features
    ):

        channels, height, width = (
            feat.shape
        )

        stage_mask = resize_valid_mask(
            valid_mask,
            height,
            width,
        )

        if stage_mask.sum() == 0:

            stage_mask[:] = True

        scores = []

        for channel_idx in range(
            channels
        ):

            channel = feat[
                channel_idx
            ]

            values = channel[
                stage_mask
            ]

            mean_abs = float(
                np.mean(
                    np.abs(values)
                )
            )

            spatial_std = float(
                np.std(
                    values
                )
            )

            score = (
                mean_abs
                * spatial_std
            )

            scores.append(
                {
                    "channel": channel_idx,
                    "score": score,
                    "mean_abs": mean_abs,
                    "spatial_std": spatial_std,
                }
            )

        scores.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        results.append(
            {
                "stage": stage_idx,
                "features": feat,
                "ranking": scores[
                    :top_k
                ],
            }
        )

    return results


# =============================================================================
# TOP CHANNEL FIGURE
# =============================================================================

def create_top_channels_figure(
    ranked_results,
    output_path,
    top_k,
):
    """
    Visualize top channels per stage.
    """

    n_stages = len(
        ranked_results
    )

    fig, axes = plt.subplots(
        n_stages,
        top_k,
        figsize=(
            20,
            4 * n_stages,
        ),
    )

    if n_stages == 1:

        axes = np.expand_dims(
            axes,
            axis=0,
        )

    if top_k == 1:

        axes = np.expand_dims(
            axes,
            axis=1,
        )

    for stage_idx, result in enumerate(
        ranked_results
    ):

        feat = result[
            "features"
        ]

        ranking = result[
            "ranking"
        ]

        for rank_idx, item in enumerate(
            ranking
        ):

            channel_idx = item[
                "channel"
            ]

            channel = feat[
                channel_idx
            ]

            channel_norm = normalize_channel(
                channel
            )

            ax = axes[
                stage_idx,
                rank_idx,
            ]

            ax.imshow(
                channel_norm,
                cmap="viridis",
            )

            ax.set_title(
                f"S{stage_idx} "
                f"C{channel_idx}\n"
                f"score={item['score']:.2e}"
            )

            ax.axis(
                "off"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# OVERLAY FIGURE
# =============================================================================

def create_overlay_figure(
    padded,
    ranked_results,
    output_path,
    top_k,
):
    """
    Create overlays for the top channels.

    Every feature map is resized to 224x224
    inside make_overlay().
    """

    n_stages = len(
        ranked_results
    )

    fig, axes = plt.subplots(
        n_stages,
        top_k,
        figsize=(
            20,
            4 * n_stages,
        ),
    )

    if n_stages == 1:

        axes = np.expand_dims(
            axes,
            axis=0,
        )

    if top_k == 1:

        axes = np.expand_dims(
            axes,
            axis=1,
        )

    for stage_idx, result in enumerate(
        ranked_results
    ):

        feat = result[
            "features"
        ]

        ranking = result[
            "ranking"
        ]

        for rank_idx, item in enumerate(
            ranking
        ):

            channel_idx = item[
                "channel"
            ]

            channel = feat[
                channel_idx
            ]

            overlay = make_overlay(
                padded,
                channel,
                alpha=0.50,
            )

            overlay_rgb = cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB,
            )

            ax = axes[
                stage_idx,
                rank_idx,
            ]

            ax.imshow(
                overlay_rgb
            )

            ax.set_title(
                f"S{stage_idx} "
                f"C{channel_idx}\n"
                f"rank={rank_idx + 1}"
            )

            ax.axis(
                "off"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_rankings(
    ranked_results,
):

    print()
    print(
        "=" * 100
    )

    print(
        "TOP CHANNELS PER STAGE"
    )

    print(
        "=" * 100
    )

    for result in ranked_results:

        stage_idx = result[
            "stage"
        ]

        feat = result[
            "features"
        ]

        ranking = result[
            "ranking"
        ]

        print()

        print(
            f"STAGE {stage_idx} "
            f"shape={feat.shape}"
        )

        print(
            "-" * 100
        )

        print(
            f"{'Rank':<6}"
            f"{'Channel':<10}"
            f"{'Score':<18}"
            f"{'MeanAbs':<18}"
            f"{'SpatialStd':<18}"
        )

        for rank_idx, item in enumerate(
            ranking,
            start=1,
        ):

            print(
                f"{rank_idx:<6}"
                f"{item['channel']:<10}"
                f"{item['score']:<18.6e}"
                f"{item['mean_abs']:<18.6e}"
                f"{item['spatial_std']:<18.6e}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Swin top-channel analysis "
            "for mammography"
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Number of top channels per stage "
            f"(default: {DEFAULT_TOP_K})"
        ),
    )

    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_NAME,
        help=(
            f"Image filename inside {DICOM_DIR}"
        ),
    )

    args = parser.parse_args()

    top_k = args.top_k

    if top_k <= 0:

        raise ValueError(
            "--top-k must be greater than 0"
        )

    ensure_output_dir()

    image_path = (
        DICOM_DIR
        / args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "SWIN TOP CHANNEL ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"Image:       {image_path}"
    )

    print(
        f"Window:      {METHOD}"
    )

    print(
        f"VOI:         {VOI_FUNC}"
    )

    print(
        f"Swin:        {MODEL_NAME}"
    )

    print(
        f"Input:       {SWIN_SIZE}x{SWIN_SIZE}"
    )

    print(
        f"Top-K:       {top_k}"
    )

    print(
        f"Device:      {DEVICE}"
    )

    print(
        "=" * 80
    )

    # -------------------------------------------------------------------------
    # SWIN
    # -------------------------------------------------------------------------

    model = load_swin()

    # -------------------------------------------------------------------------
    # DICOM
    # -------------------------------------------------------------------------

    raw, ds = load_dicom(
        image_path
    )

    # -------------------------------------------------------------------------
    # WINDOWING
    # -------------------------------------------------------------------------

    windowed = apply_windowing(
        raw,
        ds,
    )

    # windowed is already uint8
    windowed = np.asarray(
        windowed,
        dtype=np.uint8,
    )

    # -------------------------------------------------------------------------
    # GRAYSCALE -> BGR
    # -------------------------------------------------------------------------

    resized_input = cv2.cvtColor(
        windowed,
        cv2.COLOR_GRAY2BGR,
    )

    # -------------------------------------------------------------------------
    # RESIZE MAX 1024
    # -------------------------------------------------------------------------

    resized = resize_max(
        resized_input,
        IMG_SIZE,
    )

    print(
        f"[RESIZED] {resized.shape}"
    )

    # -------------------------------------------------------------------------
    # BREAST ROI
    # -------------------------------------------------------------------------

    (
        x1,
        y1,
        x2,
        y2,
    ) = detect_breast_roi(
        resized
    )

    print(
        f"[ROI] x1={x1}, "
        f"y1={y1}, "
        f"x2={x2}, "
        f"y2={y2}"
    )

    crop = resized[
        y1:y2,
        x1:x2,
    ]

    print(
        f"[ROI] crop={crop.shape}"
    )

    # -------------------------------------------------------------------------
    # RESIZE + PAD
    # -------------------------------------------------------------------------

    (
        padded,
        valid_mask,
    ) = resize_and_pad(
        crop,
        SWIN_SIZE,
    )

    # -------------------------------------------------------------------------
    # ROI DEBUG
    # -------------------------------------------------------------------------

    roi_debug_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_roi_debug.png"
    )

    create_roi_debug_figure(
        raw,
        resized,
        (
            x1,
            y1,
            x2,
            y2,
        ),
        padded,
        roi_debug_path,
    )

    print(
        f"[OUTPUT] ROI debug: "
        f"{roi_debug_path}"
    )

    # -------------------------------------------------------------------------
    # SWIN FEATURES
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 75
    )

    print(
        "ANALITZANT CANALS"
    )

    print(
        "=" * 75
    )

    features = extract_swin_features(
        model,
        padded,
    )

    # -------------------------------------------------------------------------
    # RANK
    # -------------------------------------------------------------------------

    ranked_results = rank_channels(
        features,
        valid_mask,
        top_k=top_k,
    )

    # -------------------------------------------------------------------------
    # PRINT
    # -------------------------------------------------------------------------

    print_rankings(
        ranked_results
    )

    # -------------------------------------------------------------------------
    # TOP CHANNELS
    # -------------------------------------------------------------------------

    top_channels_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_top_channels.png"
    )

    create_top_channels_figure(
        ranked_results,
        top_channels_path,
        top_k,
    )

    print()

    print(
        f"[OUTPUT] Top channels: "
        f"{top_channels_path}"
    )

    # -------------------------------------------------------------------------
    # OVERLAYS
    # -------------------------------------------------------------------------

    overlay_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_top_channel_overlays.png"
    )

    create_overlay_figure(
        padded,
        ranked_results,
        overlay_path,
        top_k,
    )

    print(
        f"[OUTPUT] Overlays: "
        f"{overlay_path}"
    )

    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "ANÀLISI COMPLETADA"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Fitxers generats:"
    )

    print()
    print(
        f"  ROI debug:"
    )

    print(
        f"    {roi_debug_path}"
    )

    print()
    print(
        f"  Top channels:"
    )

    print(
        f"    {top_channels_path}"
    )

    print()
    print(
        f"  Top channel overlays:"
    )

    print(
        f"    {overlay_path}"
    )

    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()