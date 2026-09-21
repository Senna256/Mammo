from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import timm

from mammo_prep.windowing import (
    apply_windowing,
    get_dicom_voi_lut_params,
    preprocess_window,
)


# =============================================================================
# CONFIG
# =============================================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

DICOM_DIR = DATASET / "images" / "train"

OUTPUT_DIR = (
    Path(
        "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/"
        "vindr_yolo_swin"
    )
    / "feature_analysis_fixed"
)

IMG_SIZE = 1024

METHOD = "breast_tissue"
CALC_WINDOW = True
VOI_FUNC = "LINEAR"

SWIN_MODEL_NAME = "swin_tiny_patch4_window7_224"
SWIN_PRETRAINED = True
SWIN_INPUT_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# VISUALIZATION
# =============================================================================

def normalize_percentile(img):
    img = np.asarray(img, dtype=np.float32)

    finite = img[np.isfinite(img)]

    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    p1 = np.percentile(finite, 1)
    p99 = np.percentile(finite, 99)

    if p99 <= p1:
        return np.zeros(img.shape, dtype=np.uint8)

    out = (img - p1) / (p99 - p1)
    out = np.clip(out, 0.0, 1.0)

    return (out * 255.0).astype(np.uint8)


def normalize_signed(img):
    img = np.asarray(img, dtype=np.float32)

    finite = img[np.isfinite(img)]

    if finite.size == 0:
        return np.full(img.shape, 127, dtype=np.uint8)

    max_abs = np.percentile(
        np.abs(finite),
        99,
    )

    if max_abs < 1e-8:
        return np.full(img.shape, 127, dtype=np.uint8)

    out = img / max_abs
    out = np.clip(out, -1.0, 1.0)

    out = (out + 1.0) / 2.0

    return (out * 255.0).astype(np.uint8)


def to_bgr(img):
    if img.ndim == 2:
        img = normalize_percentile(img)
        return cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR,
        )

    if img.ndim == 3 and img.shape[2] == 1:
        img = normalize_percentile(
            img[:, :, 0]
        )
        return cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR,
        )

    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(
                img,
                0,
                255,
            ).astype(np.uint8)

        return img

    raise ValueError(
        f"Shape no suportat: {img.shape}"
    )


def add_title(
    img,
    title,
    height=45,
):
    h, w = img.shape[:2]

    canvas = np.zeros(
        (h + height, w, 3),
        dtype=np.uint8,
    )

    canvas[height:] = img

    cv2.putText(
        canvas,
        title,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def make_panel(
    img,
    title,
    width=500,
    height=500,
):
    img = to_bgr(img)

    img = cv2.resize(
        img,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    return add_title(
        img,
        title,
    )


def make_overlay(
    base,
    feature_map,
    alpha=0.45,
):
    base = to_bgr(base)

    feature = normalize_percentile(
        feature_map
    )

    feature_color = cv2.applyColorMap(
        feature,
        cv2.COLORMAP_JET,
    )

    return cv2.addWeighted(
        base,
        1.0 - alpha,
        feature_color,
        alpha,
        0,
    )


# =============================================================================
# DICOM
# =============================================================================

def read_and_process_dicom(
    image_path,
    imgsz=1024,
    method="breast_tissue",
    calc_window=True,
    voi_func="LINEAR",
):
    """
    Els .jpg de vindr_yolo són links als DICOM originals.

    Per tant:
        .jpg -> pydicom.dcmread()
    """

    print(
        f"[IMAGE] Llegint: {image_path}",
        flush=True,
    )

    ds = pydicom.dcmread(
        str(image_path)
    )

    im = ds.pixel_array

    if im.ndim > 2:
        im = np.squeeze(im)

    im = im.astype(
        np.float32,
        copy=False,
    )

    # -------------------------------------------------------------------------
    # MONOCHROME1
    # -------------------------------------------------------------------------

    if getattr(
        ds,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":

        im = (
            im.max()
            + im.min()
            - im
        )

    raw = im.copy()

    print(
        f"[RAW] shape={raw.shape} "
        f"dtype={raw.dtype} "
        f"min={raw.min():.2f} "
        f"max={raw.max():.2f}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # WINDOWING
    # -------------------------------------------------------------------------

    if calc_window:

        print(
            f"[WINDOW] method={method} "
            f"VOI={voi_func}",
            flush=True,
        )

        im = preprocess_window(
            im,
            dicom_dataset=ds,
            method=method,
            voi_func=voi_func,
        )

    else:

        params = get_dicom_voi_lut_params(ds)

        im = apply_windowing(
            im,
            window_width=params["window_width"],
            window_center=params["window_center"],
            voi_func=voi_func,
            y_min=0,
            y_max=255,
        )

        im = np.rint(
            np.clip(im, 0, 255)
        ).astype(np.uint8)

    # -------------------------------------------------------------------------
    # GRAYSCALE -> BGR
    # -------------------------------------------------------------------------

    if im.ndim == 2:

        im = cv2.cvtColor(
            im,
            cv2.COLOR_GRAY2BGR,
        )

    elif (
        im.ndim == 3
        and im.shape[2] == 1
    ):

        im = cv2.cvtColor(
            im,
            cv2.COLOR_GRAY2BGR,
        )

    elif not (
        im.ndim == 3
        and im.shape[2] == 3
    ):

        raise RuntimeError(
            f"Shape inesperat després del windowing: "
            f"{im.shape}"
        )

    windowed = im.copy()

    # -------------------------------------------------------------------------
    # RESIZE mantenint aspect ratio
    # -------------------------------------------------------------------------

    h, w = im.shape[:2]

    scale = min(
        imgsz / h,
        imgsz / w,
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
        im,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    resized = np.ascontiguousarray(
        resized,
        dtype=np.uint8,
    )

    print(
        f"[RESIZED] shape={resized.shape}",
        flush=True,
    )

    return raw, windowed, resized


# =============================================================================
# BREAST ROI
# =============================================================================

def detect_breast_roi(image):
    """
    Detecta aproximadament la regió mamària.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = normalize_percentile(
        gray
    )

    threshold = max(
        5,
        int(
            np.percentile(
                gray,
                30,
            )
        ),
    )

    binary = (
        gray > threshold
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "No s'ha pogut detectar la mama."
        )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    contour = contours[0]

    x, y, w, h = cv2.boundingRect(
        contour
    )

    image_h, image_w = gray.shape

    pad_x = int(
        0.03 * w
    )

    pad_y = int(
        0.03 * h
    )

    x1 = max(
        0,
        x - pad_x,
    )

    y1 = max(
        0,
        y - pad_y,
    )

    x2 = min(
        image_w,
        x + w + pad_x,
    )

    y2 = min(
        image_h,
        y + h + pad_y,
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    print(
        f"[ROI] bbox=({x1}, {y1}, {x2}, {y2})",
        flush=True,
    )

    print(
        f"[ROI] crop shape={crop.shape}",
        flush=True,
    )

    return (
        crop,
        (x1, y1, x2, y2),
    )


# =============================================================================
# ASPECT RATIO + PADDING
# =============================================================================

def resize_with_aspect_ratio_and_pad(
    image,
    target_size=224,
    pad_value=0,
):
    """
    Redimensiona sense deformar i afegeix padding
    fins a target_size x target_size.
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
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (
            target_size,
            target_size,
            image.shape[2],
        ),
        pad_value,
        dtype=image.dtype,
    )

    top = (
        target_size - new_h
    ) // 2

    left = (
        target_size - new_w
    ) // 2

    canvas[
        top:top + new_h,
        left:left + new_w,
    ] = resized

    bottom = (
        target_size
        - new_h
        - top
    )

    right = (
        target_size
        - new_w
        - left
    )

    print(
        f"[PAD] original={w}x{h}",
        flush=True,
    )

    print(
        f"[PAD] resized={new_w}x{new_h}",
        flush=True,
    )

    print(
        f"[PAD] padding="
        f"left:{left}, "
        f"top:{top}, "
        f"right:{right}, "
        f"bottom:{bottom}",
        flush=True,
    )

    print(
        f"[PAD] final={canvas.shape}",
        flush=True,
    )

    return canvas


# =============================================================================
# SWIN MODEL
# =============================================================================

def load_swin_model():

    print()
    print("=" * 70)
    print("CARREGANT SWIN")
    print("=" * 70)
    print(
        f"Model:  {SWIN_MODEL_NAME}"
    )
    print(
        f"Input:  {SWIN_INPUT_SIZE}x{SWIN_INPUT_SIZE}"
    )
    print(
        f"Device: {DEVICE}"
    )
    print("=" * 70)

    model = timm.create_model(
        SWIN_MODEL_NAME,
        pretrained=SWIN_PRETRAINED,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )

    model.to(DEVICE)
    model.eval()

    print(
        "[SWIN] Model carregat.",
        flush=True,
    )

    return model


# =============================================================================
# FEATURE FORMAT
# =============================================================================

def convert_to_nchw(
    feat,
    stage,
):
    """
    Converteix una feature de timm a:

        [B, C, H, W]

    Acceptem tant:

        NCHW = [B, C, H, W]

    com:

        NHWC = [B, H, W, C]

    Per Swin, els canals són molt més grans que H/W.
    """

    if feat.ndim != 4:
        raise RuntimeError(
            f"Stage {stage}: "
            f"feature inesperada {feat.shape}"
        )

    b, d1, d2, d3 = feat.shape

    print(
        f"[FORMAT] Stage {stage} original: "
        f"{tuple(feat.shape)}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # NCHW
    #
    # Exemple:
    # [1, 96, 56, 56]
    #
    # d1 és el canal i és molt més gran que H/W.
    # -------------------------------------------------------------------------

    if d1 > d2 and d1 > d3:

        print(
            f"[FORMAT] Stage {stage}: NCHW",
            flush=True,
        )

        return feat.contiguous()

    # -------------------------------------------------------------------------
    # NHWC
    #
    # Exemple:
    # [1, 56, 56, 96]
    #
    # d3 és el canal i és molt més gran que H/W.
    # -------------------------------------------------------------------------

    if d3 > d1 and d3 > d2:

        print(
            f"[FORMAT] Stage {stage}: NHWC -> NCHW",
            flush=True,
        )

        feat = feat.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        return feat

    raise RuntimeError(
        f"No puc determinar el format de Stage {stage}: "
        f"{feat.shape}"
    )


# =============================================================================
# EXTRACT FEATURES
# =============================================================================

@torch.no_grad()
def extract_swin_features(
    model,
    image_224,
):
    """
    Extreu les features de les quatre stages.

    IMPORTANT:
    Primer convertim explícitament a NCHW.

    Per tant:

        feat = [B, C, H, W]

    i:

        mean(dim=1)

    és realment una mitjana dels CANALS.
    """

    rgb = cv2.cvtColor(
        image_224,
        cv2.COLOR_BGR2RGB,
    )

    tensor = torch.from_numpy(
        rgb.astype(
            np.float32
        ) / 255.0
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    # -------------------------------------------------------------------------
    # ImageNet normalization
    # -------------------------------------------------------------------------

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=DEVICE,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=DEVICE,
    ).view(
        1,
        3,
        1,
        1,
    )

    tensor = (
        tensor - mean
    ) / std

    # -------------------------------------------------------------------------
    # SWIN
    # -------------------------------------------------------------------------

    features = model(
        tensor
    )

    if not isinstance(
        features,
        (list, tuple),
    ):
        features = [features]

    processed = []

    print()
    print("=" * 80)
    print("SWIN FEATURES - FORMAT CORREGIT")
    print("=" * 80)

    for stage, feat in enumerate(
        features
    ):

        # ---------------------------------------------------------------------
        # CORREGIM FORMAT
        # ---------------------------------------------------------------------

        feat = convert_to_nchw(
            feat,
            stage,
        )

        b, c, h, w = feat.shape

        print(
            f"[FEATURE] Stage {stage}: "
            f"B={b}, C={c}, H={h}, W={w}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # MEAN DELS CANALS
        # ---------------------------------------------------------------------

        mean_map = feat.mean(
            dim=1,
            keepdim=True,
        )

        # ---------------------------------------------------------------------
        # ABS MEAN DELS CANALS
        # ---------------------------------------------------------------------

        abs_mean_map = feat.abs().mean(
            dim=1,
            keepdim=True,
        )

        # ---------------------------------------------------------------------
        # L2 NORM DELS CANALS
        # ---------------------------------------------------------------------

        l2_map = torch.sqrt(
            torch.sum(
                feat ** 2,
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        processed.append(
            {
                "stage": stage,
                "shape": tuple(
                    feat.shape
                ),
                "mean": (
                    mean_map
                    .squeeze()
                    .cpu()
                    .numpy()
                ),
                "abs_mean": (
                    abs_mean_map
                    .squeeze()
                    .cpu()
                    .numpy()
                ),
                "l2": (
                    l2_map
                    .squeeze()
                    .cpu()
                    .numpy()
                ),
            }
        )

    print("=" * 80)

    return processed


# =============================================================================
# RESIZE FEATURE MAP
# =============================================================================

def resize_feature_map(
    feature_map,
    target_shape,
):
    h, w = target_shape

    return cv2.resize(
        feature_map.astype(
            np.float32
        ),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )


# =============================================================================
# FEATURE FIGURE
# =============================================================================

def create_feature_figure(
    padded,
    features,
    output_path,
):
    PANEL_W = 500
    PANEL_H = 500

    GAP = 10
    HEADER_H = 45

    rows = len(features)
    cols = 4

    cell_h = (
        PANEL_H
        + HEADER_H
    )

    figure_w = (
        cols * PANEL_W
        + (cols + 1) * GAP
    )

    figure_h = (
        rows * cell_h
        + (rows + 1) * GAP
    )

    canvas = np.zeros(
        (
            figure_h,
            figure_w,
            3,
        ),
        dtype=np.uint8,
    )

    for row, item in enumerate(
        features
    ):

        stage = item["stage"]

        mean_map = resize_feature_map(
            item["mean"],
            padded.shape[:2],
        )

        abs_mean_map = resize_feature_map(
            item["abs_mean"],
            padded.shape[:2],
        )

        l2_map = resize_feature_map(
            item["l2"],
            padded.shape[:2],
        )

        mean_display = normalize_signed(
            mean_map
        )

        abs_display = normalize_percentile(
            abs_mean_map
        )

        l2_display = normalize_percentile(
            l2_map
        )

        mean_overlay = make_overlay(
            padded,
            mean_map,
            alpha=0.45,
        )

        panels = [
            make_panel(
                mean_display,
                f"STAGE {stage} - MEAN",
                PANEL_W,
                PANEL_H,
            ),
            make_panel(
                abs_display,
                f"STAGE {stage} - ABS MEAN",
                PANEL_W,
                PANEL_H,
            ),
            make_panel(
                l2_display,
                f"STAGE {stage} - L2 NORM",
                PANEL_W,
                PANEL_H,
            ),
            make_panel(
                mean_overlay,
                f"STAGE {stage} - MEAN OVERLAY",
                PANEL_W,
                PANEL_H,
            ),
        ]

        y = (
            GAP
            + row * (
                cell_h + GAP
            )
        )

        for col, panel in enumerate(
            panels
        ):

            x = (
                GAP
                + col * (
                    PANEL_W + GAP
                )
            )

            canvas[
                y:y + panel.shape[0],
                x:x + panel.shape[1],
            ] = panel

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )

    print(
        f"[OUTPUT] Guardat: {output_path}",
        flush=True,
    )


# =============================================================================
# PADDING DEBUG
# =============================================================================

def save_padding_debug(
    crop,
    padded,
    output_path,
):
    panels = [
        make_panel(
            crop,
            "CROPPED BREAST",
        ),
        make_panel(
            padded,
            "224x224 ASPECT RATIO + PADDING",
        ),
    ]

    gap = 10

    width = (
        len(panels) * 500
        + (len(panels) + 1) * gap
    )

    height = (
        panels[0].shape[0]
        + 2 * gap
    )

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    x = gap

    for panel in panels:

        canvas[
            gap:gap + panel.shape[0],
            x:x + panel.shape[1],
        ] = panel

        x += (
            panel.shape[1]
            + gap
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )

    print(
        f"[PADDING DEBUG] Guardat: {output_path}",
        flush=True,
    )


# =============================================================================
# ROI DEBUG
# =============================================================================

def save_roi_debug(
    resized,
    crop,
    bbox,
    output_path,
):
    x1, y1, x2, y2 = bbox

    detected = resized.copy()

    cv2.rectangle(
        detected,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        4,
    )

    panels = [
        make_panel(
            resized,
            "RESIZED",
        ),
        make_panel(
            detected,
            "DETECTED BREAST ROI",
        ),
        make_panel(
            crop,
            "CROPPED BREAST",
        ),
    ]

    gap = 10

    width = (
        len(panels) * 500
        + (len(panels) + 1) * gap
    )

    height = (
        panels[0].shape[0]
        + 2 * gap
    )

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    x = gap

    for panel in panels:

        canvas[
            gap:gap + panel.shape[0],
            x:x + panel.shape[1],
        ] = panel

        x += (
            panel.shape[1]
            + gap
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )

    print(
        f"[ROI DEBUG] Guardat: {output_path}",
        flush=True,
    )


# =============================================================================
# STATISTICS
# =============================================================================

def print_statistics(
    features,
):

    print()
    print("=" * 90)
    print("ESTADÍSTIQUES DELS MAPES")
    print("=" * 90)

    for item in features:

        print()
        print(
            f"STAGE {item['stage']} "
            f"{item['shape']}"
        )

        print("-" * 90)

        for name in (
            "mean",
            "abs_mean",
            "l2",
        ):

            x = item[name]

            print(
                f"{name:10s} "
                f"min={x.min(): .6f} "
                f"max={x.max(): .6f} "
                f"mean={x.mean(): .6f} "
                f"std={x.std(): .6f}"
            )

    print("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analitza correctament les features "
            "de Swin amb crop + aspect ratio + padding."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path al .jpg symlink del dataset.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # IMAGE
    # -------------------------------------------------------------------------

    if args.image:

        image_path = Path(
            args.image
        )

    else:

        candidates = sorted(
            DICOM_DIR.glob("*")
        )

        if not candidates:

            raise FileNotFoundError(
                f"No hi ha imatges a {DICOM_DIR}"
            )

        image_path = candidates[0]

    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "SWIN FEATURE ANALYSIS - "
        "CROP + PADDING + FORMAT FIX"
    )
    print("=" * 80)
    print(
        f"Image:        {image_path}"
    )
    print(
        f"Window:       {METHOD}"
    )
    print(
        f"VOI:          {VOI_FUNC}"
    )
    print(
        f"Swin:         {SWIN_MODEL_NAME}"
    )
    print(
        f"Swin input:   {SWIN_INPUT_SIZE}x{SWIN_INPUT_SIZE}"
    )
    print(
        f"Device:       {DEVICE}"
    )
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # SWIN
    # -------------------------------------------------------------------------

    model = load_swin_model()

    # -------------------------------------------------------------------------
    # DICOM + WINDOW + RESIZE
    # -------------------------------------------------------------------------

    raw, windowed, resized = (
        read_and_process_dicom(
            image_path=image_path,
            imgsz=IMG_SIZE,
            method=METHOD,
            calc_window=CALC_WINDOW,
            voi_func=VOI_FUNC,
        )
    )

    # -------------------------------------------------------------------------
    # BREAST ROI
    # -------------------------------------------------------------------------

    crop, bbox = detect_breast_roi(
        resized
    )

    # -------------------------------------------------------------------------
    # ROI DEBUG
    # -------------------------------------------------------------------------

    roi_output = (
        OUTPUT_DIR
        / f"{image_path.stem}_roi_debug.png"
    )

    save_roi_debug(
        resized=resized,
        crop=crop,
        bbox=bbox,
        output_path=roi_output,
    )

    # -------------------------------------------------------------------------
    # ASPECT RATIO + PADDING
    # -------------------------------------------------------------------------

    padded = (
        resize_with_aspect_ratio_and_pad(
            crop,
            target_size=SWIN_INPUT_SIZE,
            pad_value=0,
        )
    )

    # -------------------------------------------------------------------------
    # PADDING DEBUG
    # -------------------------------------------------------------------------

    padding_output = (
        OUTPUT_DIR
        / f"{image_path.stem}_padding_debug.png"
    )

    save_padding_debug(
        crop=crop,
        padded=padded,
        output_path=padding_output,
    )

    # -------------------------------------------------------------------------
    # SWIN
    # -------------------------------------------------------------------------

    features = extract_swin_features(
        model=model,
        image_224=padded,
    )

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    print_statistics(
        features
    )

    # -------------------------------------------------------------------------
    # FEATURE FIGURE
    # -------------------------------------------------------------------------

    feature_output = (
        OUTPUT_DIR
        / f"{image_path.stem}_swin_features_FIXED.png"
    )

    create_feature_figure(
        padded=padded,
        features=features,
        output_path=feature_output,
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("COMPLETAT")
    print("=" * 80)
    print(
        f"ROI:       {roi_output}"
    )
    print(
        f"Padding:   {padding_output}"
    )
    print(
        f"Features:  {feature_output}"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()