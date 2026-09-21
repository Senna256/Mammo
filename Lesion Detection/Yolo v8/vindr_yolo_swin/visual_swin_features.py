from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import timm

from mammo_prep.windowing import (
    apply_windowing,
    get_dicom_voi_lut_params,
    preprocess_window,
)


# ==============================================================================
# CONFIGURACIÓ
# ==============================================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

DICOM_DIR = DATASET / "images" / "train"

OUTPUT_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin"
) / "feature_analysis"

IMG_SIZE = 1024

METHOD = "breast_tissue"
CALC_WINDOW = True
VOI_FUNC = "LINEAR"

SWIN_MODEL_NAME = "swin_tiny_patch4_window7_224"
SWIN_PRETRAINED = True

SWIN_INPUT_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# UTILITATS DE VISUALITZACIÓ
# ==============================================================================

def normalize_percentile(img: np.ndarray) -> np.ndarray:
    """
    Normalitza una imatge positiva utilitzant percentils 1-99.
    Només per visualització.
    """
    img = np.asarray(img, dtype=np.float32)

    if not np.isfinite(img).any():
        return np.zeros(img.shape, dtype=np.uint8)

    finite = img[np.isfinite(img)]

    p1 = np.percentile(finite, 1)
    p99 = np.percentile(finite, 99)

    if p99 <= p1:
        return np.zeros(img.shape, dtype=np.uint8)

    out = (img - p1) / (p99 - p1)
    out = np.clip(out, 0.0, 1.0)

    return (out * 255.0).astype(np.uint8)


def normalize_signed(img: np.ndarray) -> np.ndarray:
    """
    Normalització simètrica per mapes amb valors positius i negatius.

    0 -> gris
    valors negatius -> fosc
    valors positius -> clar

    Només per visualització.
    """
    img = np.asarray(img, dtype=np.float32)

    finite = img[np.isfinite(img)]

    if finite.size == 0:
        return np.full(img.shape, 127, dtype=np.uint8)

    max_abs = np.percentile(np.abs(finite), 99)

    if max_abs < 1e-8:
        return np.full(img.shape, 127, dtype=np.uint8)

    out = img / max_abs
    out = np.clip(out, -1.0, 1.0)

    out = (out + 1.0) / 2.0

    return (out * 255.0).astype(np.uint8)


def to_display_bgr(img: np.ndarray) -> np.ndarray:
    """
    Converteix qualsevol mapa 2D a BGR uint8 per OpenCV.
    """
    if img.ndim == 2:
        img = normalize_percentile(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        img = normalize_percentile(img[:, :, 0])
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    raise ValueError(f"Shape no suportat: {img.shape}")


def resize_for_display(
    img: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Redimensiona una imatge per ocupar exactament el panel.
    És només visualització.
    """
    return cv2.resize(
        img,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def add_title(
    img: np.ndarray,
    title: str,
    height: int = 45,
) -> np.ndarray:
    """
    Afegeix una capçalera negra amb el títol.
    """
    h, w = img.shape[:2]

    canvas = np.zeros(
        (h + height, w, 3),
        dtype=np.uint8,
    )

    canvas[height:, :] = img

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
    img: np.ndarray,
    title: str,
    width: int = 500,
    height: int = 500,
) -> np.ndarray:
    """
    Crea un panel de mida fixa.
    """
    img = to_display_bgr(img)
    img = resize_for_display(img, width, height)

    return add_title(
        img,
        title,
    )


def make_overlay(
    base: np.ndarray,
    feature_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Superposa un feature map sobre la mamografia.

    El feature map es mostra amb una colormap només per facilitar
    la visualització.
    """
    base = to_display_bgr(base)

    feature = normalize_percentile(feature_map)

    feature_color = cv2.applyColorMap(
        feature,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        base,
        1.0 - alpha,
        feature_color,
        alpha,
        0,
    )

    return overlay


# ==============================================================================
# LECTURA I PREPROCESSING DICOM
# ==============================================================================

def read_and_process_dicom(
    image_path: Path,
    imgsz: int,
    method: str = "breast_tissue",
    calc_window: bool = True,
    voi_func: str = "LINEAR",
):
    """
    Llegeix el DICOM real.

    IMPORTANT:
    Els fitxers .jpg de vindr_yolo són links als DICOM originals.
    Per tant NO utilitzem cv2.imread() ni Pillow.
    pydicom.dcmread() segueix el link i llegeix el DICOM.
    """

    print(f"[IMAGE] Llegint: {image_path}", flush=True)

    # --------------------------------------------------------------------------
    # DICOM
    # --------------------------------------------------------------------------

    ds = pydicom.dcmread(str(image_path))

    im = ds.pixel_array

    if im.ndim > 2:
        im = np.squeeze(im)

    im = im.astype(np.float32, copy=False)

    # --------------------------------------------------------------------------
    # MONOCHROME1
    # --------------------------------------------------------------------------

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        im = im.max() + im.min() - im

    raw = im.copy()

    print(
        f"[DICOM] RAW shape={raw.shape}, "
        f"dtype={raw.dtype}, "
        f"min={raw.min():.2f}, "
        f"max={raw.max():.2f}",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # WINDOWING
    # --------------------------------------------------------------------------

    if calc_window:

        print(
            f"[WINDOW] method={method}, VOI={voi_func}",
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

    # --------------------------------------------------------------------------
    # GRAYSCALE -> BGR
    # --------------------------------------------------------------------------

    if im.ndim == 2:

        im = cv2.cvtColor(
            im,
            cv2.COLOR_GRAY2BGR,
        )

    elif im.ndim == 3 and im.shape[2] == 1:

        im = cv2.cvtColor(
            im,
            cv2.COLOR_GRAY2BGR,
        )

    elif im.ndim != 3 or im.shape[2] != 3:

        raise RuntimeError(
            f"Shape inesperat després del preprocessing: {im.shape}"
        )

    windowed = im.copy()

    # --------------------------------------------------------------------------
    # RESIZE mantenint aspect ratio
    # --------------------------------------------------------------------------

    h0, w0 = im.shape[:2]

    scale = min(
        imgsz / h0,
        imgsz / w0,
    )

    new_h = max(
        1,
        int(round(h0 * scale)),
    )

    new_w = max(
        1,
        int(round(w0 * scale)),
    )

    if (new_h, new_w) != (h0, w0):

        im = cv2.resize(
            im,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

    resized = np.ascontiguousarray(
        im,
        dtype=np.uint8,
    )

    print(
        f"[RESIZED] shape={resized.shape}",
        flush=True,
    )

    return raw, windowed, resized


# ==============================================================================
# SWIN
# ==============================================================================

def load_swin_model():

    print()
    print("=" * 70)
    print("SWIN FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Model:       {SWIN_MODEL_NAME}")
    print(f"Pretrained:  {SWIN_PRETRAINED}")
    print(f"Input:       {SWIN_INPUT_SIZE}x{SWIN_INPUT_SIZE}")
    print(f"Device:      {DEVICE}")
    print("=" * 70)
    print()

    print(
        f"[SWIN] Carregant {SWIN_MODEL_NAME}...",
        flush=True,
    )

    model = timm.create_model(
        SWIN_MODEL_NAME,
        pretrained=SWIN_PRETRAINED,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"[SWIN] Model carregat a {DEVICE}",
        flush=True,
    )

    return model


# ==============================================================================
# EXTRACCIÓ DE FEATURES
# ==============================================================================

@torch.no_grad()
def extract_swin_features(
    model,
    image: np.ndarray,
):
    """
    Extreu les features de les 4 etapes del Swin.

    La imatge original pot ser 1024xN.
    Només l'entrada del Swin es redimensiona a 224x224.
    """

    # --------------------------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------------------------------
    # 224x224 només per Swin
    # --------------------------------------------------------------------------

    swin_img = cv2.resize(
        rgb,
        (
            SWIN_INPUT_SIZE,
            SWIN_INPUT_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------------------------
    # [H,W,C] -> [1,C,H,W]
    # --------------------------------------------------------------------------

    tensor = torch.from_numpy(
        swin_img.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # --------------------------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------------------------

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    tensor = (
        tensor - mean
    ) / std

    # --------------------------------------------------------------------------
    # SWIN
    # --------------------------------------------------------------------------

    features = model(tensor)

    if not isinstance(features, (list, tuple)):
        features = [features]

    processed = []

    print()
    print("=" * 70)
    print("SWIN FEATURES")
    print("=" * 70)

    for idx, feat in enumerate(features):

        # ----------------------------------------------------------------------
        # Timm pot retornar NCHW o NHWC depenent de la configuració/versió.
        # Detectem automàticament el format.
        # ----------------------------------------------------------------------

        if feat.ndim != 4:
            raise RuntimeError(
                f"Feature inesperada a stage {idx}: {feat.shape}"
            )

        # NCHW típic:
        # [1, C, H, W]
        #
        # NHWC:
        # [1, H, W, C]

        if feat.shape[1] > feat.shape[-1]:

            # probablement NHWC
            feat = feat.permute(
                0,
                3,
                1,
                2,
            ).contiguous()

        # Ara sempre:
        # [1, C, H, W]

        _, channels, h, w = feat.shape

        print(
            f"Stage {idx}: "
            f"shape={tuple(feat.shape)} "
            f"C={channels} "
            f"H={h} "
            f"W={w}",
            flush=True,
        )

        # ----------------------------------------------------------------------
        # 1. MEAN
        # ----------------------------------------------------------------------

        mean_map = feat.mean(
            dim=1,
            keepdim=True,
        )

        # ----------------------------------------------------------------------
        # 2. ABS MEAN
        # ----------------------------------------------------------------------

        abs_mean_map = feat.abs().mean(
            dim=1,
            keepdim=True,
        )

        # ----------------------------------------------------------------------
        # 3. L2 NORM
        # ----------------------------------------------------------------------

        l2_map = torch.sqrt(
            torch.sum(
                feat ** 2,
                dim=1,
                keepdim=True,
            ) + 1e-8
        )

        # ----------------------------------------------------------------------
        # Convertim els 3 mapes a numpy
        # ----------------------------------------------------------------------

        mean_np = mean_map.squeeze().cpu().numpy()

        abs_mean_np = (
            abs_mean_map
            .squeeze()
            .cpu()
            .numpy()
        )

        l2_np = (
            l2_map
            .squeeze()
            .cpu()
            .numpy()
        )

        processed.append(
            {
                "stage": idx,
                "shape": tuple(feat.shape),
                "mean": mean_np,
                "abs_mean": abs_mean_np,
                "l2": l2_np,
            }
        )

    print("=" * 70)
    print()

    return processed


# ==============================================================================
# RESIZE DELS FEATURE MAPS
# ==============================================================================

def resize_feature_map(
    feature_map: np.ndarray,
    target_shape,
):
    """
    Porta el feature map a la mida de la imatge RESIZED.
    """

    target_h, target_w = target_shape

    return cv2.resize(
        feature_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )


# ==============================================================================
# CREACIÓ DE LA FIGURA
# ==============================================================================

def create_feature_figure(
    resized: np.ndarray,
    features,
    output_path: Path,
):
    """
    Crea una figura:

                MEAN | ABS MEAN | L2 | MEAN OVERLAY

    Stage 0
    Stage 1
    Stage 2
    Stage 3
    """

    PANEL_W = 500
    PANEL_H = 500

    HEADER_H = 45

    GAP = 10

    rows = len(features)
    cols = 4

    cell_w = PANEL_W
    cell_h = PANEL_H + HEADER_H

    figure_w = (
        cols * cell_w
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

    for row, item in enumerate(features):

        stage = item["stage"]

        # ----------------------------------------------------------------------
        # Resize dels mapes a la mida RESIZED
        # ----------------------------------------------------------------------

        mean_map = resize_feature_map(
            item["mean"],
            resized.shape[:2],
        )

        abs_mean_map = resize_feature_map(
            item["abs_mean"],
            resized.shape[:2],
        )

        l2_map = resize_feature_map(
            item["l2"],
            resized.shape[:2],
        )

        # ----------------------------------------------------------------------
        # Panels
        # ----------------------------------------------------------------------

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
            resized,
            mean_map,
            alpha=0.45,
        )

        # ----------------------------------------------------------------------
        # Crear panels
        # ----------------------------------------------------------------------

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

        # ----------------------------------------------------------------------
        # Posicionar
        # ----------------------------------------------------------------------

        y = GAP + row * (
            cell_h + GAP
        )

        for col, panel in enumerate(panels):

            x = GAP + col * (
                cell_w + GAP
            )

            canvas[
                y:y + panel.shape[0],
                x:x + panel.shape[1],
            ] = panel

    # --------------------------------------------------------------------------
    # Guardar
    # --------------------------------------------------------------------------

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


# ==============================================================================
# ESTADÍSTIQUES
# ==============================================================================

def print_feature_statistics(features):

    print()
    print("=" * 80)
    print("ESTADÍSTIQUES DELS FEATURE MAPS")
    print("=" * 80)

    for item in features:

        stage = item["stage"]

        print()
        print(f"STAGE {stage}")
        print("-" * 80)

        for name in [
            "mean",
            "abs_mean",
            "l2",
        ]:

            x = item[name]

            print(
                f"{name:10s} "
                f"min={x.min(): .6f} "
                f"max={x.max(): .6f} "
                f"mean={x.mean(): .6f} "
                f"std={x.std(): .6f}"
            )

    print("=" * 80)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analitza les features internes de Swin "
            "sobre una mamografia ViNDR."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Path al .jpg symlink del dataset. "
            "Si no es dona, s'agafa la primera imatge."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Seleccionar imatge
    # --------------------------------------------------------------------------

    if args.image is not None:

        image_path = Path(
            args.image
        )

    else:

        candidates = sorted(
            DICOM_DIR.glob("*")
        )

        if not candidates:

            raise FileNotFoundError(
                f"No s'han trobat imatges a {DICOM_DIR}"
            )

        image_path = candidates[0]

    if not image_path.exists():

        raise FileNotFoundError(
            f"No existeix: {image_path}"
        )

    # --------------------------------------------------------------------------
    # Info
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("SWIN FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Device:       {DEVICE}")
    print(f"Image:        {image_path}")
    print(f"Window:       {METHOD}")
    print(f"Calc window:  {CALC_WINDOW}")
    print(f"VOI function: {VOI_FUNC}")
    print(f"Swin:         {SWIN_MODEL_NAME}")
    print(f"Swin input:   {SWIN_INPUT_SIZE}x{SWIN_INPUT_SIZE}")
    print("=" * 70)
    print()

    # --------------------------------------------------------------------------
    # Carregar Swin
    # --------------------------------------------------------------------------

    model = load_swin_model()

    # --------------------------------------------------------------------------
    # DICOM + preprocessing
    # --------------------------------------------------------------------------

    raw, windowed, resized = read_and_process_dicom(
        image_path=image_path,
        imgsz=IMG_SIZE,
        method=METHOD,
        calc_window=CALC_WINDOW,
        voi_func=VOI_FUNC,
    )

    # --------------------------------------------------------------------------
    # Features Swin
    # --------------------------------------------------------------------------

    features = extract_swin_features(
        model,
        resized,
    )

    # --------------------------------------------------------------------------
    # Estadístiques
    # --------------------------------------------------------------------------

    print_feature_statistics(
        features
    )

    # --------------------------------------------------------------------------
    # Output
    # --------------------------------------------------------------------------

    output_name = (
        f"{image_path.stem}_"
        f"swin_feature_analysis.png"
    )

    output_path = (
        OUTPUT_DIR / output_name
    )

    create_feature_figure(
        resized=resized,
        features=features,
        output_path=output_path,
    )

    # --------------------------------------------------------------------------
    # Final
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANÀLISI COMPLETADA")
    print("=" * 70)
    print(f"Output: {output_path}")
    print()
    print(
        "No s'ha entrenat res."
    )
    print(
        "No s'han modificat els DICOM."
    )
    print(
        "No s'han creat còpies de les imatges del dataset."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()