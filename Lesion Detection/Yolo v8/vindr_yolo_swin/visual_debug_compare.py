from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    raise RuntimeError(
        "Falta 'timm'. Instala: pip install timm"
    ) from exc

from mammo_prep.windowing import (
    apply_windowing,
    get_dicom_voi_lut_params,
    preprocess_window,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DICOM_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo/images/train"
)

OUTPUT_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/debug_visual_compare"
)

# Resolución utilizada por el pipeline YOLO
IMG_SIZE = 1024

# Preprocessing
METHOD = "breast_tissue"
VOI_FUNC = "LINEAR"
CALC_WINDOW = True

# Swin
SWIN_MODEL_NAME = "swin_tiny_patch4_window7_224"
SWIN_PRETRAINED = True
SWIN_WEIGHT = 0.35

# Swin-Tiny 224
SWIN_INPUT_SIZE = 224

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# VISUALIZACIÓN
# ============================================================

def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen para visualización.

    Especialmente importante para RAW DICOM de 12/14/16 bits.
    """

    img = img.astype(np.float32)

    p1, p99 = np.percentile(
        img,
        [1, 99],
    )

    if p99 <= p1:
        p1 = img.min()
        p99 = img.max()

    img = (
        (img - p1)
        / (p99 - p1 + 1e-6)
        * 255.0
    )

    return np.clip(
        img,
        0,
        255,
    ).astype(np.uint8)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Asegura que la imagen sea uint8 RGB."""

    if img.dtype != np.uint8:
        img = np.clip(
            img,
            0,
            255,
        ).astype(np.uint8)

    if img.ndim == 2:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB,
        )

    elif img.ndim == 3 and img.shape[2] == 1:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB,
        )

    return img


def save_side_by_side(
    images: list[tuple[str, np.ndarray]],
    name: str,
) -> Path:
    """
    Crea una cuadrícula de visualización.

    El resize realizado aquí es SOLO para visualizar.
    No modifica las imágenes utilizadas por el pipeline.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CELL_H = 850
    CELL_W = 850

    MARGIN = 20
    TITLE_HEIGHT = 50

    panels = []

    for title, img in images:

        if img.dtype != np.uint8:
            img = normalize_for_display(img)

        img = ensure_rgb(img)

        h, w = img.shape[:2]

        available_h = CELL_H - TITLE_HEIGHT - 20
        available_w = CELL_W - 20

        scale = min(
            available_w / w,
            available_h / h,
        )

        new_w = max(
            1,
            int(round(w * scale)),
        )

        new_h = max(
            1,
            int(round(h * scale)),
        )

        img_display = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        panel = np.zeros(
            (
                CELL_H,
                CELL_W,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            panel,
            title,
            (15, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        x0 = (
            CELL_W - new_w
        ) // 2

        y0 = (
            TITLE_HEIGHT
            + (
                CELL_H
                - TITLE_HEIGHT
                - new_h
            ) // 2
        )

        panel[
            y0:y0 + new_h,
            x0:x0 + new_w,
        ] = img_display

        panels.append(panel)

    cols = 2

    rows = int(
        np.ceil(
            len(panels) / cols
        )
    )

    canvas_h = (
        rows * CELL_H
        + (rows + 1) * MARGIN
    )

    canvas_w = (
        cols * CELL_W
        + (cols + 1) * MARGIN
    )

    canvas = (
        np.ones(
            (
                canvas_h,
                canvas_w,
                3,
            ),
            dtype=np.uint8,
        )
        * 30
    )

    for idx, panel in enumerate(panels):

        row = idx // cols
        col = idx % cols

        y0 = (
            MARGIN
            + row * (
                CELL_H + MARGIN
            )
        )

        x0 = (
            MARGIN
            + col * (
                CELL_W + MARGIN
            )
        )

        canvas[
            y0:y0 + CELL_H,
            x0:x0 + CELL_W,
        ] = panel

    path = OUTPUT_DIR / f"{name}.png"

    cv2.imwrite(
        str(path),
        cv2.cvtColor(
            canvas,
            cv2.COLOR_RGB2BGR,
        ),
    )

    return path


# ============================================================
# SWIN
# ============================================================

def load_swin_model():
    """Carga el modelo Swin una sola vez."""

    print(
        f"[SWIN] Cargando {SWIN_MODEL_NAME}...",
        flush=True,
    )

    model = timm.create_model(
        SWIN_MODEL_NAME,
        pretrained=SWIN_PRETRAINED,
        features_only=True,
        out_indices=(1, 2, 3),
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"[SWIN] Modelo cargado en {DEVICE}",
        flush=True,
    )

    return model


@torch.no_grad()
def generate_swin_enhancement(
    resized: np.ndarray,
    model,
):
    """
    Genera:

    1. Feature map espacial derivado del Swin.
    2. Imagen enhanced.

    La imagen del pipeline permanece a resolución YOLO.
    Solo la entrada al Swin se redimensiona a 224x224.
    """

    original_h, original_w = resized.shape[:2]

    # ========================================================
    # Entrada Swin: 224x224
    # ========================================================

    swin_input = cv2.resize(
        resized,
        (
            SWIN_INPUT_SIZE,
            SWIN_INPUT_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # ========================================================
    # RGB -> float32 [0,1]
    # ========================================================

    x = (
        swin_input.astype(
            np.float32
        )
        / 255.0
    )

    x = torch.from_numpy(x)

    x = x.permute(
        2,
        0,
        1,
    ).unsqueeze(0)

    x = x.to(DEVICE)

    # ========================================================
    # Normalización ImageNet
    # ========================================================

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

    x = (
        x - mean
    ) / std

    # ========================================================
    # Forward Swin
    # ========================================================

    features = model(x)

    if isinstance(
        features,
        (list, tuple),
    ):
        feat = features[-1]
    else:
        feat = features

    # ========================================================
    # Feature map -> mapa espacial
    # ========================================================

    spatial_map = feat.mean(
        dim=1,
        keepdim=True,
    )

    # ========================================================
    # Volver a resolución original
    # ========================================================

    spatial_map = F.interpolate(
        spatial_map,
        size=(
            original_h,
            original_w,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # ========================================================
    # Tensor -> NumPy
    # ========================================================

    spatial_map = (
        spatial_map
        .squeeze(0)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # Normalizar [0,1]
    # ========================================================

    spatial_map = (
        spatial_map
        - spatial_map.min()
    ) / (
        spatial_map.max()
        - spatial_map.min()
        + 1e-6
    )

    # ========================================================
    # Enhancement
    # ========================================================

    enhanced = (
        resized.astype(
            np.float32
        )
        * (
            1.0
            + SWIN_WEIGHT
            * spatial_map[..., None]
        )
    )

    enhanced = np.clip(
        enhanced,
        0,
        255,
    ).astype(np.uint8)

    return (
        spatial_map,
        enhanced,
    )


# ============================================================
# LECTURA
# ============================================================

def try_read_dicom(
    image_path: Path,
):
    """
    Intenta leer el archivo como DICOM independientemente
    de su extensión.

    Esto es importante porque en este dataset hay archivos
    con extensión .jpg cuyo contenido es realmente DICOM.
    """

    try:

        ds = pydicom.dcmread(
            str(image_path),
            force=False,
        )

        # Comprobamos que realmente tenga pixel data
        _ = ds.pixel_array

        return ds

    except Exception:

        return None


def read_and_process(
    image_path: Path,
    swin_model,
):
    """
    Procesa una imagen.

    Primero intenta leerla como DICOM, independientemente
    de su extensión.

    Si no es DICOM, intenta leerla como imagen convencional.
    """

    print(
        f"[IMAGE] Leyendo: {image_path}",
        flush=True,
    )

    # ========================================================
    # IMPORTANTE:
    # No usamos la extensión para decidir el formato.
    # ========================================================

    ds = try_read_dicom(
        image_path
    )

    # ========================================================
    # CASO 1: DICOM
    # ========================================================

    if ds is not None:

        print(
            "[IMAGE] Formato detectado: DICOM",
            flush=True,
        )

        im = ds.pixel_array

        if im.ndim > 2:
            im = np.squeeze(im)

        raw = im.astype(
            np.float32,
            copy=False,
        )

        # ----------------------------------------------------
        # MONOCHROME1
        # ----------------------------------------------------

        if (
            getattr(
                ds,
                "PhotometricInterpretation",
                "",
            )
            == "MONOCHROME1"
        ):

            raw = (
                raw.max()
                + raw.min()
                - raw
            )

        # ----------------------------------------------------
        # Windowing
        # ----------------------------------------------------

        if CALC_WINDOW:

            processed = preprocess_window(
                raw,
                dicom_dataset=ds,
                method=METHOD,
                voi_func=VOI_FUNC,
            )

        else:

            params = get_dicom_voi_lut_params(
                ds
            )

            processed = apply_windowing(
                raw,
                window_width=params[
                    "window_width"
                ],
                window_center=params[
                    "window_center"
                ],
                voi_func=VOI_FUNC,
                y_min=0,
                y_max=255,
            )

            processed = np.rint(
                np.clip(
                    processed,
                    0,
                    255,
                )
            ).astype(np.uint8)

        # ----------------------------------------------------
        # Grayscale -> RGB
        # ----------------------------------------------------

        if processed.ndim == 2:

            rgb = cv2.cvtColor(
                processed.astype(
                    np.uint8
                ),
                cv2.COLOR_GRAY2RGB,
            )

        elif (
            processed.ndim == 3
            and processed.shape[2] == 1
        ):

            rgb = cv2.cvtColor(
                processed.astype(
                    np.uint8
                ),
                cv2.COLOR_GRAY2RGB,
            )

        else:

            rgb = processed.astype(
                np.uint8
            )

    # ========================================================
    # CASO 2: imagen convencional
    # ========================================================

    else:

        print(
            "[IMAGE] No es DICOM. Intentando imagen convencional...",
            flush=True,
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise RuntimeError(
                "No se pudo leer el archivo "
                "ni como DICOM ni como imagen convencional:\n"
                f"{image_path}"
            )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        raw = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY,
        ).astype(
            np.float32
        )

        processed = raw.astype(
            np.uint8
        )

    # ========================================================
    # Resize para YOLO
    # ========================================================

    h0, w0 = rgb.shape[:2]

    scale = min(
        IMG_SIZE / h0,
        IMG_SIZE / w0,
    )

    new_h = max(
        1,
        int(
            round(
                h0 * scale
            )
        ),
    )

    new_w = max(
        1,
        int(
            round(
                w0 * scale
            )
        ),
    )

    resized = cv2.resize(
        rgb,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    print(
        f"[IMAGE] Original: {h0}x{w0}",
        flush=True,
    )

    print(
        f"[IMAGE] YOLO:     {new_h}x{new_w}",
        flush=True,
    )

    print(
        f"[SWIN] Input:     "
        f"{SWIN_INPUT_SIZE}x{SWIN_INPUT_SIZE}",
        flush=True,
    )

    # ========================================================
    # Swin
    # ========================================================

    print(
        "[SWIN] Extrayendo features...",
        flush=True,
    )

    swin_map, enhanced = (
        generate_swin_enhancement(
            resized,
            swin_model,
        )
    )

    # ========================================================
    # Mapa Swin
    # ========================================================

    swin_map_uint8 = np.rint(
        swin_map * 255.0
    ).astype(np.uint8)

    swin_map_rgb = cv2.cvtColor(
        swin_map_uint8,
        cv2.COLOR_GRAY2RGB,
    )

    # ========================================================
    # Resultado
    # ========================================================

    return [
        (
            "RAW",
            normalize_for_display(
                raw
            ),
        ),
        (
            "WINDOW",
            ensure_rgb(
                processed.astype(
                    np.uint8
                )
            ),
        ),
        (
            "RESIZED",
            ensure_rgb(
                resized
            ),
        ),
        (
            "SWIN FEATURE MAP",
            swin_map_rgb,
        ),
        (
            "SWIN ENHANCED",
            ensure_rgb(
                enhanced
            ),
        ),
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualización del pipeline "
            "DICOM -> windowing -> resize -> "
            "Swin -> enhancement"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a una imagen concreta.",
    )

    args = parser.parse_args()

    # ========================================================
    # Seleccionar imagen
    # ========================================================

    if args.image:

        image_path = Path(
            args.image
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"No existe: {image_path}"
            )

    else:

        files = sorted(
            DICOM_DIR.glob("*")
        )

        if not files:

            raise FileNotFoundError(
                f"No hay imágenes en {DICOM_DIR}"
            )

        image_path = files[0]

    # ========================================================
    # Información
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SWIN VISUAL PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Device:       {DEVICE}"
    )

    print(
        f"Image:        {image_path}"
    )

    print(
        f"Window:       {METHOD}"
    )

    print(
        f"Calc window:  {CALC_WINDOW}"
    )

    print(
        f"VOI function: {VOI_FUNC}"
    )

    print(
        f"Swin:         {SWIN_MODEL_NAME}"
    )

    print(
        f"Swin input:   "
        f"{SWIN_INPUT_SIZE}x{SWIN_INPUT_SIZE}"
    )

    print(
        f"Swin weight:  {SWIN_WEIGHT}"
    )

    print(
        "=" * 70
        + "\n"
    )

    # ========================================================
    # Cargar Swin una sola vez
    # ========================================================

    swin_model = load_swin_model()

    # ========================================================
    # Procesar
    # ========================================================

    stages = read_and_process(
        image_path,
        swin_model,
    )

    # ========================================================
    # Guardar comparación
    # ========================================================

    output_name = (
        f"{image_path.stem}"
        "_raw_window_resize_swin"
    )

    output_path = save_side_by_side(
        stages,
        output_name,
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "[DONE] Comparación guardada:"
    )

    print(
        output_path
    )

    print(
        "=" * 70
        + "\n"
    )


if __name__ == "__main__":
    main()