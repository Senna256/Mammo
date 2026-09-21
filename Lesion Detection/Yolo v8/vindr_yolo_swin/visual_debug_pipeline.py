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
    raise RuntimeError("Falta 'timm'. Instala: pip install timm") from exc

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
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/debug_visual_only"
)

IMG_SIZE = 1024
METHOD = "breast_tissue"
VOI_FUNC = "LINEAR"
CALC_WINDOW = True
SWIN_MODEL_NAME = "swin_tiny_patch4_window7_224"
SWIN_PRETRAINED = True
SWIN_WEIGHT = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HELPERS
# ============================================================

def save_image(img, name):
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def normalize_for_display(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def preproceso_dicom(dicom_path: Path):
    ds = pydicom.dcmread(str(dicom_path))
    im = ds.pixel_array

    if im.ndim > 2:
        im = np.squeeze(im)

    raw = im.astype(np.float32, copy=False)
    save_image(normalize_for_display(raw), "00_raw")

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        raw = raw.max() + raw.min() - raw
        save_image(normalize_for_display(raw), "01_monochrome1_inverted")

    if CALC_WINDOW:
        im_proc = preprocess_window(
            raw,
            dicom_dataset=ds,
            method=METHOD,
            voi_func=VOI_FUNC,
        )
        save_image(normalize_for_display(im_proc), "02_windowed")
    else:
        params = get_dicom_voi_lut_params(ds)
        im_proc = apply_windowing(
            raw,
            window_width=params["window_width"],
            window_center=params["window_center"],
            voi_func=VOI_FUNC,
            y_min=0,
            y_max=255,
        )
        im_proc = np.rint(np.clip(im_proc, 0, 255)).astype(np.uint8)
        save_image(normalize_for_display(im_proc), "02_windowed")

    if im_proc.ndim == 2:
        rgb = cv2.cvtColor(im_proc.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif im_proc.ndim == 3 and im_proc.shape[2] == 1:
        rgb = cv2.cvtColor(im_proc.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb = im_proc.astype(np.uint8)

    save_image(rgb, "03_rgb")

    h0, w0 = rgb.shape[:2]
    scale = min(IMG_SIZE / h0, IMG_SIZE / w0)
    new_h = max(1, int(round(h0 * scale)))
    new_w = max(1, int(round(w0 * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    save_image(resized, "04_resized")

    return raw, im_proc, rgb, resized


def swint_attention_enhance(img_rgb):
    model = timm.create_model(
        SWIN_MODEL_NAME,
        pretrained=SWIN_PRETRAINED,
        features_only=True,
        out_indices=(1, 2, 3),
    )
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        x = img_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        x = (x - mean) / std

        feat = model(x)[-1]
        attn = feat.mean(dim=1, keepdim=True)
        attn = F.interpolate(
            attn,
            size=(img_rgb.shape[0], img_rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        attn_np = attn.squeeze(0).squeeze(0).detach().cpu().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-6)
        enhanced = img_rgb.astype(np.float32) * (1.0 + SWIN_WEIGHT * attn_np[..., None])
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    save_image(enhanced, "05_swin_enhanced")
    return enhanced


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a una imagen DICOM concreta para inspeccionar. Si no se da, usa la primera de train.",
    )
    args = parser.parse_args()

    if args.image:
        dicom_path = Path(args.image)
    else:
        files = sorted(DICOM_DIR.glob("*"))
        if not files:
            raise FileNotFoundError(f"No hay imágenes en {DICOM_DIR}")
        dicom_path = files[0]

    print(f"[VISUAL] Procesando: {dicom_path}", flush=True)
    raw, im_proc, rgb, resized = preproceso_dicom(dicom_path)
    swint_attention_enhance(resized)

    print(f"[VISUAL] Guardado en: {OUTPUT_DIR}", flush=True)
    print("[VISUAL] Ficheros generados:")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
