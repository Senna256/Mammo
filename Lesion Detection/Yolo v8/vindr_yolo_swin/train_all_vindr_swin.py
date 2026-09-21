from __future__ import annotations

import argparse
import multiprocessing as mp
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

try:
    import timm
except ImportError:  # optional; only needed for the visual Swin stage
    timm = None

from mammo_prep.windowing import (
    apply_windowing,
    get_dicom_voi_lut_params,
    preprocess_window,
)


# ============================================================================
# CONFIGURACIÓN BASE Y TESTS (mantiene el comportamiento de la versión anterior)
# ============================================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

CSV_PATH = Path(
    "/home/enric_sena/Desktop/prova_enric/vindr_dataset/finding_annotations.csv"
)

DATA_YAML = DATASET / "data.yaml"
MODEL = "yolov8n.pt"
IMG_SIZE = 1024
BATCH = 4
WORKERS = 8
EPOCHS = 100
DEVICE = 0
CACHE = False

# Swin is optional and only used for visual debugging / enhancement preview
SWIN_MODEL_NAME = "swin_tiny_patch4_window7_224"
SWIN_PRETRAINED = True
SWIN_ENABLED = False
SWIN_WEIGHT = 0.35
SWIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output folder for visualization of preprocessing stages
VISUAL_DEBUG_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo_swin/debug_outputs"
)


print("[INIT] Carregant dimensions del CSV...", flush=True)
_annotations = pd.read_csv(CSV_PATH)
IMAGE_DIMS = {
    str(row.image_id): (int(row.height), int(row.width))
    for row in _annotations.itertuples(index=False)
}
print(
    f"[INIT] Dimensions carregades per {len(IMAGE_DIMS)} imatges",
    flush=True,
)


# ============================================================================
# VISUAL DEBUG: guarda imágenes de cada paso del preprocessing
# ============================================================================

def save_debug_image(img, name, subdir="pipeline"):
    """Guarda una imagen RGB/uint8 en disco para inspección visual."""
    out_dir = VISUAL_DEBUG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def debug_preprocessing_pipeline(dicom_path, method, calc_window, voi_func):
    """Muestra/guarda cada etapa del preprocessing para inspeccionar visualmente."""
    dicom_path = Path(dicom_path)
    ds = pydicom.dcmread(str(dicom_path))
    original = ds.pixel_array

    if original.ndim > 2:
        original = np.squeeze(original)

    original_u8 = np.clip(original.astype(np.float32), 0, 255).astype(np.uint8)
    save_debug_image(original_u8, "00_raw", "pipeline")

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        original = original.max() + original.min() - original
        save_debug_image(np.clip(original.astype(np.float32), 0, 255).astype(np.uint8), "01_mono1_inverted", "pipeline")

    if calc_window:
        processed = preprocess_window(
            original.astype(np.float32),
            dicom_dataset=ds,
            method=method,
            voi_func=voi_func,
        )
        save_debug_image(np.clip(processed, 0, 255).astype(np.uint8), "02_windowed", "pipeline")
    else:
        params = get_dicom_voi_lut_params(ds)
        processed = apply_windowing(
            original.astype(np.float32),
            window_width=params["window_width"],
            window_center=params["window_center"],
            voi_func=voi_func,
            y_min=0,
            y_max=255,
        )
        processed = np.rint(np.clip(processed, 0, 255)).astype(np.uint8)
        save_debug_image(processed, "02_windowed_no_calc", "pipeline")

    if processed.ndim == 2:
        rgb = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif processed.ndim == 3 and processed.shape[2] == 1:
        rgb = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        rgb = processed.astype(np.uint8)

    save_debug_image(rgb, "03_rgb", "pipeline")

    h0, w0 = rgb.shape[:2]
    scale = min(IMG_SIZE / h0, IMG_SIZE / w0)
    new_h = max(1, int(round(h0 * scale)))
    new_w = max(1, int(round(w0 * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    save_debug_image(resized, "04_resized", "pipeline")

    if SWIN_ENABLED:
        if timm is None:
            print("[DEBUG] Swin habilitado pero no está instalado 'timm'.", flush=True)
        else:
            model = timm.create_model(
                SWIN_MODEL_NAME,
                pretrained=SWIN_PRETRAINED,
                features_only=True,
                out_indices=(1, 2, 3),
            )
            model.to(SWIN_DEVICE)
            model.eval()

            with torch.no_grad():
                x = resized.astype(np.float32) / 255.0
                x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(SWIN_DEVICE)
                mean = torch.tensor([0.485, 0.456, 0.406], device=SWIN_DEVICE).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=SWIN_DEVICE).view(1, 3, 1, 1)
                x = (x - mean) / std
                feat = model(x)[-1]
                attn = feat.mean(dim=1, keepdim=True)
                attn = torch.nn.functional.interpolate(
                    attn,
                    size=(resized.shape[0], resized.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                attn_np = attn.squeeze(0).squeeze(0).detach().cpu().numpy()
                attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-6)
                enhanced = resized.astype(np.float32) * (1.0 + SWIN_WEIGHT * attn_np[..., None])
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                save_debug_image(enhanced, "05_swin_enhanced", "pipeline")

    print(f"[DEBUG] imagens guardades a: {VISUAL_DEBUG_DIR / 'pipeline'}", flush=True)
    return resized


# ============================================================================
# PREPROCESADO DICOM ORIGINAL (igual que la versión anterior)
# ============================================================================

def read_and_process_dicom(
    dicom_path,
    imgsz,
    method="breast_tissue",
    calc_window=True,
    voi_func="LINEAR",
):
    """Llegeix un DICOM i aplica el preprocessing de mammo_prep."""
    dicom_path = str(dicom_path)
    ds = pydicom.dcmread(dicom_path)
    im = ds.pixel_array

    if im.ndim > 2:
        im = np.squeeze(im)

    im = im.astype(np.float32, copy=False)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        im = im.max() + im.min() - im

    if calc_window:
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
        im = np.rint(np.clip(im, 0, 255)).astype(np.uint8)

    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.ndim == 3 and im.shape[2] == 1:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.ndim != 3 or im.shape[2] != 3:
        raise RuntimeError(
            f"Shape inesperat després del preprocessing: {im.shape}"
        )

    h0, w0 = im.shape[:2]
    scale = min(imgsz / h0, imgsz / w0)
    new_h = max(1, int(round(h0 * scale)))
    new_w = max(1, int(round(w0 * scale)))

    if (new_h, new_w) != (h0, w0):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    im = np.ascontiguousarray(im, dtype=np.uint8)
    return im, (h0, w0), im.shape[:2]


# ============================================================================
# CUSTOM DATASET (identical to the original training code)
# ============================================================================

class DICOMYOLODataset(YOLODataset):
    def __init__(
        self,
        *args,
        ram_cache=False,
        ram_cache_name="dataset",
        windowing_method="breast_tissue",
        calc_window=True,
        voi_func="LINEAR",
        **kwargs,
    ):
        self.ram_cache_enabled = bool(ram_cache)
        self.ram_cache_name = str(ram_cache_name)
        self.windowing_method = windowing_method
        self.calc_window = bool(calc_window)
        self.voi_func = str(voi_func)
        self._ram_cache = {}
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = []

        for im_file in self.im_files:
            im_path = Path(im_file)
            split = im_path.parent.name
            label_path = DATASET / "labels" / split / f"{im_path.stem}.txt"
            image_id = im_path.stem

            if image_id not in IMAGE_DIMS:
                raise KeyError(
                    f"No trobo dimensions al CSV per image_id={image_id}"
                )

            h, w = IMAGE_DIMS[image_id]
            cls_list = []
            bbox_list = []

            if label_path.exists():
                text = label_path.read_text().strip()
                if text:
                    for line in text.splitlines():
                        parts = line.split()
                        if len(parts) != 5:
                            raise ValueError(
                                f"Format de label invàlid a {label_path}: {line}"
                            )
                        cls_list.append([float(parts[0])])
                        bbox_list.append(
                            [
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
                            ]
                        )

            if cls_list:
                cls = np.asarray(cls_list, dtype=np.float32)
                bboxes = np.asarray(bbox_list, dtype=np.float32)
            else:
                cls = np.zeros((0, 1), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)

            labels.append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        print(
            f"[DICOM DATASET] {len(labels)} labels carregats ({self.ram_cache_name})",
            flush=True,
        )
        return labels

    def load_image(self, i, *args, **kwargs):
        return read_and_process_dicom(
            self.im_files[i],
            self.imgsz,
            method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
        )


# ============================================================================
# CUSTOM VALIDATOR / TRAINER
# ============================================================================

class DICOMValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        stride = getattr(model, "stride", 32)
        if isinstance(stride, torch.Tensor):
            stride = int(stride.max().item())
        else:
            stride = int(stride)

        gs = max(stride, 32)

        return DICOMYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=False,
            single_cls=self.args.single_cls,
            stride=gs,
            pad=0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            ram_cache=False,
            ram_cache_name="val",
            windowing_method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
        )


class DICOMTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        stride = getattr(model, "stride", 32)
        if isinstance(stride, torch.Tensor):
            stride = int(stride.max().item())
        else:
            stride = int(stride)

        gs = max(stride, 32)

        return DICOMYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=(mode == "val"),
            cache=False,
            single_cls=self.args.single_cls,
            stride=gs,
            pad=(0.0 if mode == "train" else 0.5),
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=(self.args.fraction if mode == "train" else 1.0),
            ram_cache=False,
            ram_cache_name=mode,
            windowing_method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
        )

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        validator = DICOMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.windowing_method = self.windowing_method
        validator.calc_window = self.calc_window
        validator.voi_func = self.voi_func
        return validator


# ============================================================================
# TRAINER FACTORY
# ============================================================================

def create_trainer(
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,
    resume=None,
    windowing_method="breast_tissue",
    calc_window=True,
    voi_func="LINEAR",
):
    if resume:
        model = str(resume)
        resume_value = str(resume)
    else:
        model = MODEL
        resume_value = False

    trainer = DICOMTrainer(
        overrides={
            "model": model,
            "data": str(DATA_YAML),
            "task": "detect",
            "imgsz": IMG_SIZE,
            "batch": batch,
            "workers": workers,
            "device": device,
            "cache": CACHE,
            "mosaic": 0,
            "mixup": 0,
            "copy_paste": 0,
            "resume": resume_value,
        }
    )

    trainer.windowing_method = windowing_method
    trainer.calc_window = calc_window
    trainer.voi_func = voi_func
    return trainer


# ============================================================================
# MULTIPROCESSING CHECK
# ============================================================================

def check_fork():
    method = mp.get_start_method()
    print(f"[MP] multiprocessing start method: {method}", flush=True)
    if method != "fork":
        raise RuntimeError(
            "\n"
            "ERROR: el RAM cache necessita multiprocessing='fork'.\n"
            f"Mètode actual: {method}\n"
            "Aturo el training per evitar duplicar el cache de RAM.\n"
        )


# ============================================================================
# PIPELINE TEST
# ============================================================================

def test_pipeline(windowing_method, calc_window, voi_func, visualize=False):
    print("\n" + "=" * 70)
    print("PIPELINE TEST")
    print("=" * 70)
    print(f"Method:      {windowing_method}", flush=True)
    print(f"Calc window: {calc_window}", flush=True)
    print(f"VOI function:{voi_func}", flush=True)

    trainer = create_trainer(
        batch=1,
        workers=0,
        device="cpu",
        resume=None,
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
    )

    loader = trainer.get_dataloader(
        str(DATASET / "images" / "train"),
        batch_size=1,
        rank=-1,
        mode="val",
    )

    print(f"[TEST] Dataset creat: {len(loader.dataset)} imatges", flush=True)
    batch = next(iter(loader))
    print(f"[TEST] batch['img'].shape = {batch['img'].shape}", flush=True)
    print(f"[TEST] batch['cls'].shape = {batch['cls'].shape}", flush=True)
    print(f"[TEST] batch['bboxes'].shape = {batch['bboxes'].shape}", flush=True)
    print(f"[TEST] batch['batch_idx'].shape = {batch['batch_idx'].shape}", flush=True)

    if visualize:
        sample_path = next(iter((DATASET / "images" / "train").glob("*")))
        print(f"[DEBUG] Generant visualització per: {sample_path}", flush=True)
        debug_preprocessing_pipeline(
            sample_path,
            method=windowing_method,
            calc_window=calc_window,
            voi_func=voi_func,
        )

    print("\nTEST SUPERAT")
    print("Preprocessing mammo_prep OK.")
    print("No s'ha entrenat res.")
    print("No s'han creat PNG/JPG.")
    print("No s'han creat fitxers .npy.")
    print("=" * 70)


# ============================================================================
# SMOKE TEST
# ============================================================================

def smoke_test(windowing_method, calc_window, voi_func):
    print("\n" + "=" * 70)
    print("SMOKE TEST")
    print("=" * 70)
    trainer = create_trainer(
        batch=2,
        workers=0,
        device="cuda",
        resume=None,
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
    )

    loader = trainer.get_dataloader(
        str(DATASET / "images" / "train"),
        batch_size=2,
        rank=-1,
        mode="train",
    )

    batch = next(iter(loader))
    trainer.setup_model()
    model = trainer.model

    if model is None:
        raise RuntimeError("trainer.model és None")

    trainer.set_model_attributes()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    batch = trainer.preprocess_batch(batch)

    print(f"[SMOKE] Input GPU: {batch['img'].shape}", flush=True)
    print(f"[SMOKE] dtype: {batch['img'].dtype}", flush=True)
    print(f"[SMOKE] device: {batch['img'].device}", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print("[SMOKE] Forward + loss...", flush=True)
    loss, loss_items = model.loss(batch)

    if not torch.is_tensor(loss):
        raise RuntimeError(f"Loss inesperada: {type(loss)}")

    loss_raw_shape = tuple(loss.shape)
    if loss.numel() != 1:
        loss_scalar = loss.sum()
    else:
        loss_scalar = loss.reshape(())

    if loss_scalar.numel() != 1:
        raise RuntimeError(
            f"No s'ha pogut convertir la loss a escalar. Shape: {loss.shape}"
        )

    print(f"[SMOKE] Loss raw shape: {loss_raw_shape}", flush=True)
    print(f"[SMOKE] Loss escalar: {loss_scalar.detach().cpu().item():.6f}", flush=True)

    if torch.is_tensor(loss_items):
        loss_items_print = loss_items.detach().float().cpu().flatten().tolist()
    else:
        loss_items_print = loss_items

    print(f"[SMOKE] Loss items: {loss_items_print}", flush=True)
    print("[SMOKE] Backward...", flush=True)
    loss_scalar.backward()

    gradients_found = False
    for parameter in model.parameters():
        if parameter.grad is not None:
            if torch.isfinite(parameter.grad).all():
                gradients_found = True
                break

    if not gradients_found:
        raise RuntimeError("SMOKE TEST: no s'han trobat gradients vàlids.")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer.step()

    print("\nSMOKE TEST SUPERAT!")
    print("DICOM -> mammo_prep -> batch -> model -> loss -> backward -> optimizer OK.")
    print("No s'han creat PNG/JPG.")
    print("No s'han creat fitxers .npy.")
    print("=" * 70)


# ============================================================================
# TRAINING
# ============================================================================

def train(windowing_method, calc_window, voi_func, resume=None):
    check_fork()

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Model:       {MODEL}", flush=True)
    print(f"Image size:  {IMG_SIZE}", flush=True)
    print(f"Batch:       {BATCH}", flush=True)
    print(f"Workers:     {WORKERS}", flush=True)
    print(f"Epochs:      {EPOCHS}", flush=True)
    print(f"Device:      {DEVICE}", flush=True)
    print(f"Method:      {windowing_method}", flush=True)
    print(f"Calc window: {calc_window}", flush=True)
    print(f"VOI function:{voi_func}", flush=True)
    print(f"RAM cache:   120 GiB", flush=True)

    if resume:
        print(f"Resume:      {resume}", flush=True)
    else:
        print("Resume:      NO", flush=True)

    print("=" * 70)

    trainer = create_trainer(
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        resume=resume,
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
    )

    trainer.args.epochs = EPOCHS
    print("[TRAIN] Iniciant entrenament...", flush=True)
    trainer.train()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["test", "smoke", "train"],
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=[
            "dicom",
            "percentile_1_99",
            "percentile_2_98",
            "percentile_5_95",
            "breast_tissue",
            "statistical",
            "statistical_wide",
            "full_range",
            "histogram_peak",
        ],
        default="breast_tissue",
    )
    parser.add_argument(
        "--calc-window",
        dest="calc_window",
        action="store_true",
    )
    parser.add_argument(
        "--no-calc-window",
        dest="calc_window",
        action="store_false",
    )
    parser.set_defaults(calc_window=True)

    parser.add_argument(
        "--voi-func",
        choices=["LINEAR", "LINEAR_EXACT", "SIGMOID"],
        default="LINEAR",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Guarda imatges de cada etapa del preprocessament per inspecció visual.",
    )

    args = parser.parse_args()

    if args.mode == "test":
        test_pipeline(
            windowing_method=args.method,
            calc_window=args.calc_window,
            voi_func=args.voi_func,
            visualize=args.visualize,
        )
    elif args.mode == "smoke":
        smoke_test(
            windowing_method=args.method,
            calc_window=args.calc_window,
            voi_func=args.voi_func,
        )
    elif args.mode == "train":
        train(
            windowing_method=args.method,
            calc_window=args.calc_window,
            voi_func=args.voi_func,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
