import argparse
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator


# ============================================================
# CONFIGURACIÓN
# ============================================================

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


# ============================================================
# INFORMACIÓN DE LAS IMÁGENES
# ============================================================

print("[INIT] Carregant CSV...", flush=True)

df = pd.read_csv(CSV_PATH)

image_shapes = (
    df[["image_id", "height", "width"]]
    .drop_duplicates("image_id")
    .set_index("image_id")[["height", "width"]]
    .to_dict("index")
)

print(
    f"[INIT] Imatges al CSV: {len(image_shapes)}",
    flush=True,
)


# ============================================================
# DATASET DICOM
# ============================================================

class DICOMYOLODataset(YOLODataset):

    def get_labels(self):

        labels = []

        for im_file in self.im_files:

            im_path = Path(im_file)
            image_id = im_path.stem

            if image_id not in image_shapes:
                raise RuntimeError(
                    f"No hi ha dimensions al CSV per {image_id}"
                )

            h = int(image_shapes[image_id]["height"])
            w = int(image_shapes[image_id]["width"])

            split = im_path.parent.name

            label_file = (
                DATASET
                / "labels"
                / split
                / f"{image_id}.txt"
            )

            if (
                not label_file.exists()
                or label_file.stat().st_size == 0
            ):

                cls = np.zeros(
                    (0, 1),
                    dtype=np.float32,
                )

                bboxes = np.zeros(
                    (0, 4),
                    dtype=np.float32,
                )

            else:

                data = np.loadtxt(
                    label_file,
                    dtype=np.float32,
                    ndmin=2,
                )

                cls = data[:, 0:1]
                bboxes = data[:, 1:5]

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
            f"[DICOM DATASET] {len(labels)} labels carregats",
            flush=True,
        )

        return labels

    def load_image(
        self,
        i,
        rect_mode=True,
        resize_short=False,
    ):

        f = self.im_files[i]

        print(
            f"[DICOM] Carregant {Path(f).name}",
            flush=True,
        )

        ds = pydicom.dcmread(f)

        im = ds.pixel_array

        h0, w0 = im.shape[:2]

        im = im.astype(np.float32)

        if getattr(
            ds,
            "PhotometricInterpretation",
            "",
        ) == "MONOCHROME1":

            im = im.max() - im

        lo, hi = np.percentile(
            im,
            (1, 99),
        )

        if hi > lo:

            im = np.clip(
                (im - lo) / (hi - lo),
                0,
                1,
            )

        else:

            im = np.zeros_like(im)

        im = (
            im * 255
        ).astype(np.uint8)

        im = cv2.cvtColor(
            im,
            cv2.COLOR_GRAY2BGR,
        )

        r = self.imgsz / max(
            h0,
            w0,
        )

        if r != 1:

            new_w = int(w0 * r)
            new_h = int(h0 * r)

            im = cv2.resize(
                im,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

        return (
            im,
            (h0, w0),
            im.shape[:2],
        )


# ============================================================
# VALIDATOR
# ============================================================

class DICOMValidator(DetectionValidator):

    def build_dataset(
        self,
        img_path,
        mode="val",
        batch=None,
    ):

        return DICOMYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=False,
            single_cls=self.args.single_cls,
            stride=32,
            pad=0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            fraction=1.0,
            data=self.data,
        )


# ============================================================
# TRAINER
# ============================================================

class DICOMTrainer(DetectionTrainer):

    def build_dataset(
        self,
        img_path,
        mode="train",
        batch=None,
    ):

        return DICOMYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=(mode == "val"),
            cache=False,
            single_cls=self.args.single_cls,
            stride=32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            fraction=(
                self.args.fraction
                if mode == "train"
                else 1.0
            ),
            data=self.data,
        )

    def get_validator(self):

        return DICOMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


# ============================================================
# CREAR TRAINER
# ============================================================

def create_trainer(
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,
    resume=None,
):

    if resume is not None:
        model = str(resume)
        resume_value = str(resume)
    else:
        model = MODEL
        resume_value = False

    return DICOMTrainer(
        overrides={
            "model": model,
            "data": str(DATA_YAML),
            "task": "detect",
            "imgsz": IMG_SIZE,
            "batch": batch,
            "workers": workers,
            "device": device,
            "cache": CACHE,
            "resume": resume_value,
            "mosaic": 0,
            "mixup": 0,
            "copy_paste": 0,
        }
    )


# ============================================================
# TEST DEL DATASET
# ============================================================

def test_pipeline():

    print()
    print("=" * 60)
    print("TEST PIPELINE")
    print("=" * 60)

    trainer = create_trainer(
        batch=2,
        workers=0,
        device="cpu",
    )

    dataset = trainer.build_dataset(
        trainer.data["train"],
        mode="val",
        batch=2,
    )

    print(
        f"[TEST] Dataset: {len(dataset)} imatges",
        flush=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    batch = next(iter(loader))

    print(
        f"[TEST] img: {batch['img'].shape}",
        flush=True,
    )

    print(
        f"[TEST] cls: {batch['cls'].shape}",
        flush=True,
    )

    print(
        f"[TEST] bboxes: {batch['bboxes'].shape}",
        flush=True,
    )

    print(
        f"[TEST] batch_idx: "
        f"{batch['batch_idx'].shape}",
        flush=True,
    )

    print()
    print("TEST SUPERAT")
    print("No s'ha entrenat res.")
    print("No s'han creat PNG/JPG.")
    print("No s'han creat .npy.")


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test():

    print()
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA no està disponible"
        )

    device = torch.device(
        f"cuda:{DEVICE}"
    )

    print(
        f"[SMOKE] GPU: "
        f"{torch.cuda.get_device_name(DEVICE)}",
        flush=True,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    print(
        "[SMOKE] Creant trainer...",
        flush=True,
    )

    trainer = create_trainer(
        batch=2,
        workers=0,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "[SMOKE] Construint dataset...",
        flush=True,
    )

    dataset = trainer.build_dataset(
        trainer.data["train"],
        mode="val",
        batch=2,
    )

    print(
        f"[SMOKE] Dataset: "
        f"{len(dataset)} imatges",
        flush=True,
    )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    print(
        "[SMOKE] Carregant batch...",
        flush=True,
    )

    batch = next(iter(loader))

    print(
        f"[SMOKE] Input CPU: "
        f"{batch['img'].shape}",
        flush=True,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "[SMOKE] Configurant model...",
        flush=True,
    )

    trainer.setup_model()

    model = trainer.model

    if model is None:

        raise RuntimeError(
            "trainer.model és None"
        )

    print(
        f"[SMOKE] Tipus model: "
        f"{type(model).__name__}",
        flush=True,
    )

    trainer.set_model_attributes()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    model = model.to(device)

    model.train()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    batch = trainer.preprocess_batch(
        batch
    )

    print(
        f"[SMOKE] Input GPU: "
        f"{batch['img'].shape}",
        flush=True,
    )

    print(
        f"[SMOKE] dtype: "
        f"{batch['img'].dtype}",
        flush=True,
    )

    print(
        f"[SMOKE] device: "
        f"{batch['img'].device}",
        flush=True,
    )

    print(
        f"[SMOKE] model.args: "
        f"{type(model.args)}",
        flush=True,
    )

    # --------------------------------------------------------
    # MEMÒRIA
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    mem_before = (
        torch.cuda.memory_allocated(device)
        / 1024**3
    )

    print(
        f"[SMOKE] GPU memory abans: "
        f"{mem_before:.2f} GB",
        flush=True,
    )

    # --------------------------------------------------------
    # FORWARD + LOSS
    # --------------------------------------------------------

    print(
        "[SMOKE] Forward + loss...",
        flush=True,
    )

    loss, loss_items = model.loss(
        batch
    )

    # --------------------------------------------------------
    # LOSS ESCALAR
    # --------------------------------------------------------

    if not torch.is_tensor(loss):

        raise RuntimeError(
            f"Loss inesperada: {type(loss)}"
        )

    loss_raw_shape = tuple(
        loss.shape
    )

    if loss.numel() != 1:

        loss_scalar = loss.sum()

    else:

        loss_scalar = loss.reshape(())

    if loss_scalar.numel() != 1:

        raise RuntimeError(
            f"No s'ha pogut convertir la loss "
            f"a escalar. Shape: {loss.shape}"
        )

    loss_value = (
        loss_scalar
        .detach()
        .cpu()
        .item()
    )

    if torch.is_tensor(loss_items):

        loss_items_print = (
            loss_items
            .detach()
            .float()
            .cpu()
            .flatten()
            .tolist()
        )

    else:

        loss_items_print = loss_items

    print(
        f"[SMOKE] Loss raw shape: "
        f"{loss_raw_shape}",
        flush=True,
    )

    print(
        f"[SMOKE] Loss escalar: "
        f"{loss_value:.6f}",
        flush=True,
    )

    print(
        f"[SMOKE] Loss items: "
        f"{loss_items_print}",
        flush=True,
    )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    print(
        "[SMOKE] Backward...",
        flush=True,
    )

    loss_scalar.backward()

    # --------------------------------------------------------
    # GRADIENTS
    # --------------------------------------------------------

    n_grad = sum(
        1
        for p in model.parameters()
        if p.grad is not None
    )

    print(
        f"[SMOKE] Paràmetres amb gradient: "
        f"{n_grad}",
        flush=True,
    )

    if n_grad == 0:

        raise RuntimeError(
            "No s'han obtingut gradients"
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    print(
        "[SMOKE] Optimizer step...",
        flush=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
    )

    optimizer.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # MEMÒRIA
    # --------------------------------------------------------

    mem_after = (
        torch.cuda.memory_allocated(device)
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated(device)
        / 1024**3
    )

    print(
        f"[SMOKE] GPU memory després: "
        f"{mem_after:.2f} GB",
        flush=True,
    )

    print(
        f"[SMOKE] GPU memory peak: "
        f"{peak:.2f} GB",
        flush=True,
    )

    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SMOKE TEST SUPERAT")
    print("=" * 60)

    print(
        "DICOM -> batch -> GPU -> forward -> loss "
        "-> backward -> optimizer OK",
        flush=True,
    )

    print(
        "No s'han creat PNG/JPG.",
        flush=True,
    )

    print(
        "No s'han creat fitxers .npy.",
        flush=True,
    )


# ============================================================
# TRAINING
# ============================================================

def train(resume=None):

    print()
    print("=" * 60)
    print("TRAINING COMPLET")
    print("=" * 60)

    print(
        f"[TRAIN] Model: "
        f"{resume if resume else MODEL}",
        flush=True,
    )

    print(
        f"[TRAIN] Dataset: {DATASET}",
        flush=True,
    )

    print(
        f"[TRAIN] Image size: {IMG_SIZE}",
        flush=True,
    )

    print(
        f"[TRAIN] Batch: {BATCH}",
        flush=True,
    )

    print(
        f"[TRAIN] Workers: {WORKERS}",
        flush=True,
    )

    print(
        f"[TRAIN] Epochs: {EPOCHS}",
        flush=True,
    )

    print(
        f"[TRAIN] Device: {DEVICE}",
        flush=True,
    )

    print(
        "[TRAIN] Cache: False",
        flush=True,
    )

    if resume:

        print(
            f"[TRAIN] Resume: {resume}",
            flush=True,
        )

    else:

        print(
            "[TRAIN] Resume: False",
            flush=True,
        )

    trainer = create_trainer(
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        resume=resume,
    )

    trainer.args.epochs = EPOCHS

    print(
        "[TRAIN] Iniciant entrenament...",
        flush=True,
    )

    trainer.train()

    print()
    print("=" * 60)
    print("TRAINING FINALITZAT")
    print("=" * 60)

    print(
        f"[TRAIN] Resultats: "
        f"{trainer.save_dir}",
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "test",
            "smoke",
            "train",
        ],
        required=True,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path al checkpoint last.pt per reprendre l'entrenament",
    )

    args = parser.parse_args()

    if args.mode == "test":

        test_pipeline()

    elif args.mode == "smoke":

        smoke_test()

    elif args.mode == "train":

        train(
            resume=args.resume
        )


if __name__ == "__main__":

    main()