import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
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
# PATHS
# ============================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

CSV_PATH = Path(
    "/home/enric_sena/Desktop/prova_enric/vindr_dataset/finding_annotations.csv"
)

DATA_YAML = DATASET / "data.yaml"


# ============================================================
# TRAINING CONFIG
# ============================================================

MODEL = "yolov8n.pt"

IMG_SIZE = 1024
BATCH = 4
WORKERS = 8
EPOCHS = 100
DEVICE = 0

# No fem servir el cache estàndard d'Ultralytics.
CACHE = False


# ============================================================
# RAM CACHE CONFIG
# ============================================================

# Límit màxim del cache:
# 120 GiB = 120 * 1024**3 bytes
RAM_CACHE_LIMIT_GB = 120
RAM_CACHE_LIMIT_BYTES = RAM_CACHE_LIMIT_GB * 1024**3

RAM_CACHE_ENABLED = True

# Threads utilitzats per carregar DICOM durant
# la construcció inicial del cache.
CACHE_BUILD_WORKERS = 8

# Nombre d'imatges que mantenim en procés simultàniament.
CACHE_CHUNK_SIZE = 64

# Comptador global.
# Train + val comparteixen el mateix límit.
RAM_CACHE_USED_BYTES = 0


# ============================================================
# IMAGE DIMENSIONS FROM CSV
# ============================================================

print(
    "[INIT] Carregant dimensions del CSV...",
    flush=True,
)

_annotations = pd.read_csv(
    CSV_PATH
)

IMAGE_DIMS = {
    str(row.image_id): (
        int(row.height),
        int(row.width),
    )
    for row in _annotations.itertuples(
        index=False
    )
}

print(
    f"[INIT] Dimensions carregades per "
    f"{len(IMAGE_DIMS)} imatges",
    flush=True,
)


# ============================================================
# DICOM READING + PREPROCESSING
# ============================================================

def read_and_process_dicom(
    dicom_path,
    imgsz,
):
    """
    Llegeix un DICOM i el converteix a uint8 BGR.

    No crea cap fitxer.
    No modifica el DICOM.
    """

    dicom_path = str(
        dicom_path
    )

    ds = pydicom.dcmread(
        dicom_path
    )

    im = ds.pixel_array

    if im.ndim > 2:
        im = np.squeeze(im)

    # MONOCHROME1:
    # valors alts = més foscos.
    if (
        getattr(
            ds,
            "PhotometricInterpretation",
            "",
        )
        == "MONOCHROME1"
    ):
        im = im.max() - im

    im = im.astype(
        np.float32,
        copy=False,
    )

    # Normalització robusta.
    lo, hi = np.percentile(
        im,
        (1, 99),
    )

    if hi > lo:

        im = np.clip(
            (im - lo)
            / (hi - lo)
            * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )

    else:

        im = np.zeros_like(
            im,
            dtype=np.uint8,
        )

    # Grayscale -> BGR.
    if im.ndim == 2:

        im = cv2.cvtColor(
            im,
            cv2.COLOR_GRAY2BGR,
        )

    # Dimensions originals.
    h0, w0 = im.shape[:2]

    # Resize mantenint aspect ratio.
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

    if (
        new_h,
        new_w,
    ) != (
        h0,
        w0,
    ):

        im = cv2.resize(
            im,
            (
                new_w,
                new_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    im = np.ascontiguousarray(
        im,
        dtype=np.uint8,
    )

    return (
        im,
        (h0, w0),
        im.shape[:2],
    )


# ============================================================
# CUSTOM DICOM YOLO DATASET
# ============================================================

class DICOMYOLODataset(
    YOLODataset
):

    def __init__(
        self,
        *args,
        ram_cache=False,
        ram_cache_name="dataset",
        **kwargs,
    ):

        self.ram_cache_enabled = bool(
            ram_cache
        )

        self.ram_cache_name = str(
            ram_cache_name
        )

        # Índex de la imatge ->
        #
        # (
        #   numpy array,
        #   original shape,
        #   resized shape
        # )
        self._ram_cache = {}

        super().__init__(
            *args,
            **kwargs,
        )

        # Construïm el cache al procés pare,
        # abans de crear els DataLoader workers.
        if self.ram_cache_enabled:

            self._build_ram_cache()

    # ========================================================
    # LABELS
    # ========================================================

    def get_labels(
        self
    ):
        """
        Carrega els labels YOLO .txt.

        Cada bbox és classe 0 = lesion.

        Les imatges No Finding tenen label buit.
        """

        labels = []

        for im_file in self.im_files:

            im_path = Path(
                im_file
            )

            split = (
                im_path.parent.name
            )

            label_path = (
                DATASET
                / "labels"
                / split
                / f"{im_path.stem}.txt"
            )

            image_id = im_path.stem

            if (
                image_id
                not in IMAGE_DIMS
            ):

                raise KeyError(
                    f"No trobo dimensions al CSV "
                    f"per image_id={image_id}"
                )

            h, w = IMAGE_DIMS[
                image_id
            ]

            cls_list = []
            bbox_list = []

            if label_path.exists():

                text = (
                    label_path
                    .read_text()
                    .strip()
                )

                if text:

                    for line in text.splitlines():

                        parts = line.split()

                        if len(parts) != 5:

                            raise ValueError(
                                f"Label incorrecte a "
                                f"{label_path}: {line}"
                            )

                        cls_id = int(
                            float(
                                parts[0]
                            )
                        )

                        x = float(
                            parts[1]
                        )

                        y = float(
                            parts[2]
                        )

                        bw = float(
                            parts[3]
                        )

                        bh = float(
                            parts[4]
                        )

                        cls_list.append(
                            [cls_id]
                        )

                        bbox_list.append(
                            [
                                x,
                                y,
                                bw,
                                bh,
                            ]
                        )

            if cls_list:

                cls = np.asarray(
                    cls_list,
                    dtype=np.float32,
                )

                bboxes = np.asarray(
                    bbox_list,
                    dtype=np.float32,
                )

            else:

                cls = np.zeros(
                    (0, 1),
                    dtype=np.float32,
                )

                bboxes = np.zeros(
                    (0, 4),
                    dtype=np.float32,
                )

            labels.append(
                {
                    "im_file": str(
                        im_file
                    ),
                    "shape": (
                        h,
                        w,
                    ),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        print(
            f"[DICOM DATASET] "
            f"{len(labels)} labels carregats "
            f"({self.ram_cache_name})",
            flush=True,
        )

        return labels

    # ========================================================
    # RAM CACHE
    # ========================================================

    def _build_ram_cache(
        self
    ):

        global RAM_CACHE_USED_BYTES

        total_images = len(
            self.im_files
        )

        if total_images == 0:
            return

        print(
            "\n"
            + "=" * 70,
            flush=True,
        )

        print(
            f"[RAM CACHE] Iniciant cache: "
            f"{self.ram_cache_name}",
            flush=True,
        )

        print(
            f"[RAM CACHE] Imatges: "
            f"{total_images}",
            flush=True,
        )

        print(
            f"[RAM CACHE] Límit global: "
            f"{RAM_CACHE_LIMIT_GB} GiB",
            flush=True,
        )

        print(
            f"[RAM CACHE] Utilitzat abans: "
            f"{RAM_CACHE_USED_BYTES / 1024**3:.2f} GiB",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        cached_count = 0
        processed_count = 0

        for start in range(
            0,
            total_images,
            CACHE_CHUNK_SIZE,
        ):

            if (
                RAM_CACHE_USED_BYTES
                >= RAM_CACHE_LIMIT_BYTES
            ):

                print(
                    "[RAM CACHE] "
                    "Límit de 120 GiB assolit.",
                    flush=True,
                )

                break

            end = min(
                start
                + CACHE_CHUNK_SIZE,
                total_images,
            )

            indices = list(
                range(
                    start,
                    end,
                )
            )

            with ThreadPoolExecutor(
                max_workers=min(
                    CACHE_BUILD_WORKERS,
                    len(indices),
                )
            ) as executor:

                results = executor.map(
                    lambda i:
                        read_and_process_dicom(
                            self.im_files[i],
                            self.imgsz,
                        ),
                    indices,
                )

                for i, result in zip(
                    indices,
                    results,
                ):

                    processed_count += 1

                    (
                        im,
                        original_shape,
                        resized_shape,
                    ) = result

                    image_bytes = int(
                        im.nbytes
                    )

                    # No superem el límit.
                    if (
                        RAM_CACHE_USED_BYTES
                        + image_bytes
                        <= RAM_CACHE_LIMIT_BYTES
                    ):

                        # Read-only al procés pare.
                        im.setflags(
                            write=False
                        )

                        self._ram_cache[
                            i
                        ] = (
                            im,
                            original_shape,
                            resized_shape,
                        )

                        RAM_CACHE_USED_BYTES += (
                            image_bytes
                        )

                        cached_count += 1

                    # Feedback de cada imatge.
                    print(
                        f"[RAM CACHE] "
                        f"Processant "
                        f"{processed_count}/"
                        f"{total_images} "
                        f"| cacheats="
                        f"{cached_count} "
                        f"| RAM="
                        f"{RAM_CACHE_USED_BYTES / 1024**3:.2f}/"
                        f"{RAM_CACHE_LIMIT_GB} GiB",
                        flush=True,
                    )

        print(
            "\n[RAM CACHE] "
            f"Final {self.ram_cache_name}: "
            f"{cached_count}/{total_images} "
            f"imatges cachejades",
            flush=True,
        )

        print(
            "[RAM CACHE] RAM total utilitzada: "
            f"{RAM_CACHE_USED_BYTES / 1024**3:.2f} GiB",
            flush=True,
        )

        if (
            cached_count
            < total_images
        ):

            print(
                "[RAM CACHE] "
                f"{total_images - cached_count} "
                "imatges no han entrat al cache.",
                flush=True,
            )

        print(
            "=" * 70,
            flush=True,
        )

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    def load_image(
        self,
        i,
        *args,
        **kwargs,
    ):
        """
        Carrega des de RAM si la imatge està cachejada.

        Es retorna una còpia perquè les augmentations
        puguin modificar-la sense tocar el cache.
        """

        cached = self._ram_cache.get(
            i
        )

        if cached is not None:

            (
                cached_im,
                original_shape,
                resized_shape,
            ) = cached

            im = cached_im.copy()

            return (
                im,
                original_shape,
                resized_shape,
            )

        # Fallback:
        # si no ha entrat al cache.
        return read_and_process_dicom(
            self.im_files[i],
            self.imgsz,
        )


# ============================================================
# CUSTOM VALIDATOR
# ============================================================

class DICOMValidator(
    DetectionValidator
):

    def build_dataset(
        self,
        img_path,
        mode="val",
        batch=None,
    ):

        # Compatible amb Ultralytics 8.4.144.
        # No utilitzem de_parallel(),
        # perquè no existeix en aquesta versió.

        model = self.model

        if hasattr(
            model,
            "module",
        ):

            model = model.module

        stride = getattr(
            model,
            "stride",
            32,
        )

        if isinstance(
            stride,
            torch.Tensor,
        ):

            stride = int(
                stride.max().item()
            )

        else:

            stride = int(
                stride
            )

        gs = max(
            stride,
            32,
        )

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

            ram_cache=(
                RAM_CACHE_ENABLED
                and self.args.workers > 0
            ),

            ram_cache_name="val",
        )


# ============================================================
# CUSTOM TRAINER
# ============================================================

class DICOMTrainer(
    DetectionTrainer
):

    def build_dataset(
        self,
        img_path,
        mode="train",
        batch=None,
    ):

        # Compatible amb Ultralytics 8.4.144.

        model = self.model

        if hasattr(
            model,
            "module",
        ):

            model = model.module

        stride = getattr(
            model,
            "stride",
            32,
        )

        if isinstance(
            stride,
            torch.Tensor,
        ):

            stride = int(
                stride.max().item()
            )

        else:

            stride = int(
                stride
            )

        gs = max(
            stride,
            32,
        )

        # Test/smoke utilitzen workers=0,
        # per tant no carreguem 120 GiB.
        use_ram_cache = (
            RAM_CACHE_ENABLED
            and self.args.workers > 0
        )

        return DICOMYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(
                mode == "train"
            ),
            hyp=self.args,
            rect=(
                mode == "val"
            ),
            cache=False,
            single_cls=self.args.single_cls,
            stride=gs,
            pad=(
                0.0
                if mode == "train"
                else 0.5
            ),
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=(
                self.args.fraction
                if mode == "train"
                else 1.0
            ),

            ram_cache=use_ram_cache,

            ram_cache_name=mode,
        )

    def get_validator(
        self
    ):

        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
        )

        return DICOMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(
                self.args
            ),
            _callbacks=self.callbacks,
        )


# ============================================================
# CREATE TRAINER
# ============================================================

def create_trainer(
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,
    resume=None,
):

    if resume:

        model = str(
            resume
        )

        resume_value = str(
            resume
        )

    else:

        model = MODEL

        resume_value = False

    return DICOMTrainer(
        overrides={
            "model": model,

            "data": str(
                DATA_YAML
            ),

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


# ============================================================
# MULTIPROCESSING CHECK
# ============================================================

def check_fork():

    method = mp.get_start_method()

    print(
        f"[MP] multiprocessing start method: "
        f"{method}",
        flush=True,
    )

    if method != "fork":

        raise RuntimeError(
            "\n"
            "ERROR: el RAM cache necessita "
            "multiprocessing='fork'.\n"
            f"Mètode actual: {method}\n"
            "Aturo el training per evitar "
            "duplicar el cache de RAM.\n"
        )


# ============================================================
# PIPELINE TEST
# ============================================================

def test_pipeline():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PIPELINE TEST"
    )

    print(
        "=" * 70
    )

    trainer = create_trainer(
        batch=2,
        workers=0,
        device="cpu",
        resume=None,
    )

    loader = trainer.get_dataloader(
        str(
            DATASET
            / "images"
            / "train"
        ),
        batch_size=2,
        rank=-1,
        mode="train",
    )

    print(
        f"[TEST] Dataset creat: "
        f"{len(loader.dataset)} imatges",
        flush=True,
    )

    batch = next(
        iter(loader)
    )

    print(
        f"[TEST] batch['img'].shape = "
        f"{batch['img'].shape}",
        flush=True,
    )

    print(
        f"[TEST] batch['cls'].shape = "
        f"{batch['cls'].shape}",
        flush=True,
    )

    print(
        f"[TEST] batch['bboxes'].shape = "
        f"{batch['bboxes'].shape}",
        flush=True,
    )

    print(
        f"[TEST] batch['batch_idx'].shape = "
        f"{batch['batch_idx'].shape}",
        flush=True,
    )

    print(
        "\nTEST SUPERAT"
    )

    print(
        "No s'ha entrenat res."
    )

    print(
        "No s'han creat PNG/JPG."
    )

    print(
        "No s'han creat fitxers .npy."
    )

    print(
        "=" * 70
    )


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SMOKE TEST"
    )

    print(
        "=" * 70
    )

    trainer = create_trainer(
        batch=2,
        workers=0,
        device="cpu",
        resume=None,
    )

    # Inicialitzar el model.
    trainer.setup_model()

    model = trainer.model

    trainer.set_model_attributes()

    device = trainer.device

    model = model.to(
        device
    )

    model.train()

    print(
        f"[SMOKE] Device: {device}",
        flush=True,
    )

    loader = trainer.get_dataloader(
        str(
            DATASET
            / "images"
            / "train"
        ),
        batch_size=2,
        rank=-1,
        mode="train",
    )

    batch = next(
        iter(loader)
    )

    print(
        f"[SMOKE] Batch: "
        f"{batch['img'].shape}",
        flush=True,
    )

    batch = trainer.preprocess_batch(
        batch
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # Forward + loss.
    loss, loss_items = model.loss(
        batch
    )

    print(
        f"[SMOKE] Loss raw shape: "
        f"{tuple(loss.shape)}",
        flush=True,
    )

    print(
        f"[SMOKE] Loss items: "
        f"{loss_items.detach().cpu().tolist()}",
        flush=True,
    )

    # Convertim a escalar.
    if loss.numel() != 1:

        loss_scalar = loss.sum()

    else:

        loss_scalar = loss.reshape(
            ()
        )

    print(
        f"[SMOKE] Loss scalar: "
        f"{loss_scalar.item():.6f}",
        flush=True,
    )

    # Backward.
    loss_scalar.backward()

    # Comprovar gradients.
    gradients_found = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            if torch.isfinite(
                parameter.grad
            ).all():

                gradients_found = True

                break

    if not gradients_found:

        raise RuntimeError(
            "SMOKE TEST: "
            "no s'han trobat gradients vàlids."
        )

    # Optimizer.
    optimizer.step()

    print(
        "\nSMOKE TEST SUPERAT!"
    )

    print(
        "DICOM -> batch -> model -> "
        "loss -> backward -> optimizer OK."
    )

    print(
        "=" * 70
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    resume=None
):

    # Comprovem fork abans de construir
    # el cache gran de RAM.
    check_fork()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"Model:       {MODEL}",
        flush=True,
    )

    print(
        f"Image size:  {IMG_SIZE}",
        flush=True,
    )

    print(
        f"Batch:       {BATCH}",
        flush=True,
    )

    print(
        f"Workers:     {WORKERS}",
        flush=True,
    )

    print(
        f"Epochs:      {EPOCHS}",
        flush=True,
    )

    print(
        f"Device:      {DEVICE}",
        flush=True,
    )

    print(
        f"RAM cache:   {RAM_CACHE_LIMIT_GB} GiB",
        flush=True,
    )

    if resume:

        print(
            f"Resume:      {resume}",
            flush=True,
        )

    else:

        print(
            "Resume:      NO",
            flush=True,
        )

    print(
        "=" * 70
    )

    trainer = create_trainer(
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        resume=resume,
    )

    trainer.args.epochs = EPOCHS

    trainer.train()


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
        help=(
            "Path to Ultralytics "
            "last.pt checkpoint."
        ),
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