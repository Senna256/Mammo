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

from mammo_prep.windowing import (
    preprocess_window,
    get_dicom_voi_lut_params,
    apply_windowing,
)


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

CACHE = False


# ============================================================
# RAM CACHE CONFIG
# ============================================================

RAM_CACHE_LIMIT_GB = 120
RAM_CACHE_LIMIT_BYTES = RAM_CACHE_LIMIT_GB * 1024**3

RAM_CACHE_ENABLED = True

CACHE_BUILD_WORKERS = 8
CACHE_CHUNK_SIZE = 64

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
    method="breast_tissue",
    calc_window=True,
    voi_func="LINEAR",
):
    """
    Llegeix un DICOM i aplica el preprocessing de mammo_prep.

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

    im = im.astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # MONOCHROME1
    # --------------------------------------------------------

    if (
        getattr(
            ds,
            "PhotometricInterpretation",
            "",
        )
        == "MONOCHROME1"
    ):
        im = (
            im.max()
            + im.min()
            - im
        )

    # --------------------------------------------------------
    # WINDOWING
    # --------------------------------------------------------

    if calc_window:

        im = preprocess_window(
            im,
            dicom_dataset=ds,
            method=method,
            voi_func=voi_func,
        )

    else:

        params = get_dicom_voi_lut_params(
            ds
        )

        im = apply_windowing(
            im,
            window_width=params[
                "window_width"
            ],
            window_center=params[
                "window_center"
            ],
            voi_func=voi_func,
            y_min=0,
            y_max=255,
        )

        im = np.rint(
            np.clip(
                im,
                0,
                255,
            )
        ).astype(
            np.uint8
        )

    # --------------------------------------------------------
    # GRAYSCALE -> BGR
    # --------------------------------------------------------

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

    elif (
        im.ndim != 3
        or im.shape[2] != 3
    ):

        raise RuntimeError(
            f"Shape inesperat després "
            f"del preprocessing: {im.shape}"
        )

    # --------------------------------------------------------
    # ORIGINAL DIMENSIONS
    # --------------------------------------------------------

    h0, w0 = im.shape[:2]

    # --------------------------------------------------------
    # RESIZE PRESERVING ASPECT RATIO
    # --------------------------------------------------------

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
        windowing_method="breast_tissue",
        calc_window=True,
        voi_func="LINEAR",
        **kwargs,
    ):

        self.ram_cache_enabled = bool(
            ram_cache
        )

        self.ram_cache_name = str(
            ram_cache_name
        )

        self.windowing_method = (
            windowing_method
        )

        self.calc_window = bool(
            calc_window
        )

        self.voi_func = str(
            voi_func
        )

        self._ram_cache = {}

        super().__init__(
            *args,
            **kwargs,
        )

        if self.ram_cache_enabled:

            self._build_ram_cache()

    # ========================================================
    # LABELS
    # ========================================================

    def get_labels(
        self
    ):

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
                                f"Label incorrecte "
                                f"a {label_path}: "
                                f"{line}"
                            )

                        cls_list.append(
                            [
                                float(parts[0])
                            ]
                        )

                        bbox_list.append(
                            [
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
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
                            method=self.windowing_method,
                            calc_window=self.calc_window,
                            voi_func=self.voi_func,
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

                    if (
                        RAM_CACHE_USED_BYTES
                        + image_bytes
                        <= RAM_CACHE_LIMIT_BYTES
                    ):

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

        return read_and_process_dicom(
            self.im_files[i],
            self.imgsz,
            method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
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
            windowing_method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
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
            windowing_method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
        )

    def get_validator(
        self
    ):

        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
        )

        validator = DICOMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(
                self.args
            ),
            _callbacks=self.callbacks,
        )

        validator.windowing_method = (
            self.windowing_method
        )

        validator.calc_window = (
            self.calc_window
        )

        validator.voi_func = (
            self.voi_func
        )

        return validator


# ============================================================
# CREATE TRAINER
# ============================================================

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

        model = str(
            resume
        )

        resume_value = str(
            resume
        )

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

    trainer.windowing_method = (
        windowing_method
    )

    trainer.calc_window = (
        calc_window
    )

    trainer.voi_func = (
        voi_func
    )

    return trainer


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

def test_pipeline(
    windowing_method,
    calc_window,
    voi_func,
):

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

    print(
        f"Method:      {windowing_method}",
        flush=True,
    )

    print(
        f"Calc window: {calc_window}",
        flush=True,
    )

    print(
        f"VOI function:{voi_func}",
        flush=True,
    )

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
        str(
            DATASET
            / "images"
            / "train"
        ),
        batch_size=1,
        rank=-1,
        mode="val",
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
        "Preprocessing mammo_prep OK."
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

def smoke_test(
    windowing_method,
    calc_window,
    voi_func,
):

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
        device="cuda",
        resume=None,
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
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

    trainer.setup_model()

    model = trainer.model

    if model is None:

        raise RuntimeError(
            "trainer.model és None"
        )

    trainer.set_model_attributes()

    device = torch.device(
        DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    model.train()

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

    if device.type == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(
            device
        )

    print(
        "[SMOKE] Forward + loss...",
        flush=True,
    )

    loss, loss_items = model.loss(
        batch
    )

    if not torch.is_tensor(
        loss
    ):

        raise RuntimeError(
            f"Loss inesperada: "
            f"{type(loss)}"
        )

    loss_raw_shape = tuple(
        loss.shape
    )

    if loss.numel() != 1:

        loss_scalar = loss.sum()

    else:

        loss_scalar = loss.reshape(
            ()
        )

    if loss_scalar.numel() != 1:

        raise RuntimeError(
            f"No s'ha pogut convertir "
            f"la loss a escalar. "
            f"Shape: {loss.shape}"
        )

    print(
        f"[SMOKE] Loss raw shape: "
        f"{loss_raw_shape}",
        flush=True,
    )

    print(
        f"[SMOKE] Loss escalar: "
        f"{loss_scalar.detach().cpu().item():.6f}",
        flush=True,
    )

    if torch.is_tensor(
        loss_items
    ):

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
        f"[SMOKE] Loss items: "
        f"{loss_items_print}",
        flush=True,
    )

    print(
        "[SMOKE] Backward...",
        flush=True,
    )

    loss_scalar.backward()

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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
    )

    optimizer.step()

    print(
        "\nSMOKE TEST SUPERAT!"
    )

    print(
        "DICOM -> mammo_prep -> batch -> "
        "model -> loss -> backward -> optimizer OK."
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
# TRAINING
# ============================================================

def train(
    windowing_method,
    calc_window,
    voi_func,
    resume=None,
):

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
        f"Method:      {windowing_method}",
        flush=True,
    )

    print(
        f"Calc window: {calc_window}",
        flush=True,
    )

    print(
        f"VOI function:{voi_func}",
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
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
    )

    trainer.args.epochs = EPOCHS

    print(
        "[TRAIN] Iniciant entrenament...",
        flush=True,
    )

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

    parser.set_defaults(
        calc_window=True
    )

    parser.add_argument(
        "--voi-func",
        choices=[
            "LINEAR",
            "LINEAR_EXACT",
            "SIGMOID",
        ],
        default="LINEAR",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.mode == "test":

        test_pipeline(
            windowing_method=args.method,
            calc_window=args.calc_window,
            voi_func=args.voi_func,
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