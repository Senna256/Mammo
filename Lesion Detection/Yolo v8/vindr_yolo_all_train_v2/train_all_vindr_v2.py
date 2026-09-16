import argparse
import os
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
# CONFIG
# ============================================================

DATASET = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/"
    "Yolo v8/vindr_yolo"
)

CSV_PATH = Path(
    "/home/enric_sena/Desktop/prova_enric/vindr_dataset/"
    "finding_annotations.csv"
)

MODEL = "yolov8n.pt"

IMG_SIZE = 1024
BATCH = 4
WORKERS = 8
EPOCHS = 100
DEVICE = 0
CACHE = False

DEFAULT_METHOD = "breast_tissue"
DEFAULT_VOI_FUNC = "LINEAR"
DEFAULT_CALC_WINDOW = True


# ============================================================
# DICOM PREPROCESSING
# ============================================================

def read_and_process_dicom(
    path,
    method="breast_tissue",
    calc_window=True,
    voi_func="LINEAR",
):
    ds = pydicom.dcmread(path)

    image = ds.pixel_array

    # --------------------------------------------------------
    # MONOCHROME1
    # --------------------------------------------------------

    if (
        getattr(ds, "PhotometricInterpretation", "")
        == "MONOCHROME1"
    ):
        image = (
            image.max()
            + image.min()
            - image
        )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # WINDOWING
    # --------------------------------------------------------

    if calc_window:

        im = preprocess_window(
            image,
            dicom_dataset=ds,
            method=method,
            voi_func=voi_func,
        )

    else:

        params = get_dicom_voi_lut_params(ds)

        im = apply_windowing(
            image,
            window_width=params["window_width"],
            window_center=params["window_center"],
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
        ).astype(np.uint8)

    # --------------------------------------------------------
    # GRAYSCALE -> BGR
    # --------------------------------------------------------

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
        raise ValueError(
            f"Unexpected image shape after preprocessing: "
            f"{im.shape}"
        )

    return im


# ============================================================
# DICOM YOLO DATASET
# ============================================================

class DICOMYOLODataset(YOLODataset):

    def __init__(
        self,
        *args,
        windowing_method=DEFAULT_METHOD,
        calc_window=DEFAULT_CALC_WINDOW,
        voi_func=DEFAULT_VOI_FUNC,
        **kwargs,
    ):

        self.windowing_method = windowing_method
        self.calc_window = calc_window
        self.voi_func = voi_func

        super().__init__(
            *args,
            **kwargs,
        )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    def get_labels(self):

        labels = []

        for im_file in self.im_files:

            im_path = Path(im_file)

            label_path = (
                im_path.parent.parent
                / "labels"
                / im_path.parent.name
                / f"{im_path.stem}.txt"
            )

            if not label_path.exists():

                # Empty label
                cls = np.zeros(
                    (0, 1),
                    dtype=np.float32,
                )

                bboxes = np.zeros(
                    (0, 4),
                    dtype=np.float32,
                )

            else:

                rows = []

                with open(
                    label_path,
                    "r",
                ) as f:

                    for line in f:

                        line = line.strip()

                        if not line:
                            continue

                        values = list(
                            map(
                                float,
                                line.split(),
                            )
                        )

                        if len(values) != 5:
                            continue

                        rows.append(values)

                if rows:

                    arr = np.asarray(
                        rows,
                        dtype=np.float32,
                    )

                    cls = arr[:, 0:1]

                    bboxes = arr[:, 1:5]

                else:

                    cls = np.zeros(
                        (0, 1),
                        dtype=np.float32,
                    )

                    bboxes = np.zeros(
                        (0, 4),
                        dtype=np.float32,
                    )

            # ------------------------------------------------
            # IMAGE SHAPE FROM CSV
            # ------------------------------------------------

            image_id = im_path.stem

            rows_csv = self.data[
                self.data["image_id"].astype(str)
                == image_id
            ]

            if len(rows_csv) > 0:

                h = int(
                    rows_csv.iloc[0]["height"]
                )

                w = int(
                    rows_csv.iloc[0]["width"]
                )

            else:

                ds = pydicom.dcmread(
                    im_file,
                    stop_before_pixels=True,
                )

                h = int(ds.Rows)
                w = int(ds.Columns)

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
            f"({self.prefix.strip()})"
        )

        return labels

    # --------------------------------------------------------
    # IMAGE LOADING
    # --------------------------------------------------------

    def _load_one(
        self,
        index,
    ):

        im_file = self.im_files[index]

        im = read_and_process_dicom(
            im_file,
            method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
        )

        h0, w0 = im.shape[:2]

        # ----------------------------------------------------
        # RESIZE PRESERVING ASPECT RATIO
        # ----------------------------------------------------

        scale = min(
            self.imgsz / h0,
            self.imgsz / w0,
        )

        if scale != 1:

            new_w = int(round(w0 * scale))
            new_h = int(round(h0 * scale))

            im = cv2.resize(
                im,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return (
            im,
            (h0, w0),
            im.shape[:2],
        )

    def load_image(
        self,
        index,
        rect_mode=False,
    ):

        return self._load_one(index)


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
            img_path,
            data=self.data,
            task=self.args.task,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=False,
            single_cls=self.args.single_cls,
            stride=self.stride,
            pad=0.5,
            prefix=f"{mode}: ",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
            windowing_method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
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

        gs = max(
            int(self.model.stride.max())
            if self.model is not None
            else 32,
            32,
        )

        return DICOMYOLODataset(
            img_path,
            data=self.data,
            task=self.args.task,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            classes=self.args.classes,
            fraction=self.args.fraction,
            windowing_method=self.windowing_method,
            calc_window=self.calc_window,
            voi_func=self.voi_func,
        )

    def get_validator(self):

        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
        )

        validator = DICOMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
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
# TRAINER CREATION
# ============================================================

def create_trainer(
    windowing_method,
    calc_window,
    voi_func,
    resume=None,
):

    if resume is not None:

        model = str(resume)
        resume_value = str(resume)

    else:

        model = MODEL
        resume_value = False

    overrides = {

        "model": model,

        "data": str(DATA_YAML),

        "epochs": EPOCHS,

        "imgsz": IMG_SIZE,

        "batch": BATCH,

        "workers": WORKERS,

        "device": DEVICE,

        "cache": CACHE,

        "project": str(
            Path(
                "/home/enric_sena/Desktop/Mammo/runs/detect"
            )
        ),

        "name": "vindr_v2",

        "exist_ok": True,

        "pretrained": True,

        "verbose": True,

        "resume": resume_value,

    }

    trainer = DICOMTrainer(
        overrides=overrides,
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
# TEST PIPELINE
# ============================================================

def test_pipeline(
    windowing_method,
    calc_window,
    voi_func,
):

    print()
    print("=" * 70)
    print("V2 TEST PIPELINE")
    print("=" * 70)

    trainer = create_trainer(
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
    )

    trainer.setup_model()

    dataset = trainer.build_dataset(
        DATASET / "images" / "train",
        mode="train",
        batch=BATCH,
    )

    print(
        f"[TEST] Dataset creat: "
        f"{len(dataset)} imatges"
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

    # --------------------------------------------------------
    # LOAD ONE BATCH
    # --------------------------------------------------------

    batch = next(iter(loader))

    print(
        f"[TEST] batch['img'].shape = "
        f"{batch['img'].shape}"
    )

    print(
        f"[TEST] batch['cls'].shape = "
        f"{batch['cls'].shape}"
    )

    print(
        f"[TEST] batch['bboxes'].shape = "
        f"{batch['bboxes'].shape}"
    )

    print(
        f"[TEST] batch['batch_idx'].shape = "
        f"{batch['batch_idx'].shape}"
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    imgs = batch["img"].to(
        trainer.device,
        non_blocking=True,
    )

    trainer.model.train()

    preds = trainer.model(imgs)

    print(
        "[TEST] Forward GPU OK"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss, loss_items = trainer.model.loss(
        batch
    )

    if loss.numel() != 1:

        loss_scalar = loss.sum()

    else:

        loss_scalar = loss.reshape(())

    print(
        f"[TEST] Loss = "
        f"{loss_scalar.item():.6f}"
    )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    loss_scalar.backward()

    print(
        "[TEST] Backward OK"
    )

    print()
    print("=" * 70)
    print("TEST SUPERAT")
    print("=" * 70)
    print()
    print(
        f"method      = {windowing_method}"
    )
    print(
        f"calc_window = {calc_window}"
    )
    print(
        f"voi_func    = {voi_func}"
    )
    print()
    print(
        "No s'han creat PNG/JPG."
    )
    print(
        "No s'han creat fitxers .npy."
    )
    print()


# ============================================================
# TRAIN
# ============================================================

def train(
    windowing_method,
    calc_window,
    voi_func,
    resume=None,
):

    print()
    print("=" * 70)
    print("VINDR YOLO V2 TRAINING")
    print("=" * 70)

    print(
        f"method      = {windowing_method}"
    )

    print(
        f"calc_window = {calc_window}"
    )

    print(
        f"voi_func    = {voi_func}"
    )

    print(
        f"resume      = {resume}"
    )

    print("=" * 70)
    print()

    trainer = create_trainer(
        windowing_method=windowing_method,
        calc_window=calc_window,
        voi_func=voi_func,
        resume=resume,
    )

    trainer.train()


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "test",
            "smoke",
            "train",
        ],
        default="test",
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
        default=DEFAULT_METHOD,
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
        calc_window=DEFAULT_CALC_WINDOW
    )

    parser.add_argument(
        "--voi-func",
        choices=[
            "LINEAR",
            "LINEAR_EXACT",
            "SIGMOID",
        ],
        default=DEFAULT_VOI_FUNC,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.mode == "test":

        test_pipeline(
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

    elif args.mode == "smoke":

        test_pipeline(
            windowing_method=args.method,
            calc_window=args.calc_window,
            voi_func=args.voi_func,
        )


if __name__ == "__main__":
    main()