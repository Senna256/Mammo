import os
import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import yaml

from ultralytics import YOLO
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset


# ============================================================
# PATHS
# ============================================================

DICOM_BASE = Path("/mnt/cafetera/mammo/vindr/images")
ANNOTATIONS = Path(
    "/home/enric_sena/Desktop/prova_enric/vindr_dataset/finding_annotations.csv"
)

OUT = Path(
    "/home/enric_sena/Desktop/Mammo/Lesion Detection/Yolo v8/vindr_yolo"
)

# NO TOQUEM DICOM_BASE
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(ANNOTATIONS)

print("Annotations:", len(df))
print("Splits:", df["split"].value_counts().to_dict())


# ============================================================
# INDEX DELS DICOM
# Una sola passada per la carpeta original
# ============================================================

print("\nIndexant DICOM...")

dicom_index = {}

for study_dir in DICOM_BASE.iterdir():
    if not study_dir.is_dir():
        continue

    for p in study_dir.glob("*.dicom"):
        dicom_index[p.stem] = p

print("DICOM trobats:", len(dicom_index))


# ============================================================
# IMAGE IDS
# ============================================================

image_df = (
    df[["image_id", "study_id", "split"]]
    .drop_duplicates("image_id")
)

print("Imatges al CSV:", len(image_df))

missing = [
    x for x in image_df["image_id"]
    if x not in dicom_index
]

print("DICOM que falten:", len(missing))

if missing:
    print(missing[:10])
    raise RuntimeError("Falten DICOM. No continuem.")


# ============================================================
# SPLIT
#
# VinDr només proporciona training/test.
# Separem un 10% dels STUDIES de training per validation.
# ============================================================

train_studies = (
    image_df[image_df["split"] == "training"]["study_id"]
    .drop_duplicates()
)

rng = np.random.default_rng(42)

val_studies = set(
    rng.choice(
        train_studies.to_numpy(),
        size=int(len(train_studies) * 0.10),
        replace=False
    )
)

def get_split(row):
    if row["split"] == "test":
        return "test"
    elif row["study_id"] in val_studies:
        return "val"
    else:
        return "train"

image_df["yolo_split"] = image_df.apply(get_split, axis=1)

print("\nYOLO split:")
print(image_df["yolo_split"].value_counts())


# ============================================================
# CREAR DIRECTORIS
# ============================================================

for split in ["train", "val", "test"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# CREAR SYMLINKS + LABELS
#
# NO ES COPIA.
# El fitxer creat a OUT/images/... només apunta al DICOM original.
# ============================================================

for _, row in image_df.iterrows():

    image_id = row["image_id"]
    study_id = row["study_id"]
    split = row["yolo_split"]

    dicom_path = dicom_index[image_id]

    # Symlink amb extensió .jpg perquè YOLO el reconegui
    image_link = OUT / "images" / split / f"{image_id}.jpg"

    if not image_link.exists():
        image_link.symlink_to(dicom_path)

    # Labels
    label_path = OUT / "labels" / split / f"{image_id}.txt"

    rows = df[df["image_id"] == image_id]

    with open(label_path, "w") as f:

        for _, ann in rows.iterrows():

            if pd.isna(ann["xmin"]):
                continue

            # Una sola classe: lesion = 0
            w = float(ann["width"])
            h = float(ann["height"])

            xc = ((ann["xmin"] + ann["xmax"]) / 2) / w
            yc = ((ann["ymin"] + ann["ymax"]) / 2) / h

            bw = (ann["xmax"] - ann["xmin"]) / w
            bh = (ann["ymax"] - ann["ymin"]) / h

            f.write(
                f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n"
            )


# ============================================================
# YAML
# ============================================================

data = {
    "path": str(OUT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["lesion"],
}

with open(OUT / "data.yaml", "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)


# ============================================================
# COMPROVACIONS
# ============================================================

for split in ["train", "val", "test"]:

    imgs = list((OUT / "images" / split).glob("*.jpg"))
    labels = list((OUT / "labels" / split).glob("*.txt"))

    print(
        f"{split}: "
        f"{len(imgs)} images | "
        f"{len(labels)} labels"
    )

print("\nDataset creat:", OUT)
print("Els JPG són SYMLINKS, no còpies.")