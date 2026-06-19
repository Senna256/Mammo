# %%
import pydicom 
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow
import fastparquet
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
DICOM_BASE = "/mnt/cafetera/mammo/vindr/images"
ANNOTATIONS = "/home/enric_sena/Desktop/prova_enric/vindr_dataset/vindr/vindr/finding_annotations.csv"
LOCAL_SAVE = "/home/enric_sena/Desktop/prova_enric/vindr_dataset/vindr/vindr"


# %%
# Bucle per a iterar sobre els dicom per extreure les metadades.

rows = []
root = Path(DICOM_BASE)

dicom_files = list(root.rglob("*.dicom"))  # o *.dcm

rows = []

for path in tqdm(dicom_files, desc="Llegint DICOMs"):

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True
        )

        row = {
            "filepath": str(path),
            "StudyInstanceUID": ds.get("StudyInstanceUID"),
            "SeriesInstanceUID": ds.get("SeriesInstanceUID"),
            "SOPInstanceUID": ds.get("SOPInstanceUID"),
        }

        for elem in ds:

            if elem.VR == "SQ":
                continue

            keyword = elem.keyword or str(elem.tag)
            row[keyword] = str(elem.value)

        rows.append(row)

    except Exception as e:
        print(f"\nError en {path}: {e}")

metadata_df = pd.DataFrame(rows)

metadata_df.to_parquet(f"{LOCAL_SAVE}.parquet")



