"""
mammo_prep
----------
Mammography image preprocessing utilities.

Full preprocessing in one call:
    image, transform = preprocess_mammogram(dicom_path)
"""
from mammo_prep.io import load_dicom
from mammo_prep.normalize import normalize_to_uint8, apply_clahe, get_breast_mask
from mammo_prep.windowing import preprocess_window
from mammo_prep.artifacts import (flip_to_standard, crop_breast, detect_breast_roi, resize_long_side,
                                  pad_to_square, resize_keep_aspect, resize_and_pad)
from mammo_prep.pipeline import window_dicom, preprocess_mammogram, transform_boxes, inverse_transform_boxes
from mammo_prep.viz import show, plot_comparison
