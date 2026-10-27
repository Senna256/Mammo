import cv2
import numpy as np

from mammo_prep.artifacts import detect_breast_roi, resize_and_pad, resize_keep_aspect
from mammo_prep.io import load_dicom
from mammo_prep.windowing import preprocess_window


def window_dicom(dicom_path):
    """
    Reads a DICOM and windows it to a uint8 image with bright tissue.

    MONOCHROME1 images are inverted as max - image so that tissue is bright.

    Returns
    -------
    np.ndarray uint8 (H, W) at the original DICOM resolution.
    """
    image, ds = load_dicom(dicom_path)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    return preprocess_window(image, dicom_dataset=ds, method="breast_tissue", voi_func="LINEAR",
                             exclude_background=True, output_dtype=np.uint8)


def preprocess_mammogram(dicom_path, img_size=1024):
    """
    Full preprocessing of a mammogram, in one call:

        DICOM -> windowing (breast_tissue, LINEAR) -> long side img_size
              -> breast crop (detect_breast_roi) -> img_size x img_size x 3, centred and padded

    Parameters
    ----------
    dicom_path : str or Path
    img_size : int
        Side of the output square.

    Returns
    -------
    image : np.ndarray uint8 (img_size, img_size, 3)
        The three channels are the same grey image.
    transform : dict
        The geometric steps applied. Pass it to transform_boxes() to move boxes of
        the original DICOM onto the image.
    """
    image = window_dicom(dicom_path)
    original_h, original_w = image.shape

    image, _ = resize_keep_aspect(image, img_size)
    x1, y1, x2, y2 = detect_breast_roi(image)
    crop = image[y1:y2, x1:x2]
    # Converted to 3 channels before the final resize.
    final, scale, pad_x, pad_y = resize_and_pad(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), img_size)

    transform = {
        "original_size": (original_w, original_h),
        "resized_size": (image.shape[1], image.shape[0]),
        "roi": (x1, y1, x2, y2),
        "scale": scale,
        "pad": (pad_x, pad_y),
        "img_size": img_size,
    }
    return final, transform


def transform_boxes(boxes, transform):
    """
    Boxes (x1, y1, x2, y2) of the original DICOM -> pixels of the preprocessed image.

    Boxes are clipped to the breast crop. Those left with <= 1 px (in the
    long-side-resized image) are dropped.

    Parameters
    ----------
    boxes : iterable of (x1, y1, x2, y2)
        In pixels of the original DICOM.
    transform : dict
        Returned by preprocess_mammogram().

    Returns
    -------
    np.ndarray float32 (N, 4) in pixels of the img_size x img_size image.
    """
    w, h = transform["original_size"]
    resized_w, resized_h = transform["resized_size"]
    rx1, ry1, rx2, ry2 = transform["roi"]
    pad_x, pad_y = transform["pad"]
    scale, img_size = transform["scale"], transform["img_size"]

    out = []
    for bx1, by1, bx2, by2 in boxes:
        # Through normalised coordinates: the resized size is rounded, so it is
        # not exactly original_size * scale.
        bx1, bx2 = np.clip([bx1, bx2], 0, w) / w * resized_w
        by1, by2 = np.clip([by1, by2], 0, h) / h * resized_h
        bx1, bx2 = np.clip([bx1 - rx1, bx2 - rx1], 0, rx2 - rx1)
        by1, by2 = np.clip([by1 - ry1, by2 - ry1], 0, ry2 - ry1)
        if bx2 - bx1 <= 1 or by2 - by1 <= 1:
            continue
        box = np.array([bx1 * scale + pad_x, by1 * scale + pad_y, bx2 * scale + pad_x, by2 * scale + pad_y])
        out.append(np.clip(box, 0, img_size))
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def inverse_transform_boxes(boxes, transform):
    """
    Boxes (x1, y1, x2, y2) of the preprocessed image -> pixels of the original DICOM.

    Use it to draw or evaluate model predictions on the original mammogram.
    """
    w, h = transform["original_size"]
    resized_w, resized_h = transform["resized_size"]
    rx1, ry1, _, _ = transform["roi"]
    pad_x, pad_y = transform["pad"]
    scale = transform["scale"]

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    xs = ((boxes[:, [0, 2]] - pad_x) / scale + rx1) / resized_w * w
    ys = ((boxes[:, [1, 3]] - pad_y) / scale + ry1) / resized_h * h
    return np.stack([xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]], axis=1).astype(np.float32)
