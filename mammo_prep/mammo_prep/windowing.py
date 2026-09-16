"""
Windowing utilities for mammography DICOM images.
"""

import numpy as np


def apply_window(
    image: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Apply intensity windowing to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in its original intensity range.
    window_center : float
        Center of the intensity window.
    window_width : float
        Width of the intensity window.

    Returns
    -------
    np.ndarray
        Windowed image normalized to [0, 1].
    """

    if window_width <= 0:
        raise ValueError("window_width must be greater than 0.")

    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    image = np.clip(image, lower, upper)

    image = (image - lower) / (upper - lower)

    return image.astype(np.float32)


def apply_window_uint8(
    image: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Apply intensity windowing and return an 8-bit image.
    """

    image = apply_window(
        image,
        window_center=window_center,
        window_width=window_width,
    )

    return np.round(image * 255).astype(np.uint8)


def apply_dicom_window(
    image: np.ndarray,
    ds,
) -> np.ndarray:
    """
    Apply the Window Center and Window Width stored in a DICOM dataset.

    Parameters
    ----------
    image : np.ndarray
        DICOM pixel array.
    ds : pydicom.dataset.FileDataset
        DICOM dataset containing WindowCenter and WindowWidth.

    Returns
    -------
    np.ndarray
        Windowed image normalized to [0, 1].
    """

    if not hasattr(ds, "WindowCenter"):
        raise ValueError("DICOM does not contain WindowCenter.")

    if not hasattr(ds, "WindowWidth"):
        raise ValueError("DICOM does not contain WindowWidth.")

    window_center = ds.WindowCenter
    window_width = ds.WindowWidth

    # DICOM values can be MultiValue
    if isinstance(window_center, (list, tuple)):
        window_center = float(window_center[0])
    else:
        window_center = float(window_center)

    if isinstance(window_width, (list, tuple)):
        window_width = float(window_width[0])
    else:
        window_width = float(window_width)

    return apply_window(
        image,
        window_center=window_center,
        window_width=window_width,
    )