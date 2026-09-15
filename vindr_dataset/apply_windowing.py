"""
Apply windowing to DICOM images using VOI LUT parameters (window center and window width).
This module is particularly useful for datasets where DICOM windowing metadata
===========================================================================================
Source: https://github.com/dangnh0611/kaggle_rsna_breast_cancer

Pydicom has its own implementation of windowing: pydicom.pixels.apply_windowing

Check also:   Module pydicom.pixels
Functions: 
- apply_color_lut(arr[, ds, palette]) --> Apply a color palette lookup table to arr.
- apply_icc_profile(arr[, ds, transform, ...]) --> Apply an ICC Profile to arr, either from the dataset ds or an existing Pillow color transformation object transform.
- apply_modality_lut(arr, ds) --> Apply a modality lookup table or rescale operation to arr.
- apply_presentation_lut(arr, ds) --> Apply a Presentation LUT to arr and return the P-values.
- apply_rescale(arr, ds) --> Apply a modality lookup table or rescale operation to arr.
- apply_voi_lut(arr, ds[, index, prefer_lut]) --> Apply a VOI lookup table or windowing operation to arr.
- apply_voi(arr, ds[, index]) --> Apply a VOI lookup table to arr.
- apply_windowing(arr, ds[, index]) --> Apply a windowing operation to arr.
"""

import numpy as np
#import torch


# slow
# from pydicom's source
def _apply_windowing_np_v1(arr,
                           window_width=None,
                           window_center=None,
                           voi_func='LINEAR',
                           y_min=0,
                           y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    
    # float64 needed (default) or just float32 ?
    # arr = arr.astype(np.float64)
    arr = arr.astype(np.float32)

    #
    if voi_func in ['LINEAR', 'LINEAR_EXACT']:
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1
        below = arr <= (window_center - window_width / 2)
        above = arr > (window_center + window_width / 2)
        between = np.logical_and(~below, ~above)
        #
        arr[below] = y_min
        arr[above] = y_max
        if between.any():
            arr[between] = ((
                (arr[between] - window_center) / window_width + 0.5) * y_range
                            + y_min)
    elif voi_func == 'SIGMOID':
        arr = y_range / (1 +
                         np.exp(-4 *
                                (arr - window_center) / window_width)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def _apply_windowing_np_v2(arr,
                           window_width=None,
                           window_center=None,
                           voi_func='LINEAR',
                           y_min=0,
                           y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.astype(np.float64)
    arr = arr.astype(np.float32)
    #
    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1
        #
        # simple trick to improve speed
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = np.clip(arr, y_min, y_max)
        #
    elif voi_func == 'SIGMOID':
        # simple trick to improve speed
        s = -4 / window_width
        arr = y_range / (1 + np.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def _apply_windowing_torch(arr,
                           window_width=None,
                           window_center=None,
                           voi_func='LINEAR',
                           y_min=0,
                           y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.double()
    arr = arr.float()

    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1

        # simple trick to improve speed
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = torch.clamp(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        # simple trick to improve speed
        s = -4 / window_width
        arr = y_range / (1 + torch.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def apply_windowing(arr,
                    window_width=None,
                    window_center=None,
                    voi_func='LINEAR',
                    y_min=0,
                    y_max=255,
                    backend='np_v2'):
    
    if backend == 'torch':
        if isinstance(arr, torch.Tensor):
            pass
        elif isinstance(arr, np.ndarray):
            if arr.dtype == np.uint16:
                arr = torch.from_numpy(arr, torch.int16)
            else:
                arr = torch.from_numpy(arr)

    if backend == 'np_v1':
        windowing_func = _apply_windowing_np_v1
    elif backend == 'np_v2':
        windowing_func = _apply_windowing_np_v2
    elif backend == 'torch':
        windowing_func = _apply_windowing_torch
    else:
        raise ValueError(
            f'Invalid backend {backend}, must be one of ["np_v1", "np_v2", "torch"]'
        )

    arr_windowed = windowing_func(arr,
                         window_width=window_width,
                         window_center=window_center,
                         voi_func=voi_func,
                         y_min=y_min,
                         y_max=y_max)
    
    ## To fit the initial image range, we can normalize the windowed image to the original range
    arr_windowed = arr.max() * (arr_windowed- arr_windowed.min()) / (arr_windowed.max() - arr_windowed.min())
    
    return arr_windowed.astype( arr.dtype if isinstance(arr, np.ndarray) else arr_windowed.dtype)
