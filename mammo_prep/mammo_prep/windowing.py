import numpy as np


WINDOW_METHODS = (
    "dicom",
    "percentile_1_99",
    "percentile_2_98",
    "percentile_5_95",
    "breast_tissue",
    "statistical",
    "statistical_wide",
    "full_range",
    "histogram_peak",
)

VOI_FUNCTIONS = (
    "LINEAR",
    "LINEAR_EXACT",
    "SIGMOID",
)


def _first_numeric(value):
    if value is None:
        return None

    try:
        if isinstance(value, (list, tuple)):
            return float(value[0])

        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return float(next(iter(value)))

        return float(value)

    except (TypeError, ValueError, StopIteration):
        return None


def calculate_windowing(
    image,
    method="percentile_2_98",
    exclude_background=True,
):
    if method == "dicom":
        raise ValueError(
            "'dicom' requires a DICOM dataset. "
            "Use get_windowing_for_dicom() or preprocess_window()."
        )

    if method not in WINDOW_METHODS:
        raise ValueError(
            f"Unknown windowing method '{method}'. "
            f"Choose from: {', '.join(WINDOW_METHODS[1:])}"
        )

    image = np.asarray(image, dtype=np.float32)

    if exclude_background:
        pixels = image[image > 0]
    else:
        pixels = image.ravel()

    if pixels.size == 0:
        raise ValueError("No valid pixels available for window calculation.")

    if method == "percentile_1_99":
        low, high = np.percentile(pixels, [1, 99])

    elif method == "percentile_2_98":
        low, high = np.percentile(pixels, [2, 98])

    elif method == "percentile_5_95":
        low, high = np.percentile(pixels, [5, 95])

    elif method == "breast_tissue":
        low, high = np.percentile(pixels, [25, 95])
        width = (high - low) * 1.5
        center = (high + low) / 2
        return float(center), float(width)

    elif method == "statistical":
        mean = np.mean(pixels)
        std = np.std(pixels)
        low = mean - 2 * std
        high = mean + 2 * std

    elif method == "statistical_wide":
        mean = np.mean(pixels)
        std = np.std(pixels)
        low = mean - 3 * std
        high = mean + 3 * std

    elif method == "full_range":
        low = np.min(pixels)
        high = np.max(pixels)

    elif method == "histogram_peak":
        hist, bin_edges = np.histogram(pixels, bins=256)

        peak_idx = np.argmax(hist)
        peak = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2

        half_max = hist[peak_idx] / 2
        indices = np.where(hist >= half_max)[0]

        if len(indices) > 1:
            fwhm = bin_edges[indices[-1] + 1] - bin_edges[indices[0]]
            width = fwhm * 1.75
        else:
            width = 4 * np.std(pixels)

        if width < 100:
            width = 4 * np.std(pixels)

        return float(peak), float(width)

    else:
        raise ValueError(f"Unsupported method: {method}")

    width = float(high - low)
    center = float((high + low) / 2)

    if width <= 0:
        width = 1.0

    return center, width


def calculate_all_methods(image, exclude_background=True):
    return {
        method: calculate_windowing(
            image,
            method=method,
            exclude_background=exclude_background,
        )
        for method in WINDOW_METHODS
        if method != "dicom"
    }


def get_dicom_voi_lut_params(dicom_dataset):
    window_center = _first_numeric(
        getattr(dicom_dataset, "WindowCenter", None)
    )

    window_width = _first_numeric(
        getattr(dicom_dataset, "WindowWidth", None)
    )

    if (
        window_center is None
        or window_width is None
        or window_width <= 0
    ):
        window_center, window_width = calculate_windowing(
            dicom_dataset.pixel_array,
            method="breast_tissue",
            exclude_background=True,
        )
        source = "calculated"
    else:
        source = "dicom"

    voi_func = getattr(
        dicom_dataset,
        "VOILUTFunction",
        "LINEAR",
    )

    if isinstance(voi_func, (list, tuple)):
        voi_func = voi_func[0]

    return {
        "window_center": float(window_center),
        "window_width": float(window_width),
        "rescale_intercept": float(
            getattr(dicom_dataset, "RescaleIntercept", 0)
        ),
        "rescale_slope": float(
            getattr(dicom_dataset, "RescaleSlope", 1)
        ),
        "voi_lut_function": str(voi_func),
        "source": source,
    }


def should_invert_monochrome1(dicom_dataset):
    return (
        getattr(
            dicom_dataset,
            "PhotometricInterpretation",
            "",
        )
        == "MONOCHROME1"
    )


def normalize_photometric(image, dicom_dataset):
    image = np.asarray(image)

    if should_invert_monochrome1(dicom_dataset):
        return (image.max() + image.min()) - image

    return image


def get_windowing_for_dicom(
    image,
    dicom_dataset,
    method="dicom",
    exclude_background=True,
):
    if method not in WINDOW_METHODS:
        raise ValueError(
            f"Unknown windowing method '{method}'. "
            f"Choose from: {', '.join(WINDOW_METHODS)}"
        )

    if method == "dicom":
        return get_dicom_voi_lut_params(dicom_dataset)

    window_center, window_width = calculate_windowing(
        image,
        method=method,
        exclude_background=exclude_background,
    )

    return {
        "window_center": window_center,
        "window_width": window_width,
        "rescale_intercept": float(
            getattr(dicom_dataset, "RescaleIntercept", 0)
        ),
        "rescale_slope": float(
            getattr(dicom_dataset, "RescaleSlope", 1)
        ),
        "voi_lut_function": str(
            getattr(dicom_dataset, "VOILUTFunction", "LINEAR")
        ),
        "source": "calculated",
    }


def _apply_windowing_np(
    image,
    window_width,
    window_center,
    voi_func="LINEAR",
    y_min=0,
    y_max=255,
):
    if window_width <= 0:
        raise ValueError("window_width must be > 0")

    image = np.asarray(image, dtype=np.float32)

    voi_func = voi_func.upper()
    y_range = y_max - y_min

    if voi_func in ("LINEAR", "LINEAR_EXACT"):

        if voi_func == "LINEAR":
            if window_width < 1:
                raise ValueError(
                    "For LINEAR VOI, window_width must be >= 1."
                )

            window_center -= 0.5
            window_width -= 1

        scale = y_range / window_width
        bias = (
            -window_center / window_width + 0.5
        ) * y_range + y_min

        result = image * scale + bias
        result = np.clip(result, y_min, y_max)

    elif voi_func == "SIGMOID":

        result = (
            y_range
            / (
                1
                + np.exp(
                    -4
                    * (image - window_center)
                    / window_width
                )
            )
            + y_min
        )

    else:
        raise ValueError(
            f"Unsupported VOI LUT function '{voi_func}'. "
            f"Use one of {VOI_FUNCTIONS}."
        )

    return result


def apply_windowing(
    image,
    window_width,
    window_center,
    voi_func="LINEAR",
    y_min=0,
    y_max=255,
):
    return _apply_windowing_np(
        image,
        window_width=window_width,
        window_center=window_center,
        voi_func=voi_func,
        y_min=y_min,
        y_max=y_max,
    )


def preprocess_window(
    image,
    dicom_dataset=None,
    method="dicom",
    voi_func=None,
    exclude_background=True,
    output_dtype=np.uint8,
):
    if method == "dicom" and dicom_dataset is None:
        raise ValueError(
            "dicom_dataset is required when method='dicom'."
        )

    if dicom_dataset is not None:
        params = get_windowing_for_dicom(
            image,
            dicom_dataset,
            method=method,
            exclude_background=exclude_background,
        )
    else:
        window_center, window_width = calculate_windowing(
            image,
            method=method,
            exclude_background=exclude_background,
        )

        params = {
            "window_center": window_center,
            "window_width": window_width,
            "voi_lut_function": "LINEAR",
        }

    if voi_func is None:
        voi_func = params.get(
            "voi_lut_function",
            "LINEAR",
        )

    result = apply_windowing(
        image,
        window_width=params["window_width"],
        window_center=params["window_center"],
        voi_func=voi_func,
        y_min=0,
        y_max=255,
    )

    if output_dtype == np.uint8:
        return np.rint(
            np.clip(result, 0, 255)
        ).astype(np.uint8)

    return result.astype(output_dtype)