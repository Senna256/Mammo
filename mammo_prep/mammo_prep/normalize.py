import cv2
import numpy as np

def normalize_to_uint8(img):
    """
    Normalize image to uint8 [0,255] cropping percentiles 1-99 to avoid oultiers effect on image.
    """

    img = img.astype(np.float32)
    p1, p99 = np.percentile(img,(1,99))
    img = np.clip(img, p1,p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

def apply_clahe(img, clip_limit=2.0, title_grid_size=(8,8)):
    """
    Apply CLAHE to improve local contrast.
    Deliers an uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
    return clahe.apply(img)

def get_breast_mask(img, min_area=2000):
    """
    Generates a binary mask of the breast using Otsu thresholding and
    connected component analysis.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image. It may contain the original DICOM intensities.
    min_area : int, optional
        Minimum connected component area to keep.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) with values 0 and 255.
    """

    # Otsu needs an 8-bit representation, but this temporary normalization is
    # used only to calculate the mask. It does not modify the returned data.
    img_for_mask = normalize_to_uint8(img)

    # Otsu threshold
    mask = cv2.threshold(
        img_for_mask,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Connected components
    total_labels, label_ids, stats, _ = cv2.connectedComponentsWithStats(
        mask, 4, cv2.CV_32S
    )

    output = np.zeros_like(mask)

    # Text and acquisition markers can also pass Otsu. The breast is expected
    # to be the largest foreground component, so retain only that component.
    if total_labels > 1:
        foreground_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(foreground_areas))

        if stats[largest_label, cv2.CC_STAT_AREA] >= min_area:
            output[label_ids == largest_label] = 255

    return output
