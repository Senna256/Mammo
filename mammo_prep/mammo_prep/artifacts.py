import cv2
import numpy as np
from mammo_prep.normalize import get_breast_mask

def flip_to_standard(img, ds):
    """
    Sets all mammography flipped to the left side.
    If ImageLaterality is R, flip horizonatally.
    """
    if hasattr(ds, "ImageLaterality") and ds.ImageLaterality == "R":
        img = np.fliplr(img)
    return img

def crop_breast(img, mask=None):
    """
    Crops the image to the bounding box defined by a breast mask.

    If no mask is provided, the breast mask is calculated using
    Otsu thresholding + connected components.
    Not used by preprocess_mammogram(), which crops with detect_breast_roi().

    Returns
    -------
    (cropped image, cropped mask uint8 0/255)
    """

    if mask is None:
        mask = get_breast_mask(img)

    if mask.shape != img.shape:
        raise ValueError("img and mask must have the same shape")

    mask = mask > 0

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return img, mask.astype(np.uint8) * 255

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1 + 1, x0:x1 + 1]
    cropped_mask = mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255

    return cropped, cropped_mask


def detect_breast_roi(img):
    """
    Bounding box (x1, y1, x2, y2) of the breast: the largest region above a fixed threshold.

    Crop used by preprocess_mammogram(). Expects the windowed uint8 image
    resized to 1024 px on its long side: the threshold (5) and the 31 px
    kernel are tuned for that input.

    Falls back to the whole image when no reasonable region is found, mainly
    when the background is grey instead of black.
    x2 / y2 are exclusive: crop with img[y1:y2, x1:x2].
    """
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("detect_breast_roi expects a 2D uint8 (windowed) image")

    h, w = img.shape
    _, mask = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    # Closing fills the gaps in fatty tissue; opening removes labels and small markers.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bw < 0.05 * w or bh < 0.05 * h:
        return 0, 0, w, h
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def resize_keep_aspect(img, max_size):
    """
    Resizes keeping the aspect ratio so that the long side is max_size.

    Uses INTER_LINEAR and rounds the new sizes.

    Returns
    -------
    (resized image, scale)
    """
    h, w = img.shape[:2]
    scale = min(max_size / float(w), max_size / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def resize_and_pad(img, img_size):
    """
    Fits a 3-channel image inside an img_size x img_size square, centred on a black canvas.

    The output always has the fixed size img_size x img_size and 3 channels.

    Returns
    -------
    (image uint8 img_size x img_size x 3, scale, pad_x, pad_y)
        A point (x, y) of the input goes to (x * scale + pad_x, y * scale + pad_y).
    """
    h, w = img.shape[:2]
    scale = min(img_size / float(w), img_size / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    pad_x, pad_y = (img_size - new_w) // 2, (img_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def resize_long_side(img, target_size=1536, interpolation=cv2.INTER_AREA):
    """
    Resizes image keeping original aspect ratio, adjust the longest side to target size.
    """

    h, w = img.shape
    scale = target_size/max(h,w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def pad_to_square(img):
    """
    Adds zero padding to square the image distributing padding symetrically.
    """

    h, w = img.shape
    size = max(h,w)
    pad_h = size - h
    pad_w =size - w

    top = pad_h// 2
    bottom =pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
