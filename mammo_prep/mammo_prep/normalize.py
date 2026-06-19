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
