import cv2
import numpy as np

def flip_to_standard(img, ds):
    """
    Sets all mammography flipped to the left side.
    If ImageLaterality is R, flip horizonatally.
    """
    if hasattr(ds, "ImageLaterality") and ds.ImageLaterality == "R":
        img = np.fliplr(img)
    return img

def crop_breast(img):
    """
    Crops image to the bounding box of mamma tissue using OTSU thresholding. Return image cropped and mask associated.
    """
    _, mask = cv2.threshold( img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask>0
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return img,(mask.astype(np.uint8) * 255)
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1,x0:x1]
    cropped_mask = mask[y0:y1,x0:x1].astype(np.uint8) * 255

    return cropped, cropped_mask


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