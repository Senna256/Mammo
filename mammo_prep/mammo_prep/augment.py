import random

import numpy as np


def augment_detection(image, labels, rng=None, p_flip=0.5, p_intensity=0.5,
                      contrast=(0.9, 1.1), brightness=(-10.0, 10.0)):
    """
    Random training augmentation for a preprocessed image and its YOLO boxes.

    Two independent transforms:
    - horizontal flip with probability `p_flip` (box centre cx -> 1 - cx);
    - with probability `p_intensity`, a linear intensity change
      image * alpha + beta, alpha drawn from `contrast` and beta from
      `brightness` (grey levels). Pixels that are exactly 0 (the black
      padding of resize_and_pad) stay 0, so the padding looks the same as
      at test time. Boxes are unchanged.

    Parameters
    ----------
    image : uint8 array, H x W or H x W x C (e.g. the output of resize_and_pad).
    labels : float array (N, 5), rows [cls, cx, cy, w, h] normalised to [0, 1]
        over `image` (the YOLO format the training scripts use). N can be 0.
    rng : random number generator with .random() and .uniform(a, b), such as
        random.Random or numpy.random.Generator. Default: the `random` module,
        which PyTorch seeds differently in each DataLoader worker.

    Returns
    -------
    (augmented image uint8, augmented labels float32): new arrays, the inputs
    are not modified.

    Apply it to TRAIN only; validation and test must stay deterministic.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 5).copy()
    if labels.size and (labels[:, 1:].min() < 0 or labels[:, 1:].max() > 1):
        raise ValueError("labels must be [cls, cx, cy, w, h] normalised to [0, 1]")
    rng = random if rng is None else rng

    if rng.random() < p_flip:
        image = image[:, ::-1]
        labels[:, 1] = 1.0 - labels[:, 1]

    if rng.random() < p_intensity:
        alpha = rng.uniform(*contrast)
        beta = rng.uniform(*brightness)
        padding = image == 0
        out = np.clip(image.astype(np.float32) * alpha + beta, 0, 255)
        out[padding] = 0
        image = out.astype(np.uint8)

    return np.ascontiguousarray(image), labels
