# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import List, Tuple, Union


def load_image(img_path: str, channel_first=False) -> np.ndarray:
    """
    Loads the RGB images for a given scene as np.ndarray.

    Args:
        img_path (str): the image path
        channel_first (bool): the flag to control whether the channel first or not. If true, then the output will be
            [3, H, W]; otherwise it will be [H, W, 3]. The default is False.

    Returns:
        img (np.ndarray): see channel first.

    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if channel_first:
        img = np.transpose(img, (2, 0, 1))

    return img


def load_mask(parse_path: str,
              hw_size: Union[List[int], Tuple[int], None] = None,
              hard_threshold: Union[float, None] = None):

    """

    Args:
        parse_path:
        hw_size:
        hard_threshold:

    Returns:
        mask (np.ndarray): [H, W] with [0, 1] color space.

    """

    mask = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)

    if hw_size is not None:
        mask = cv2.resize(mask, (hw_size[1], hw_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255

    if hard_threshold is not None:
        mask = (mask >= hard_threshold).astype(np.float32, copy=False)

    return mask


