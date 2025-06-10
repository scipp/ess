# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tools for image analysis.
"""

import numpy as np
import scipp as sc
from PIL import Image, ImageFilter


def sharpness(data: sc.Variable | sc.DataArray, max_size: int = 256**2) -> float:
    """
    Calculate the sharpness of an image.

    Sharpness is defined as the sum of the edges in the image, computed
    by applying a median filter to remove noise and then finding edges.
    The image is resized to a maximum size to avoid excessive computation.
    The sharpness is normalized by the total number of pixels in the image.

    Parameters
    ----------
    data:
        The image data to analyze, expected to be a 2D scipp data array
        (Scitiff is also supported).
    max_size:
        The maximum size of the image in pixels. If the image exceeds this
        size, it will be resized to fit within this limit.

    Returns
    -------
    float
        The sharpness of the image, normalized by the total number of pixels.
        A higher value indicates a sharper image. The absolute value is arbitrary and
        holds no physical meaning, but it can be used to compare sharpness
        between different images.
    """
    array = data.values
    # Scale the image to 8-bit unsigned integers
    vmin = array.min()
    vspan = array.max() - vmin
    a = (array - vmin) * 255.0 / vspan if vspan > 0 else array - vmin
    image = Image.fromarray(a.astype('uint8'))
    # Convert to grayscale
    gray = image.convert("L")
    # Resizing to something not too large yields better results
    nx, ny = gray.size
    size = nx * ny
    if size > max_size:
        scaling = np.sqrt(max_size / size)
        gray = gray.resize((int(nx * scaling), int(ny * scaling)))
    # Remove noise via median filter
    denoised = gray.filter(ImageFilter.MedianFilter(3))
    # Find edges
    edges = np.array(denoised.filter(ImageFilter.FIND_EDGES))
    # Sum edges to compute sharpness (removing borders to avoid edge effects)
    sharpness = edges[1:-1, 1:-1].sum() / np.prod(edges.shape)
    return sharpness
