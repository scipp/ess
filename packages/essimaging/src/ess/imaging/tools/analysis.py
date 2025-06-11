# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tools for image analysis and manipulation.
"""

import uuid

import numpy as np
import scipp as sc


def blockify(image: sc.Variable | sc.DataArray, **sizes) -> sc.Variable | sc.DataArray:
    """
    Blockify an image by folding it into blocks of specified sizes.
    The sizes should be provided as keyword arguments, where the keys are
    dimension names and the values are the sizes of the blocks.
    The shape of the input image must be divisible by the block sizes.

    Parameters
    ----------
    image:
        The image to blockify.
    sizes:
        Keyword arguments specifying the block sizes for each dimension.
        For example, `x=4, y=4` will create blocks of size 4x4.
    """
    out = image
    for dim, size in sizes.items():
        out = out.fold(dim=dim, sizes={dim: -1, uuid.uuid4().hex: size})
    return out


def resample(
    image: sc.Variable | sc.DataArray, sizes: dict[str, int], method: str = 'sum'
) -> sc.Variable | sc.DataArray:
    """
    Resample an image by folding it into blocks of specified sizes and applying a
    reduction method.
    The sizes should be provided as a dictionary where the keys are dimension names
    and the values are the sizes of the blocks. The shape of the input image must be
    divisible by the block sizes.

    Parameters
    ----------
    image:
        The image to resample.
    sizes:
        A dictionary specifying the block sizes for each dimension.
        For example, `{'x': 4, 'y': 4}` will create blocks of size 4x4.
    method:
        The reduction method to apply to the blocks. This can be any valid
        reduction method, such as 'sum', 'mean', 'max', etc.
    """
    blocked = blockify(image, **sizes)
    return getattr(sc, method)(blocked, set(blocked.dims) - set(image.dims))


def laplace_2d(
    image: sc.Variable | sc.DataArray, dims: tuple[str, str] | list[str]
) -> sc.Variable | sc.DataArray:
    """
    Compute the Laplace operator of a 2d image using a kernel that approximates
    the second derivative in two dimensions. The kernel is designed to
    highlight areas of rapid intensity change, which are indicative of edges
    in the image.
    The kernel is applied to the image by convolving it with the Laplace operator,
    which is a discrete approximation of the second derivative. The result is
    a new image where each pixel value represents the sum of the second
    derivatives in the x and y directions, effectively highlighting areas of
    high curvature or rapid intensity change.

    Parameters
    ----------
    image:
        The input image to compute the Laplace operator on.
    dims:
        The dimensions of the image over which to compute the Laplace operator.
        Other dimensions will be preserved in the output.
    """
    kernel = [8] + ([-1] * 8)
    ii = np.repeat([0, -1, 1], 3)
    jj = np.tile([0, -1, 1], 3)

    return sc.reduce(
        (
            image[dims[0], (1 + j) : (image.sizes[dims[0]] - 1 + j)][
                dims[1], (1 + i) : (image.sizes[dims[1]] - 1 + i)
            ]
            * k
            for i, j, k in zip(ii, jj, kernel, strict=True)
        )
    ).sum()


def sharpness(
    image: sc.Variable | sc.DataArray, dims: tuple[str, str] | list[str]
) -> sc.Variable | sc.DataArray:
    """
    Calculate the sharpness of an image by computing the Laplace operator
    and summing the absolute values of the results over specified dimensions.
    The sharpness is a measure of the amount of detail in the image, with
    higher values indicating sharper images. The Laplace operator is used to
    detect edges in the image, and the sum of the absolute values of the
    Laplacian highlights areas of rapid intensity change, which are indicative
    of sharp features.

    Parameters
    ----------
    image:
        The input image to compute the sharpness on.
    dims:
        The dimensions of the image over which to compute the sharpness.
        Other dimensions will be preserved in the output.
    """
    lap = laplace_2d(image, dims=dims)
    return sc.abs(lap).sum(dims)
