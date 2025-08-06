# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tools for image analysis and manipulation.
"""

import uuid
from collections.abc import Callable
from itertools import combinations

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
        out = out.fold(dim=dim, sizes={dim: -1, uuid.uuid4().hex[:7]: size})
    return out


def resample(
    image: sc.Variable | sc.DataArray,
    sizes: dict[str, int],
    method: str | Callable = 'sum',
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
        For example, ``{'x': 4, 'y': 4}`` will create blocks of size 4x4.
    method:
        The reduction method to apply to the blocks. This can be a string referring to
        any valid Scipp reduction method, such as 'sum', 'mean', 'max', etc.
        Alternatively, a custom reduction function can be provided. The function
        signature should accept a ``scipp.Variable`` or ``scipp.DataArray`` as first
        argument and a set of dimensions to reduce over as second argument. The
        function should return a ``scipp.Variable`` or ``scipp.DataArray``.
    """
    blocked = blockify(image, **sizes)
    if isinstance(method, str):
        return getattr(sc, method)(blocked, set(blocked.dims) - set(image.dims))
    return method(blocked, set(blocked.dims) - set(image.dims))


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

    lp2d = sc.reduce(
        (
            image[dims[0], (1 + j) : (image.sizes[dims[0]] - 1 + j)][
                dims[1], (1 + i) : (image.sizes[dims[1]] - 1 + i)
            ]
            * k
            for i, j, k in zip(ii, jj, kernel, strict=True)
        )
    ).sum()

    lp2d.unit = ""  # Laplacian is dimensionless
    out = (
        sc.DataArray(data=sc.zeros(sizes=image.sizes, dtype=lp2d.dtype))
        .assign_coords(image.coords)
        .assign_masks(image.masks)
    )
    out[dims[0], 1:-1][dims[1], 1:-1] = lp2d
    return out


def _prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            n //= i
        else:
            i += 1
    if n > 1:
        factors.append(n)
    return factors


def _best_subset_product(factors, target):
    best_product = 1
    for r in range(1, len(factors) + 1):
        for combo in combinations(factors, r):
            prod = np.prod(combo)
            if abs(prod - target) < abs(best_product - target):
                best_product = prod
    return best_product


def sharpness(
    image: sc.Variable | sc.DataArray,
    dims: tuple[str, str] | list[str],
    max_size: int | None = 512,
) -> sc.Variable | sc.DataArray:
    """
    Calculate the sharpness of an image by computing the Laplace operator
    and summing the absolute values of the results over specified dimensions.
    The sharpness is a measure of the amount of detail in the image, with
    higher values indicating sharper images. The Laplace operator is used to
    detect edges in the image, and the variance of the Laplacian highlights areas of
    rapid intensity change, which are indicative of sharp features.

    Parameters
    ----------
    image:
        The input image to compute the sharpness on.
    dims:
        The dimensions of the image over which to compute the sharpness.
        Other dimensions will be preserved in the output.
    max_size:
        The maximum size of the image to compute the sharpness on. If the
        image is larger than this size, it will be downsampled to fit within
        the specified maximum size. This is useful for large images where
        computing the Laplace operator directly would be computationally
        expensive.
    """
    if max_size is not None:
        sizes = {}
        for dim in dims:
            if image.sizes[dim] > max_size:
                # Decompose size into prime numbers to find the best subset product
                # closest to the maximum size
                factors = _prime_factors(image.sizes[dim])
                best_product = _best_subset_product(factors, max_size)
                sizes[dim] = image.sizes[dim] // best_product
        image = resample(image, sizes=sizes)

    return laplace_2d(image, dims=dims).var(dim=dims, ddof=1)
