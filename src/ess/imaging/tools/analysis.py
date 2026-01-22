# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tools for image analysis and manipulation.
"""

from collections.abc import Callable
from itertools import combinations

import numpy as np
import scipp as sc


def blockify(
    image: sc.Variable | sc.DataArray, sizes: dict[str, int]
) -> sc.Variable | sc.DataArray:
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
        The block sizes for each dimension.
        For example, `{'x': 4, 'y': 4}` will create blocks of size 4x4.
    """
    out = image
    for dim, size in sizes.items():
        i = 0
        while f'newdim{i}' in out.dims:
            i += 1
        out = out.fold(dim=dim, sizes={dim: -1, f'newdim{i}': size})
    return out


def _is_1d_sorted_bin_edges(coords: sc.Coords, coord_name: str) -> bool:
    return (
        (coord := coords[coord_name]).ndim == 1
        and coords.is_edges(coord_name)
        and (
            bool(sc.issorted(coord, dim=coord.dim, order='ascending'))
            or bool(sc.issorted(coord, dim=coord.dim, order='descending'))
        )
    )


def _is_non_bin_edges(coords: sc.Coords, coord_name: str) -> bool:
    return not any(
        coords.is_edges(coord_name, dim=dim) for dim in coords[coord_name].dims
    )


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
        Any dimensions with size ``1`` will be ignored.
        If all sizes are set to ``1``,
        it will not apply ``method`` and return a copy of the input resampling image.
    method:
        The reduction method to apply to the blocks. This can be a string referring to
        any valid Scipp reduction method, such as 'sum', 'mean', 'max', etc.
        Alternatively, a custom reduction function can be provided. The function
        signature should accept a ``scipp.Variable`` or ``scipp.DataArray`` as first
        argument and a set of dimensions to reduce over as second argument. The
        function should return a ``scipp.Variable`` or ``scipp.DataArray``.


    Notes
    -----
    If the image is a ``scipp.DataArray``,
    bin edges in the resampling dimensions
    will be preserved if they are 1-dimensional and sorted (they are dropped otherwise).
    New bin edges will be chosen according to the resampling sizes.
    For midpoint coordinates, new coordinates will be average values
    of each resampled block.

    .. warning::
        The coordinates in the output may not be correct
        if they are not sorted or not linear.

    """
    # Filter the resample sizes first.
    sizes = {dim: size for dim, size in sizes.items() if size != 1}
    if not sizes:
        return image.copy()

    blocked = blockify(image, sizes=sizes)
    _method = getattr(sc, method) if isinstance(method, str) else method
    out = _method(blocked, set(blocked.dims) - set(image.dims))

    if isinstance(image, sc.DataArray):
        # Restore the coordinates dropped by the `_method` if possible.
        _dropped_cnames = set(image.coords.keys()) - set(out.coords.keys())

        for name in _dropped_cnames:
            coord = image.coords[name]
            if _is_1d_sorted_bin_edges(image.coords, name):
                out.coords[name] = coord[coord.dim, :: sizes[coord.dim]]
            elif _is_non_bin_edges(image.coords, name):
                folded_coord = blocked.coords[name]
                reduced_dims = set(folded_coord.dims) - set(coord.dims)
                out.coords[name] = folded_coord.mean(reduced_dims)

    return out


def resize(
    image: sc.Variable | sc.DataArray,
    sizes: dict[str, int],
    method: str | Callable = 'sum',
) -> sc.Variable | sc.DataArray:
    """
    Resize an image by folding it into blocks of specified sizes and applying a
    reduction method.
    The sizes should be provided as a dictionary where the keys are dimension names
    and the values are the sizes of the blocks. The shape of the input image must be
    divisible by the block sizes.

    Parameters
    ----------
    image:
        The image to resample.
    sizes:
        A dictionary specifying the desired size of the output image for each dimension.
        The original sizes should be divisible by the specified sizes.
        For example, ``{'x': 128, 'y': 128}`` will create an output image of size
        128x128.
        Any dimensions with same sizes to the resizing image will be ignored.
        If the output image sizes will be same as the input image sizes,
        it will not apply the ``method`` and return a copy of the input image.
    method:
        The reduction method to apply to the blocks. This can be a string referring to
        any valid Scipp reduction method, such as 'sum', 'mean', 'max', etc.
        Alternatively, a custom reduction function can be provided. The function
        signature should accept a ``scipp.Variable`` or ``scipp.DataArray`` as first
        argument and a set of dimensions to reduce over as second argument. The
        function should return a ``scipp.Variable`` or ``scipp.DataArray``.


    Notes
    -----
    If the image is a ``scipp.DataArray``,
    bin edges in the resizing dimensions
    will be preserved if they are 1-dimensional and sorted.
    New bin edges will be chosen according to the resizing sizes.
    For other coordinates, new coordinates will be average values
    of each resized blocks.

    .. warning::
        The coordinates in the output may not be correct
        if they are not sorted or not linear.

    """
    block_sizes = {}
    for dim, size in sizes.items():
        if image.sizes[dim] % size != 0:
            raise ValueError(
                f"Size of dimension '{dim}' ({image.sizes[dim]}) is not divisible by"
                f" the requested size ({size})."
            )
        block_sizes[dim] = image.sizes[dim] // size

    return resample(image, sizes=block_sizes, method=method)


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
                sizes[dim] = _best_subset_product(factors, max_size)
        image = resize(image, sizes=sizes)

    return laplace_2d(image, dims=dims).var(dim=dims, ddof=1)
