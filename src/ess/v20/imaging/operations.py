# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc


def _shift(var, dim, forward, out_of_bounds):
    fill = var[dim, 0:1].copy()
    fill.values = np.full_like(fill.values, out_of_bounds)
    if forward:
        return sc.concat([fill, var[dim, :-1]], dim)
    else:
        return sc.concat([var[dim, 1:], fill], dim)


def mask_from_adj_pixels(mask):
    """
    Checks if the adjacent pixels (in 8 directions) are masked to remove
    any noisy pixels which are erroneously masked or unmasked compared to
    it's neighbours

    If all adj. pixels are then the pixel considered is set to True
    If no adj. pixels are then the pixel considered is set to False
    If surrounding pixels have a mix of True/False the val is left as-is

    This function handles border pixels as if they aren't there. So that
    the following happens:
    ------------------------
    |F|T|     ->      |T|T|
    |T|T|             |T|T|
    -----------------------

    Parameters
    ----------
    mask: Existing mask with some positions masked

    Returns
    -------
    mask: Mask copy after completing the op. described above

    """

    mask = mask.copy()

    def make_flip(fill):
        flip = sc.empty(dims=['neighbor', 'y', 'x'],
                        shape=(8, ) + mask.shape,
                        dtype=bool)
        flip['neighbor', 0] = _shift(mask, "x", True, fill)
        flip['neighbor', 1] = _shift(mask, "x", False, fill)
        flip['neighbor', 2] = _shift(mask, "y", True, fill)
        flip['neighbor', 3] = _shift(mask, "y", False, fill)
        flip['neighbor', 4:6] = _shift(flip['neighbor', 0:2], "y", True, fill)
        flip['neighbor', 6:8] = _shift(flip['neighbor', 0:2], "y", False, fill)
        return flip

    # mask if all neighbors masked
    mask = mask | sc.all(make_flip(True), 'neighbor')
    # unmask if no neighbor masked
    mask = mask & sc.any(make_flip(False), 'neighbor')
    return mask


def mean_from_adj_pixels(data):
    """
    Applies a mean across 8 neighboring pixels (plus centre value)
    for data with 'x' and 'y' dimensions (at least).
    Result will calculate mean from slices across additional dimensions.

    For example if there is a tof dimension in addition to x, and y,
    for each set of neighbours the returned mean will take the mean
    tof value in the neighbour group.
    """
    fill = np.finfo(data.values.dtype).min
    has_variances = data.variances is not None
    container = sc.empty(dims=('neighbor', ) + data.dims,
                         dtype=data.dtype,
                         shape=(9, ) + data.shape,
                         with_variances=has_variances,
                         unit=data.unit)
    container['neighbor', 0] = data
    container['neighbor', 1] = _shift(data, "x", True, fill)
    container['neighbor', 2] = _shift(data, "x", False, fill)
    container['neighbor', 3] = _shift(data, "y", True, fill)
    container['neighbor', 4] = _shift(data, "y", False, fill)
    container['neighbor', 5:7] = _shift(container['neighbor', 1:3], "y", True, fill)
    container['neighbor', 7:9] = _shift(container['neighbor', 1:3], "y", False, fill)

    edges_mask = container <= sc.scalar(value=fill, unit=data.unit)
    da = sc.DataArray(data=container, masks={'edges': edges_mask})
    return sc.mean(da, dim='neighbor').data


def _median(neighbors, edges_mask, dim):
    masked_values = np.ma.array(neighbors.values, mask=edges_mask.values, copy=False)
    masked_median_v = np.ma.median(masked_values, axis=0)
    if neighbors.variances is not None:
        masked_median_var = np.ma.median(neighbors.variances, axis=0)
        np.ma.array(neighbors.variances, mask=edges_mask.values, copy=False)
        return sc.Variable(dims=neighbors.dims[1:],
                           values=masked_median_v,
                           variances=masked_median_var)
    return sc.Variable(dims=neighbors.dims[1:], values=masked_median_v)


def median_from_adj_pixels(data):
    """
    Applies a median across 8 neighboring pixels (plus centre value)
    for data with 'x' and 'y' dimensions (at least).
    Result will calculate median from slices across additional dimensions.

    For example if there is a tof dimension in addition to x, and y,
    for each set of neighbours the returned median will take the median
    tof value in the neighbour group.
    """
    fill = np.finfo(data.values.dtype).min
    has_variances = data.variances is not None
    container = sc.empty(dims=('neighbor', ) + data.dims,
                         dtype=data.dtype,
                         shape=(9, ) + data.shape,
                         with_variances=has_variances)
    container['neighbor', 0] = data
    container['neighbor', 1] = _shift(data, "x", True, fill)
    container['neighbor', 2] = _shift(data, "x", False, fill)
    container['neighbor', 3] = _shift(data, "y", True, fill)
    container['neighbor', 4] = _shift(data, "y", False, fill)
    container['neighbor', 5:7] = _shift(container['neighbor', 1:3], "y", True, fill)
    container['neighbor', 7:9] = _shift(container['neighbor', 1:3], "y", False, fill)

    edges_mask = container <= sc.scalar(value=fill, variance=fill)
    return _median(container, edges_mask, dim='neighbor')


def groupby2D(data, nx_target, ny_target, x='x', y='y', z='wavelength'):

    element_width_x = data.sizes[x] // nx_target
    element_width_y = data.sizes[y] // ny_target

    xx = sc.Variable(dims=[x], values=np.arange(data.sizes[x]) // element_width_x)
    yy = sc.Variable(dims=[y], values=np.arange(data.sizes[y]) // element_width_y)
    grid = xx + nx_target * yy
    spectrum_mapping = sc.Variable(dims=["spectrum"],
                                   values=np.ravel(grid.values, order='F'))

    reshaped = sc.Dataset()
    for key, val in data.items():
        reshaped[key] = sc.flatten(x=val, dims=[y, x], to='spectrum')

    reshaped.coords["spectrum_mapping"] = spectrum_mapping

    grouped = sc.groupby(reshaped, "spectrum_mapping").sum("spectrum")

    reshaped = sc.Dataset()
    for key, val in grouped.items():
        item = sc.fold(x=val,
                       dim="spectrum_mapping",
                       dims=[y, x],
                       shape=(ny_target, nx_target))
        reshaped[key] = item
    return reshaped
