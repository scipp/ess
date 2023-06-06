# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Union

import plopp as pp
import scipp as sc


def _group_and_concat(da: sc.DataArray, dims: tuple) -> sc.DataArray:
    """
    Group the input data using groups specified by the ``dims``.
    Then, remove any extra dimension that is not in ``dims`` by concatenating the
    event data in the bins.

    Parameters
    ----------
    da:
        DataArray to group.
    dims:
        Coordinates to use for grouping.
    """
    in_dims = set(da.dims)
    out_dims = set(dims)
    if (not dims) or (in_dims == out_dims):
        return da
    grouped = da.group(*list(out_dims - in_dims))
    return grouped.bins.concat(list(set(grouped.dims) - out_dims))


def to_logical_dims(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to logical dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return da.group('module', 'segment', 'counter', 'wire', 'strip')


def wire_vs_strip(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to wire vs strip dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return _group_and_concat(da, dims=('wire', 'strip'))


def module_vs_segment(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to module vs segment dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return _group_and_concat(da, dims=('module', 'segment'))


def module_vs_wire(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to module vs wire dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return _group_and_concat(da, dims=('module', 'wire'))


def module_vs_strip(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to module vs strip dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return _group_and_concat(da, dims=('module', 'strip'))


def segment_vs_strip(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to segment vs strip dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return _group_and_concat(da, dims=('segment', 'strip'))


def segment_vs_wire(da: sc.DataArray) -> sc.DataArray:
    """
    Reshape the input data to segment vs wire dimensions.

    Parameters
    ----------
    da:
        DataArray to reshape.
    """
    return _group_and_concat(da, dims=('segment', 'wire'))


def tof_navigator(
    da: sc.DataArray, bins: Union[int, sc.Variable] = 300
) -> sc.DataArray:
    """
    Make a plot of the data with a slider to navigate the time-of-flight dimension.

    Parameters
    ----------
    da:
        DataArray to plot.
    bins:
        Binning to use for histogramming along the `tof` dimension before plotting.
    """
    dims = list(set(da.dims) - {'tof'})
    return pp.slicer(da.hist(tof=bins), keep=dims)
