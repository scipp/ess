# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Detector and shaping tools for DREAM."""

from ess.powder.types import FlatDetectorData, RawDetectorData, RunType


def flatten_detector_dimensions(
    data: RawDetectorData[RunType],
) -> FlatDetectorData[RunType]:
    """Flatten logical detector dimensions to a single ``spectrum``.

    Parameters
    ----------
    data:
        Raw detector data in logical dimensions.
        Logical detector dimensions must be contiguous.

    Returns
    -------
    :
        Flattened detector data with a ``spectrum`` dimension.
    """
    logical_dims = {'module', 'segment', 'counter', 'wire', 'strip', 'sector'}
    actual_dims = tuple(dim for dim in data.dims if dim in logical_dims)
    return FlatDetectorData[RunType](data.flatten(dims=actual_dims, to='spectrum'))


providers = (flatten_detector_dimensions,)
"""Sciline providers for DREAM detector handling."""
