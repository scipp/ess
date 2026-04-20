# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Smoothing arrays data.
"""

import scipp as sc
from scipp.scipy.signal import butter

from .logging import get_logger


def _ensure_no_variances(var: sc.DataArray) -> sc.DataArray:
    if var.variances is not None:
        get_logger().warning(
            'Tried to smoothen data with uncertainties. '
            'This is not supported because the results would be highly correlated.\n'
            'Instead, the variances are ignored and the output '
            'will be returned without any!'
            '\n--------------------------------------------------\n'
            'If you know a good solution for handling uncertainties in such a case, '
            'please contact the scipp developers! (e.g. via https://github.com/scipp)'
            '\n--------------------------------------------------\n'
        )
        return sc.values(var)
    return var


def lowpass(
    da: sc.DataArray, *, dim: str, N: int, Wn: sc.Variable, coord: str | None = None
) -> sc.DataArray:
    """
    Smooth data using a lowpass frequency filter.

    Applies a lowpass Butterworth filter to `da.data` based on the sampling rate
    defined by `coord`.
    See :py:func:`scipp.signal.butter` for information on filter design.

    Important
    ---------
    If `coord` is bin-edges, it is first converted to bin-centers using
    :func:`scipp.midpoints`.
    This is only valid for linearly-spaced edges.

    Parameters
    ----------
    da:
        Data to smoothen.
    dim:
        Dimension along which to smooth.
    coord:
        Name of the coordinate that defines the sampling frequency.
        Defaults to `dim`.
    N:
        Order of the lowpass filter.
    Wn:
        Critical frequency of the filter.

    Returns
    -------
    :
        Smoothed `da`.

    See Also
    --------
    scipp.signal.butter scipp.signal.sosfiltfilt

    Examples
    --------

       >>> from ess.powder.smoothing import lowpass
       >>> x = sc.linspace(dim='x', start=1.1, stop=4.0, num=1000, unit='m')
       >>> y = sc.sin(x * sc.scalar(1.0, unit='rad/m'))
       >>> y += sc.sin(x * sc.scalar(400.0, unit='rad/m'))
       >>> noisy = sc.DataArray(data=y, coords={'x': x})
       >>> smooth = lowpass(noisy, dim='x', N=4, Wn=20 / x.unit)
    """
    da = _ensure_no_variances(da)
    coord = dim if coord is None else coord

    if da.coords[coord].sizes[dim] == da.sizes[dim] + 1:
        da = da.copy(deep=False)
        da.coords[coord] = sc.midpoints(da.coords[coord], dim)

    return butter(da.coords[coord], N=N, Wn=Wn).filtfilt(da, dim)
