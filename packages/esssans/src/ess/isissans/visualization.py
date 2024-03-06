# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Plotting functions for ISIS data.
"""
import warnings
from typing import Any

import scipp as sc


def plot_flat_detector_xy(
    da: sc.DataArray, pixels_per_tube: int = 512, **kwargs: Any
) -> Any:
    """
    Plot a 2-D view of data from a single flat detector bank.

    This is an alternative to the `scn.instrument_view` function, avoiding the 3-D
    rendering of the instrument. The exact X and Y coordinates of the pixels are
    used for the 2-D plot.

    There are currently no checks that the data array is actually from a single
    detector bank, or that it is flat. The caller is responsible for ensuring this.

    Parameters
    ----------
    da:
        The data array to plot. Must have a 'position' coord and a single dimension.
    pixels_per_tube:
        The number of pixels per tube. Defaults to 512.
    kwargs:
        Additional arguments passed to `sc.plot`.
    """
    if da.bins is not None:
        da = da.hist()
    da.coords['x'] = da.coords['position'].fields.x.copy()
    da.coords['y'] = da.coords['position'].fields.y.copy()
    folded = da.fold(da.dim, sizes={'y': -1, 'x': pixels_per_tube})
    y = folded.coords['y']
    if sc.all(y.min('x') == y.max('x')):
        folded.coords['y'] = y.min('x')
    else:
        raise ValueError(
            'Cannot plot 2-D instrument view of data array with non-constant '
            'y coordinate along tubes. Use scippneutron.instrument_view instead.'
        )
    plot_kwargs = dict(aspect='equal')
    plot_kwargs.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = folded.plot(**plot_kwargs)
    return out
