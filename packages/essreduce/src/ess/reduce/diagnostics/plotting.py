# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Diagnostics plots.
"""

import matplotlib.pyplot as plt
import plopp as pp
import scipp as sc
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d_with_marginals(
    da: sc.DataArray,
    *,
    title: str = "",
    event_coord: str = "event_time_offset",
    event_n_bins: int = 64,
) -> object:
    """Plot a 2D histogram and its two marginal histograms.

    This function creates a 2-dimensional histogram plot and shows both
    marginal 1-dimensional histograms above and to the right of the main plot.

    Parameters
    ----------
    da:
        2D data array. May be dense or events.
    title:
        Optional title for the figure.
    event_coord:
        If ``da`` is event data, the event coordinate with this name will
        be used to make a histogram.
        The default is meant to neutron events in either a detector or monitor.
    event_n_bins:
        Number of bins to use if binning by ``event_coord``.

    Returns
    -------
    :
        A figure with the plots.
    """
    with plt.ioff():  # stop ipympl from auto-displaying the raw figure as well
        _, ax = plt.subplots()

    data = _maybe_histogram(da, coord=event_coord, n_bins=event_n_bins)

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_right = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    cax = divider.append_axes("right", "4%", pad="5%")

    fig = data.plot(ax=ax, cax=cax)
    data.sum(data.dims[0]).plot(ax=ax_top)

    data_y = data.sum(data.dims[1])
    pp.xyplot(data_y.data, data_y.coords[data.dims[0]], ax=ax_right)

    ax_top.xaxis.set_tick_params(labelbottom=False)
    ax_right.yaxis.set_tick_params(labelleft=False)
    ax_top.set_xlabel("")
    ax_right.set_ylabel("")

    if title:
        ax_top.set_title(title)

    return fig


def _maybe_histogram(
    monitor: sc.DataArray, *, coord: str, n_bins: int = 64
) -> sc.DataArray:
    if monitor.is_binned:
        return monitor.hist({coord: n_bins})
    return monitor
