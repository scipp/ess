# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Dict, Optional

import scipp as sc
from scippneutron.conversion.graph import beamline, tof

from .logging import get_logger
from .smoothing import lowpass


def normalize_by_monitor(
    data: sc.DataArray,
    *,
    monitor: str,
    wavelength_edges: Optional[sc.Variable] = None,
    smooth_args: Optional[Dict[str, Any]] = None,
) -> sc.DataArray:
    """
    Normalize event data by a monitor.

    The input is converted to wavelength if it does not already contain wavelengths.

    Parameters
    ----------
    data:
        Input event data.
    monitor:
        Name of a histogrammed monitor. Must be stored as metadata in `data`.
    wavelength_edges:
        If given, rebin the monitor with these edges.
    smooth_args:
        If given, the monitor histogram is smoothed with
        :func:`ess.diffraction.lowpass` before dividing into `data`.
        `smooth_args` is passed as keyword arguments to
        :func:`ess.diffraction.lowpass`. If ``None``, the monitor is not smoothed.

    Returns
    -------
    :
        `data` normalized by a monitor.
    """
    mon = data.meta[monitor].value
    if 'wavelength' not in mon.coords:
        mon = mon.transform_coords(
            'wavelength',
            graph={**beamline.beamline(scatter=False), **tof.elastic("tof")},
            keep_inputs=False,
            keep_intermediate=False,
            keep_aliases=False,
        )

    if wavelength_edges is not None:
        mon = mon.rebin(wavelength=wavelength_edges)
    if smooth_args is not None:
        get_logger().info(
            "Smoothing monitor '%s' for normalization using "
            "ess.diffraction.smoothing.lowpass with %s.",
            monitor,
            smooth_args,
        )
        mon = lowpass(mon, dim='wavelength', **smooth_args)
    return data.bins / sc.lookup(func=mon, dim='wavelength')


def normalize_by_vanadium(
    data: sc.DataArray, *, vanadium: sc.DataArray, edges: sc.Variable
) -> sc.DataArray:
    """
    Normalize sample data by a vanadium measurement.

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    edges:
        `vanadium` is histogrammed into these bins before dividing the data by it.

    Returns
    -------
    :
        `data` normalized by `vanadium`.
    """
    norm = sc.lookup(vanadium.hist({edges.dim: edges}), dim=edges.dim)
    # Converting to unit 'one' because the division might produce a unit
    # with a large scale if the proton charges in data and vanadium were
    # measured with different units.
    return (data.bins / norm).to(unit='one', copy=False)
