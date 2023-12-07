# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Dict, Optional

import scipp as sc
from scippneutron.conversion.graph import beamline, tof

from .logging import get_logger
from .smoothing import lowpass
from .types import (
    AccumulatedProtonCharge,
    DspacingBins,
    FilteredData,
    FocussedData,
    NormalizedByProtonCharge,
    NormalizedByVanadium,
    RunType,
    SampleRun,
    VanadiumRun,
)


def normalize_by_monitor(
    data: sc.DataArray,
    *,
    monitor: sc.DataArray,
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
        A histogrammed monitor.
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
    if 'wavelength' not in monitor.coords:
        monitor = monitor.transform_coords(
            'wavelength',
            graph={**beamline.beamline(scatter=False), **tof.elastic("tof")},
            keep_inputs=False,
            keep_intermediate=False,
            keep_aliases=False,
        )

    if wavelength_edges is not None:
        monitor = monitor.rebin(wavelength=wavelength_edges)
    if smooth_args is not None:
        get_logger().info(
            "Smoothing monitor for normalization using "
            "ess.diffraction.smoothing.lowpass with %s.",
            smooth_args,
        )
        monitor = lowpass(monitor, dim='wavelength', **smooth_args)
    return data.bins / sc.lookup(func=monitor, dim='wavelength')


def normalize_by_vanadium(
    data: FocussedData[SampleRun],
    vanadium: FocussedData[VanadiumRun],
    edges: DspacingBins,
) -> NormalizedByVanadium:
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


def normalize_by_proton_charge(
    data: FilteredData[RunType], proton_charge: AccumulatedProtonCharge[RunType]
) -> NormalizedByProtonCharge[RunType]:
    """Normalize data by an accumulated proton charge.

    Parameters
    ----------
    data:
        Un-normalized data array as events or a histogram.
    proton_charge:
        Accumulated proton charge over the entire run.

    Returns
    -------
    :
        ``data / proton_charge``
    """
    return NormalizedByProtonCharge[RunType](data / proton_charge)


providers = (normalize_by_proton_charge, normalize_by_vanadium)
"""Sciline providers for diffraction corrections."""
