# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Prototype for event filtering.

IMPORTANT Will be moved to a different place and potentially modified.
"""

from contextlib import contextmanager
from numbers import Real

import scipp as sc

from .types import DetectorTofData, FilteredData, RunType


def _equivalent_bin_indices(a, b) -> bool:
    a_begin = a.bins.constituents["begin"].flatten(to="")
    a_end = a.bins.constituents["end"].flatten(to="")
    b_begin = b.bins.constituents["begin"].flatten(to="")
    b_end = b.bins.constituents["end"].flatten(to="")
    non_empty = a_begin != a_end
    return (
        sc.all((a_begin == b_begin)[non_empty]).value
        and sc.all((a_end == b_end)[non_empty]).value
    )


@contextmanager
def _temporary_bin_coord(data: sc.DataArray, name: str, coord: sc.Variable) -> None:
    if not _equivalent_bin_indices(data, coord):
        raise ValueError("data and coord do not have equivalent bin indices")
    coord = sc.bins(
        data=coord.bins.constituents["data"],
        begin=data.bins.coords["pulse_time"].bins.constituents["begin"],
        end=data.bins.coords["pulse_time"].bins.constituents["end"],
        dim=coord.bins.constituents["dim"],
    )
    data.bins.coords[name] = coord
    yield
    del data.bins.coords[name]


# TODO non-monotonic proton charge -> raise?
def _with_pulse_time_edges(da: sc.DataArray, dim: str) -> sc.DataArray:
    pulse_time = da.coords[dim]
    one = sc.scalar(1, dtype="int64", unit=pulse_time.unit)
    lo = pulse_time[0] - one
    hi = pulse_time[-1] + one
    mid = sc.midpoints(pulse_time)
    da.coords[dim] = sc.concat([lo, mid, hi], dim)
    return da


def remove_bad_pulses(
    data: sc.DataArray, *, proton_charge: sc.DataArray, threshold_factor: Real
) -> sc.DataArray:
    """
    assumes that there are bad pulses
    """
    min_charge = proton_charge.data.mean() * threshold_factor
    good_pulse = _with_pulse_time_edges(proton_charge >= min_charge, proton_charge.dim)
    with _temporary_bin_coord(
        data,
        "good_pulse",
        sc.lookup(good_pulse, good_pulse.dim)[data.bins.coords[good_pulse.dim]],
    ):
        filtered = data.group(sc.array(dims=["good_pulse"], values=[True]))
    filtered = filtered.squeeze("good_pulse").copy(deep=False)
    del filtered.coords["good_pulse"]
    return filtered


def filter_events(data: DetectorTofData[RunType]) -> FilteredData[RunType]:
    """Remove bad events.

    Attention
    ---------
    This function currently does nothing because it is unclear how to filter
    events at ESS.
    In the future, this function will filter out events that
    cannot be used for analysis.

    Parameters
    ----------
    data:
        Input events to be filtered.

    Returns
    -------
    :
        `data` with bad events removed.
    """
    # TODO this needs to filter by proton charge once we know how
    return FilteredData[RunType](data)


providers = (filter_events,)
"""Sciline providers for event filtering."""
