# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Cutting BIFROST data."""

from collections.abc import Callable

import scipp as sc
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    InstrumentAngle,
    RawDetector,
    RunType,
    SampleAngle,
)


def group_by_rotation(
    data: RawDetector[RunType],
    sample_angle: SampleAngle[RunType],
    instrument_angle: InstrumentAngle[RunType],
) -> DataGroupedByRotation[RunType]:
    """Group data by rotation angles.

    Parameters
    ----------
    data:
        Detector events with time coordinates.
    sample_angle:
        Sample rotation angle "a3".
        May be time-dependent (1d array with dim "time") or scalar.
    instrument_angle:
        Instrument rotation angle "a4".
        May be time-dependent (1d array with dim "time") or scalar.

    Returns
    -------
    :
        ``data`` grouped by rotation angles "a3" and "a4".
    """
    graph = {'a3': _make_angle_from_time_calculator(sample_angle)}
    group_coords = ['a3']
    drop_coords = []

    if 'time' in data.dims:
        # If the data is time-dependent, a4 corresponds to that time-dependence
        # (the instrument angle is the only dynamic parameter). The data has a 'time'
        # bin-edge coord in this case. Since a4 changes at the bin edges, not within
        # each bin, we can look it up safely using midpoints.
        graph['a4'] = lambda time: sc.lookup(
            instrument_angle, dim='time', mode='previous'
        )[sc.midpoints(time)]
        # a4 replaces the 'time' coord.
        drop_coords.append('time')
    else:
        graph['a4'] = _make_angle_from_time_calculator(instrument_angle)
        group_coords.append('a4')

    grouped = (
        data.transform_coords(('a3', 'a4'), graph=graph)
        .group(*group_coords)
        .drop_coords(drop_coords)
    )
    return DataGroupedByRotation[RunType](grouped)


def _make_angle_from_time_calculator(angle: sc.DataArray) -> Callable[..., sc.Variable]:
    if angle.ndim == 0:
        return lambda: angle.data
    else:
        lut = sc.lookup(angle, 'time', mode='previous')
        return lambda event_time_zero: lut[event_time_zero]


providers = (group_by_rotation,)
