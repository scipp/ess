# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Cutting BIFROST data."""

from collections.abc import Callable

import scipp as sc

from ess.spectroscopy.types import (
    DataGroupedByRotation,
    DetectorData,
    InstrumentAngles,
    RunType,
)


def group_by_rotation(
    data: DetectorData[RunType],
    angles: InstrumentAngles[RunType],
) -> DataGroupedByRotation[RunType]:
    """Group data by rotation angles.

    Parameters
    ----------
    data:
        Detector events with time coordinates.
    angles:
        Data group with entries "a3" and "a4".
        Each entry may be time-dependent (1d array with dim "time") or scalar.

    Returns
    -------
    :
        ``data`` grouped by rotation angles "a3" and "a4".
    """
    graph = {
        name: _make_angle_from_time_calculator(angle) for name, angle in angles.items()
    }
    grouped = data.transform_coords(('a3', 'a4'), graph=graph).group('a3', 'a4')
    return DataGroupedByRotation[RunType](grouped)


def _make_angle_from_time_calculator(angle: sc.DataArray) -> Callable[..., sc.Variable]:
    if angle.ndim == 0:
        return lambda: angle.data
    else:
        lut = sc.lookup(angle, 'time')
        return lambda event_time_zero: lut[event_time_zero]


providers = (group_by_rotation,)
