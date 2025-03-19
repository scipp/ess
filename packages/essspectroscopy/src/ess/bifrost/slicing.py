# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

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
    a3 = sc.lookup(angles['a3'], 'time')
    a4 = sc.lookup(angles['a4'], 'time')
    graph = {
        'a3': lambda event_time_zero: a3[event_time_zero],
        'a4': lambda event_time_zero: a4[event_time_zero],
    }
    grouped = data.transform_coords(('a3', 'a4'), graph=graph).group('a3', 'a4')
    return DataGroupedByRotation[RunType](grouped)


providers = (group_by_rotation,)
