# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .mcstas.xml import McStasInstrument
from .types import DetectorName, TimeBinSteps

NMXData = NewType("NMXData", sc.DataGroup)
NMXReducedData = NewType("NMXReducedData", sc.DataGroup)


def bin_time_of_arrival(
    nmx_data: NMXData,
    detector_name: DetectorName,
    instrument: McStasInstrument,
    time_bin_step: TimeBinSteps,
) -> NMXReducedData:
    """Bin time of arrival data into ``time_bin_step`` bins."""

    counts = nmx_data.pop('weights').hist(t=time_bin_step)
    new_coords = instrument.to_coords(detector_name)
    new_coords.pop('pixel_id')

    return NMXReducedData(sc.DataGroup(counts=counts, **{**nmx_data, **new_coords}))


def merge_panels(*panel: NMXReducedData) -> NMXReducedData:
    return NMXReducedData(sc.concat(panel, 'panel'))
