# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc

from .mcstas.xml import McStasInstrument
from .types import DetectorIndex, DetectorName, TimeBinSteps

NMXData = NewType("NMXData", sc.DataGroup)
NMXReducedData = NewType("NMXData", sc.DataGroup)


def bin_time_of_arrival(
    nmx_data: sciline.Series[DetectorIndex, NMXData],
    detector_name: sciline.Series[DetectorIndex, DetectorName],
    instrument: McStasInstrument,
    time_bin_step: TimeBinSteps,
) -> NMXReducedData:
    """Bin time of arrival data into ``time_bin_step`` bins."""

    nmx_data = list(nmx_data.values())
    nmx_data = sc.concat(nmx_data, 'panel')
    counts = nmx_data.pop('weights').hist(t=time_bin_step)
    new_coords = instrument.to_coords(*detector_name.values())
    new_coords.pop('pixel_id')

    return NMXReducedData(
        sc.DataGroup(
            dict(
                counts=counts,
                **{**nmx_data, **new_coords},
            )
        )
    )
