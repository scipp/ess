# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc

from ..types import RawData, RunType


class RawDataWithComponentUserOffsets(
    sciline.Scope[RunType, sc.DataArray], sc.DataArray
):
    """Raw data with applied user configuration for component positions."""


SampleOffset = NewType('SampleOffset', sc.Variable)
DetectorBankOffset = NewType('DetectorBankOffset', sc.Variable)


def apply_component_user_offsets_to_raw_data(
    data: RawData[RunType],
    sample_offset: SampleOffset,
    detector_bank_offset: DetectorBankOffset,
) -> RawDataWithComponentUserOffsets[RunType]:
    """Apply user configuration to raw data.

    Parameters
    ----------
    data:
        Raw data.
    sample_offset:
        Sample offset.
    detector_bank_offset:
        Detector bank offset.
    """
    data = data.copy(deep=False)
    sample_pos = data.coords['sample_position']
    data.coords['sample_position'] = sample_pos + sample_offset.to(
        unit=sample_pos.unit, copy=False
    )
    pos = data.coords['position']
    data.coords['position'] = pos + detector_bank_offset.to(unit=pos.unit, copy=False)
    return RawDataWithComponentUserOffsets[RunType](data)
