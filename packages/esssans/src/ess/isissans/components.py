# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc

from ..sans.types import (
    BeamCenter,
    CalibratedDetector,
    MonitorType,
    RawDetector,
    RawMonitor,
    RawMonitorData,
    RunType,
    ScatteringRunType,
)


class MonitorOffset(sciline.Scope[MonitorType, sc.Variable], sc.Variable):
    """Offset for monitor position"""


SampleOffset = NewType('SampleOffset', sc.Variable)
DetectorBankOffset = NewType('DetectorBankOffset', sc.Variable)


def apply_component_user_offsets_to_raw_data(
    data: RawDetector[ScatteringRunType],
    sample_offset: SampleOffset,
    detector_bank_offset: DetectorBankOffset,
    beam_center: BeamCenter,
) -> CalibratedDetector[ScatteringRunType]:
    """Apply user offsets to raw data.

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
    data.coords['user_position'] = pos + detector_bank_offset.to(
        unit=pos.unit, copy=False
    )
    data.coords['position'] = data.coords['user_position'] - beam_center
    return CalibratedDetector[ScatteringRunType](data)


def apply_component_user_offsets_to_raw_monitor(
    monitor_data: RawMonitor[RunType, MonitorType],
    monitor_offset: MonitorOffset[MonitorType],
) -> RawMonitorData[RunType, MonitorType]:
    """Apply user offsets to raw monitor.
    Parameters
    ----------
    monitor_data:
        Raw monitor data.
    monitor_offset:
        Offset to apply to monitor position.
    """
    monitor_data = monitor_data.copy(deep=False)
    pos = monitor_data.coords['position']
    monitor_data.coords['position'] = pos + monitor_offset.to(unit=pos.unit, copy=False)
    return RawMonitorData[RunType, MonitorType](monitor_data)


providers = (
    apply_component_user_offsets_to_raw_data,
    apply_component_user_offsets_to_raw_monitor,
)
