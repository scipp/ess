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


def apply_beam_center(
    data: RawDetector[ScatteringRunType], beam_center: BeamCenter
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
    return CalibratedDetector[ScatteringRunType](
        data.assign_coords(position=data.coords['position'] - beam_center)
    )


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
    apply_beam_center,
    apply_component_user_offsets_to_raw_monitor,
)
