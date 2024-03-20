# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import sciline
import scipp as sc

from ..sans.types import (
    ConfiguredReducibleDataData,
    ConfiguredReducibleMonitor,
    MonitorType,
    RawData,
    RawMonitor,
    RunType,
    ScatteringRunType,
)


class MonitorOffset(sciline.Scope[MonitorType, sc.Variable], sc.Variable):
    """Offset for monitor position"""


SampleOffset = NewType('SampleOffset', sc.Variable)
DetectorBankOffset = NewType('DetectorBankOffset', sc.Variable)


def apply_component_user_offsets_to_raw_data(
    data: RawData[ScatteringRunType],
    sample_offset: Optional[SampleOffset],
    detector_bank_offset: Optional[DetectorBankOffset],
) -> ConfiguredReducibleDataData[ScatteringRunType]:
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
    if sample_offset is not None:
        sample_pos = data.coords['sample_position']
        data.coords['sample_position'] = sample_pos + sample_offset.to(
            unit=sample_pos.unit, copy=False
        )
    if detector_bank_offset is not None:
        pos = data.coords['position']
        data.coords['position'] = pos + detector_bank_offset.to(
            unit=pos.unit, copy=False
        )
    return ConfiguredReducibleDataData[ScatteringRunType](data)


def apply_component_user_offsets_to_raw_monitor(
    monitor_data: RawMonitor[RunType, MonitorType],
    monitor_offset: Optional[MonitorOffset[MonitorType]],
) -> ConfiguredReducibleMonitor[RunType, MonitorType]:
    """Apply user offsets to raw monitor.
    Parameters
    ----------
    monitor_data:
        Raw monitor data.
    monitor_offset:
        Offset to apply to monitor position.
    """
    monitor_data = monitor_data.copy(deep=False)
    if monitor_offset is not None:
        pos = monitor_data.coords['position']
        monitor_data.coords['position'] = pos + monitor_offset.to(
            unit=pos.unit, copy=False
        )
    return ConfiguredReducibleMonitor[RunType, MonitorType](monitor_data)


providers = (
    apply_component_user_offsets_to_raw_data,
    apply_component_user_offsets_to_raw_monitor,
)
