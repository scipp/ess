# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""

import scipp as sc
from ess.reduce import nexus
from ess.sans.types import (
    DetectorEventData,
    Filename,
    MonitorEventData,
    MonitorType,
    NeXusDetector,
    NeXusDetectorName,
    NeXusMonitor,
    NeXusMonitorName,
    RawSample,
    RawSource,
    RunType,
)


def load_nexus_sample(file_path: Filename[RunType]) -> RawSample[RunType]:
    return RawSample[RunType](nexus.load_sample(file_path))


def dummy_load_sample(file_path: Filename[RunType]) -> RawSample[RunType]:
    return RawSample[RunType](
        sc.DataGroup({'position': sc.vector(value=[0, 0, 0], unit='m')})
    )


def load_nexus_source(file_path: Filename[RunType]) -> RawSource[RunType]:
    return RawSource[RunType](nexus.load_source(file_path))


def load_nexus_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> NeXusDetector[RunType]:
    # Events will be loaded later. Should we set something else as data instead, or
    # use different NeXus definitions to completely bypass the (empty) event load?
    dg = nexus.load_detector(
        file_path=file_path,
        detector_name=detector_name,
        selection={'event_time_zero': slice(0, 0)},
    )
    # The name is required later, e.g., for determining logical detector shape
    dg['detector_name'] = detector_name
    return NeXusDetector[RunType](dg)


def load_nexus_monitor(
    file_path: Filename[RunType], monitor_name: NeXusMonitorName[MonitorType]
) -> NeXusMonitor[RunType, MonitorType]:
    return NeXusMonitor[RunType, MonitorType](
        nexus.load_monitor(
            file_path=file_path,
            monitor_name=monitor_name,
            selection={'event_time_zero': slice(0, 0)},
        )
    )


def load_detector_event_data(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> DetectorEventData[RunType]:
    da = nexus.load_event_data(file_path=file_path, component_name=detector_name)
    return DetectorEventData[RunType](da)


def load_monitor_event_data(
    file_path: Filename[RunType], monitor_name: NeXusMonitorName[MonitorType]
) -> MonitorEventData[RunType, MonitorType]:
    da = nexus.load_event_data(file_path=file_path, component_name=monitor_name)
    return MonitorEventData[RunType, MonitorType](da)


providers = (
    load_nexus_detector,
    load_nexus_monitor,
    load_nexus_sample,
    load_nexus_source,
    load_detector_event_data,
    load_monitor_event_data,
)
"""Providers for loading single files."""
