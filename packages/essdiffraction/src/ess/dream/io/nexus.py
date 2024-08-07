# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for DREAM.

Notes on the detector dimensions (2024-05-22):

See https://confluence.esss.lu.se/pages/viewpage.action?pageId=462000005
and the ICD DREAM interface specification for details.

- The high-resolution and SANS detectors have a very odd numbering scheme.
  The scheme attempts to follows some sort of physical ordering in space (x,y,z),
  but it is not possible to reshape the data into all the logical dimensions.
"""

import warnings
from typing import Any

import scipp as sc
import scippnexus as snx
from ess.reduce import nexus

from ess.powder.types import (
    DetectorEventData,
    Filename,
    MonitorEventData,
    MonitorType,
    NeXusDetector,
    NeXusDetectorName,
    NeXusMonitor,
    NeXusMonitorName,
    RawDetector,
    RawMonitor,
    RawMonitorData,
    RawSample,
    RawSource,
    ReducibleDetectorData,
    RunType,
    SamplePosition,
    SourcePosition,
)

DETECTOR_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 32,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {
        "strip": 32,
        "other": -1,
    },
    "sans_detector": lambda x: x.fold(
        dim="detector_number",
        sizes={
            "strip": 32,
            "other": -1,
        },
    ),
}


def load_nexus_sample(file_path: Filename[RunType]) -> RawSample[RunType]:
    return RawSample[RunType](nexus.load_sample(file_path))


def dummy_load_sample(file_path: Filename[RunType]) -> RawSample[RunType]:
    """
    In test files there is not always a sample, so we need a dummy.
    """
    return RawSample[RunType](
        sc.DataGroup({'position': sc.vector(value=[0, 0, 0], unit='m')})
    )


def load_nexus_source(file_path: Filename[RunType]) -> RawSource[RunType]:
    return RawSource[RunType](nexus.load_source(file_path))


def load_nexus_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> NeXusDetector[RunType]:
    definitions = snx.base_definitions()
    definitions["NXdetector"] = FilteredDetector
    # Events will be loaded later. Should we set something else as data instead, or
    # use different NeXus definitions to completely bypass the (empty) event load?
    dg = nexus.load_detector(
        file_path=file_path,
        detector_name=detector_name,
        selection={'event_time_zero': slice(0, 0)},
        definitions=definitions,
    )
    # The name is required later, e.g., for determining logical detector shape
    dg['detector_name'] = detector_name
    return NeXusDetector[RunType](dg)


def load_nexus_monitor(
    file_path: Filename[RunType], monitor_name: NeXusMonitorName[MonitorType]
) -> NeXusMonitor[RunType, MonitorType]:
    # It would be simpler to use something like
    #    selection={'event_time_zero': slice(0, 0)},
    # to avoid loading events, but currently we have files with empty NXevent_data
    # groups so that does not work. Instead, skip event loading and create empty dummy.
    definitions = snx.base_definitions()
    definitions["NXmonitor"] = NXmonitor_no_events
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Failed to load",
        )
        monitor = nexus.load_monitor(
            file_path=file_path,
            monitor_name=monitor_name,
            definitions=definitions,
        )
    empty_events = sc.DataArray(
        sc.empty(dims=['event'], shape=[0], dtype='float32', unit='counts'),
        coords={'event_time_offset': sc.array(dims=['event'], values=[], unit='ns')},
    )
    monitor[f'{monitor_name}_events'] = sc.DataArray(
        sc.bins(
            dim='event',
            data=empty_events,
            begin=sc.empty(dims=['event_time_zero'], shape=[0], unit=None),
        ),
        coords={
            'event_time_zero': sc.datetimes(
                dims=['event_time_zero'], values=[], unit='ns'
            )
        },
    )
    return NeXusMonitor[RunType, MonitorType](monitor)


def get_source_position(
    raw_source: RawSource[RunType],
) -> SourcePosition[RunType]:
    return SourcePosition[RunType](raw_source["position"])


def get_sample_position(
    raw_sample: RawSample[RunType],
) -> SamplePosition[RunType]:
    return SamplePosition[RunType](raw_sample["position"])


def get_detector_data(
    detector: NeXusDetector[RunType],
) -> RawDetector[RunType]:
    da = nexus.extract_detector_data(detector)
    if (sizes := DETECTOR_BANK_SIZES.get(detector['detector_name'])) is not None:
        da = da.fold(dim="detector_number", sizes=sizes)
    return RawDetector[RunType](da)


def get_monitor_data(
    monitor: NeXusMonitor[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor[RunType, MonitorType](
        nexus.extract_monitor_data(monitor).assign_coords(
            position=monitor['position'], source_position=source_position
        )
    )


def assemble_detector_data(
    detector: RawDetector[RunType],
    event_data: DetectorEventData[RunType],
    source_position: SourcePosition[RunType],
    sample_position: SamplePosition[RunType],
) -> ReducibleDetectorData[RunType]:
    """
    Assemble a detector data object with source and sample positions and event data.
    Also adds variances to the event data if they are missing.
    """
    grouped = nexus.group_event_data(
        event_data=event_data, detector_number=detector.coords['detector_number']
    )
    detector.data = grouped.data
    return ReducibleDetectorData[RunType](
        _add_variances(da=detector).assign_coords(
            source_position=source_position, sample_position=sample_position
        )
    )


def assemble_monitor_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    event_data: MonitorEventData[RunType, MonitorType],
) -> RawMonitorData[RunType, MonitorType]:
    meta = monitor_data.drop_coords('event_time_zero')
    da = event_data.assign_coords(meta.coords).assign_masks(meta.masks)
    return RawMonitorData[RunType, MonitorType](_add_variances(da=da))


def _skip(
    _: str, obj: snx.Field | snx.Group, classes: tuple[snx.NXobject, ...]
) -> bool:
    return isinstance(obj, snx.Group) and (obj.nx_class in classes)


class FilteredDetector(snx.NXdetector):
    def __init__(
        self, attrs: dict[str, Any], children: dict[str, snx.Field | snx.Group]
    ):
        children = {
            name: child
            for name, child in children.items()
            if not _skip(name, child, classes=(snx.NXoff_geometry,))
        }
        super().__init__(attrs=attrs, children=children)


class NXmonitor_no_events(snx.NXmonitor):
    def __init__(
        self, attrs: dict[str, Any], children: dict[str, snx.Field | snx.Group]
    ):
        children = {
            name: child
            for name, child in children.items()
            if not _skip(name, child, classes=(snx.NXevent_data,))
        }
        super().__init__(attrs=attrs, children=children)


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


def _add_variances(da: sc.DataArray) -> sc.DataArray:
    out = da.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents['data']
        if content.variances is None:
            content.variances = content.values
    return out


providers = (
    assemble_detector_data,
    assemble_monitor_data,
    get_detector_data,
    get_monitor_data,
    get_sample_position,
    get_source_position,
    load_detector_event_data,
    load_monitor_event_data,
    load_nexus_detector,
    load_nexus_monitor,
    load_nexus_sample,
    load_nexus_source,
)
"""
Providers for loading and processing DREAM NeXus data.
"""
