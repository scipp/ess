# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for ESS powder reduction."""

from typing import Any

import scipp as sc
import scippnexus as snx
from ess.reduce import nexus

from ess.powder.types import (
    DetectorBankSizes,
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
    """
    Load detector from NeXus, but with event data replaced by placeholders.

    Currently the placeholder is the detector number, but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the detector that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Loading thus proceeds in three steps:

    1. This function loads the detector, but replaces the event data with placeholders.
    2. :py:func:`get_detector_array` drops the additional information, returning only
       the contained scipp.DataArray, reshaped to the logical detector shape.
       This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_detector_data` replaces placeholder data values with the
       event data, and adds source and sample positions.
    """
    definitions = snx.base_definitions()
    definitions["NXdetector"] = _StrippedDetector
    dg = nexus.load_detector(
        file_path=file_path,
        detector_name=detector_name,
        definitions=definitions,
    )
    # The name is required later, e.g., for determining logical detector shape
    dg['detector_name'] = detector_name
    return NeXusDetector[RunType](dg)


def load_nexus_monitor(
    file_path: Filename[RunType], monitor_name: NeXusMonitorName[MonitorType]
) -> NeXusMonitor[RunType, MonitorType]:
    """
    Load monitor from NeXus, but with event data replaced by placeholders.

    Currently the placeholder is a size-0 array, but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the monitor that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Loading thus proceeds in three steps:

    1. This function loads the monitor, but replaces the event data with placeholders.
    2. :py:func:`get_monitor_array` drops the additional information, returning only
         the contained scipp.DataArray.
         This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_monitor_data` replaces placeholder data values with the
         event data, and adds source and sample positions.
    """
    definitions = snx.base_definitions()
    definitions["NXmonitor"] = _StrippedMonitor
    monitor = nexus.load_monitor(
        file_path=file_path, monitor_name=monitor_name, definitions=definitions
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


def get_detector_signal_array(
    detector: NeXusDetector[RunType],
    bank_sizes: DetectorBankSizes | None = None,
) -> RawDetector[RunType]:
    """
    Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector. The
    data array is reshaped to the logical detector shape, which by folding the data
    array along the detector_number dimension.
    """
    da = nexus.extract_detector_data(detector)
    if (sizes := (bank_sizes or {}).get(detector['detector_name'])) is not None:
        da = da.fold(dim="detector_number", sizes=sizes)
    return RawDetector[RunType](da)


def get_monitor_signal_array(
    monitor: NeXusMonitor[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> RawMonitor[RunType, MonitorType]:
    """
    Extract the data array corresponding to a monitor's signal field.

    The returned data array includes coords pertaining directly to the
    signal values array, but not additional information about the monitor.
    """
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
    Assemble a detector data array with event data and source- and sample-position.

    Also adds variances to the event data if they are missing.
    """
    grouped = nexus.group_event_data(
        event_data=event_data, detector_number=detector.coords['detector_number']
    )
    return ReducibleDetectorData[RunType](
        _add_variances(grouped)
        .assign_coords(source_position=source_position, sample_position=sample_position)
        .assign_coords(detector.coords)
        .assign_masks(detector.masks)
    )


def assemble_monitor_data(
    monitor: RawMonitor[RunType, MonitorType],
    event_data: MonitorEventData[RunType, MonitorType],
) -> RawMonitorData[RunType, MonitorType]:
    """
    Assemble a monitor data array with event data.

    Also adds variances to the event data if they are missing.
    """
    da = event_data.assign_coords(monitor.coords).assign_masks(monitor.masks)
    return RawMonitorData[RunType, MonitorType](_add_variances(da=da))


def _drop(
    children: dict[str, snx.Field | snx.Group], classes: tuple[snx.NXobject, ...]
) -> dict[str, snx.Field | snx.Group]:
    return {
        name: child
        for name, child in children.items()
        if not (isinstance(child, snx.Group) and (child.nx_class in classes))
    }


class _StrippedDetector(snx.NXdetector):
    """Detector definition without large geometry or event data for ScippNexus.

    Drops NXoff_geometry and NXevent_data groups, data is replaced by detector_number.
    """

    def __init__(
        self, attrs: dict[str, Any], children: dict[str, snx.Field | snx.Group]
    ):
        children = _drop(children, (snx.NXoff_geometry, snx.NXevent_data))
        children['data'] = children['detector_number']
        super().__init__(attrs=attrs, children=children)


class _DummyField:
    """Dummy field that can replace snx.Field in NXmonitor."""

    def __init__(self):
        self.attrs = {}
        self.sizes = {'event_time_zero': 0}
        self.dims = ('event_time_zero',)
        self.shape = (0,)

    def __getitem__(self, key: Any) -> sc.Variable:
        return sc.empty(dims=self.dims, shape=self.shape, unit=None)


class _StrippedMonitor(snx.NXmonitor):
    """Monitor definition without event data for ScippNexus.

    Drops NXevent_data group, data is replaced by a dummy field.
    """

    def __init__(
        self, attrs: dict[str, Any], children: dict[str, snx.Field | snx.Group]
    ):
        children = _drop(children, (snx.NXevent_data,))
        children['data'] = _DummyField()
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
    get_detector_signal_array,
    get_monitor_signal_array,
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
Providers for loading and processing NeXus data.
"""
