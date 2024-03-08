# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""
from collections.abc import Mapping
from functools import reduce
from typing import Optional, Union

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce import nexus

from ..sans.common import gravity_vector
from ..sans.types import (
    BackgroundRun,
    DataFolder,
    Filename,
    FilenameType,
    FilePath,
    Incident,
    LoadedDetector,
    LoadedMonitor,
    LoadedSingleFileDetector,
    LoadedSingleFileMonitor,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    NeXusSampleName,
    NeXusSourceName,
    RawSample,
    RawSource,
    RunType,
    SampleRun,
    ScatteringRunType,
    TransformationPath,
    Transmission,
)
from .general import NEXUS_INSTRUMENT_PATH
from .general import default_parameters as params

DETECTOR_BANK_RESHAPING = {
    params[NeXusDetectorName]: lambda x: x.fold(
        dim='detector_number', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
    )
}


def _patch_data(
    da: sc.DataArray,
    source_position: sc.Variable,
    sample_position: Optional[sc.Variable] = None,
) -> sc.DataArray:
    out = da.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents['data']
        if content.variances is None:
            content.variances = content.values
    if sample_position is not None:
        out.coords['sample_position'] = sample_position
    out.coords['source_position'] = source_position
    out.coords['gravity'] = gravity_vector()
    return out


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def _preprocess_data(
    da: sc.DataArray,
    source_position: sc.Variable,
    sample_position: Optional[sc.Variable] = None,
) -> sc.DataArray:
    out = _patch_data(
        da=da, sample_position=sample_position, source_position=source_position
    )
    out = _convert_to_tof(out)
    return out


def _merge_events(a, b):
    # Note: the concatenate operation will check that all coordinates are the same.
    return a.bins.concatenate(b)


def _merge_runs(
    data_groups: Mapping[Filename[ScatteringRunType], sc.DataGroup],
    name: Union[
        NeXusDetectorName, NeXusMonitorName[Incident], NeXusMonitorName[Transmission]
    ],
) -> sc.DataGroup:
    """
    Merge events from multiple runs into a single run.
    """
    # TODO: we need some additional checks that the data is compatible. For example,
    # the sample and the source positions should be the same for all runs. Also, the
    # detector geometry (pixel_shapes, lab transform) should be the same for all runs.
    out = next(iter(data_groups.values())).copy(deep=False)
    data_arrays = []
    for dg in data_groups.values():
        events = dg[f'{name}_events']
        if 'event_time_zero' in events.dims:
            events = events.bins.concat('event_time_zero')
        data_arrays.append(events)
    out[f'{name}_events'] = reduce(_merge_events, data_arrays)
    return out


def merge_sample_runs(
    detectors: sciline.Series[Filename[SampleRun], LoadedSingleFileDetector[SampleRun]],
    detector_name: NeXusDetectorName,
) -> LoadedDetector[SampleRun]:
    """
    Merge detector events from multiple sample runs into a single sample run.
    """
    return LoadedDetector[SampleRun](
        _merge_runs(data_groups=detectors, name=detector_name)
    )


def merge_background_runs(
    detectors: sciline.Series[
        Filename[BackgroundRun], LoadedSingleFileDetector[BackgroundRun]
    ],
    detector_name: NeXusDetectorName,
) -> LoadedDetector[BackgroundRun]:
    """
    Merge detector events from multiple background runs into a single background run.
    """
    return LoadedDetector[BackgroundRun](
        _merge_runs(data_groups=detectors, name=detector_name)
    )


def merge_sample_monitor_runs(
    monitors: sciline.Series[
        Filename[SampleRun], LoadedSingleFileMonitor[SampleRun, MonitorType]
    ],
    monitor_name: NeXusMonitorName[MonitorType],
) -> LoadedMonitor[SampleRun, MonitorType]:
    """
    Merge monitor events from multiple sample runs into a single sample run.
    """
    return LoadedMonitor[SampleRun, MonitorType](
        _merge_runs(data_groups=monitors, name=monitor_name)
    )


def merge_background_monitor_runs(
    monitors: sciline.Series[
        Filename[BackgroundRun], LoadedSingleFileMonitor[BackgroundRun, MonitorType]
    ],
    monitor_name: NeXusMonitorName[MonitorType],
) -> LoadedMonitor[BackgroundRun, MonitorType]:
    """
    Merge monitor events from multiple background runs into a single sample run.
    """
    return LoadedMonitor[BackgroundRun, MonitorType](
        _merge_runs(data_groups=monitors, name=monitor_name)
    )


def _load_source_and_sample_positions(
    instrument: snx.Group,
    source_name: NeXusSourceName,
    sample_name: Optional[NeXusSampleName],
) -> tuple[sc.Variable, sc.Variable]:
    source_position = snx.compute_positions(instrument[source_name][()])['position']
    if sample_name is None:
        sample_position = sc.vector(value=[0, 0, 0], unit='m')
    else:
        sample_position = snx.compute_positions(instrument[sample_name][()])['position']
    return source_position, sample_position


def _load_events(
    filename: FilePath[Filename[RunType]],
    data_name: Union[NeXusDetectorName, NeXusMonitorName[MonitorType]],
    transform_path: TransformationPath,
    source_name: NeXusSourceName,
    sample_name: Optional[NeXusSampleName],
) -> sc.DataGroup:
    with snx.File(filename) as f:
        instrument = f['entry'][NEXUS_INSTRUMENT_PATH]
        dg = instrument[data_name][()]
        source_position, sample_position = _load_source_and_sample_positions(
            instrument, source_name, sample_name
        )
    dg = snx.compute_positions(dg, store_transform=transform_path)

    events = _preprocess_data(
        dg[f'{data_name}_events'],
        sample_position=sample_position,
        source_position=source_position,
    )
    if data_name in DETECTOR_BANK_RESHAPING:
        events = DETECTOR_BANK_RESHAPING[data_name](events)

    dg[f'{data_name}_events'] = events
    return dg


def load_nexus_sample(file_path: FilePath[Filename[RunType]]) -> RawSample[RunType]:
    return RawSample[RunType](nexus.load_sample(file_path))


def dummy_load_sample(file_path: FilePath[Filename[RunType]]) -> RawSample[RunType]:
    return RawSample[RunType](
        sc.DataGroup({'position': sc.vector(value=[0, 0, 0], unit='m')})
    )


def load_nexus_source(file_path: FilePath[Filename[RunType]]) -> RawSource[RunType]:
    return RawSource[RunType](nexus.load_source(file_path))


def load_nexus_detector(
    file_path: FilePath[Filename[RunType]],
    detector_name: NeXusDetectorName,
    raw_source: RawSource[RunType],
    raw_sample: RawSample[RunType],
) -> LoadedSingleFileDetector[RunType]:
    out = nexus.load_detector(file_path=file_path, detector_name=detector_name)
    # Note here we specify the name instead of using
    # ess.reduce.nexus.extract_detector_data because we need the name to put the
    # events back into the original data group.
    key = f'{detector_name}_events'
    events = _preprocess_data(
        out[key],
        sample_position=raw_sample['position'],
        source_position=raw_source['position'],
    )
    if detector_name in DETECTOR_BANK_RESHAPING:
        events = DETECTOR_BANK_RESHAPING[detector_name](events)
    out[key] = events
    return LoadedSingleFileDetector[RunType](out)


def load_nexus_monitor(
    file_path: FilePath[Filename[RunType]],
    monitor_name: NeXusMonitorName[MonitorType],
    raw_source: RawSource[RunType],
) -> LoadedSingleFileMonitor[RunType, MonitorType]:
    out = nexus.load_monitor(file_path=file_path, monitor_name=monitor_name)
    key = f'{monitor_name}_events'
    out[key] = _preprocess_data(out[key], source_position=raw_source['position'])
    return LoadedSingleFileMonitor[RunType, MonitorType](out)


def to_detector(
    data: LoadedSingleFileDetector[RunType],
) -> LoadedDetector[RunType]:
    """Dummy provider to convert a single-file detector to a combined detector."""
    return LoadedDetector[RunType](data)


def to_monitor(
    data: LoadedSingleFileMonitor[RunType, MonitorType],
) -> LoadedMonitor[RunType, MonitorType]:
    """Dummy provider to convert a single-file monitor to a combined monitor."""
    return LoadedMonitor[RunType, MonitorType](data)


def to_path(filename: FilenameType, path: DataFolder) -> FilePath[FilenameType]:
    return FilePath[FilenameType](f'{path}/{filename}')


providers = (
    load_nexus_detector,
    load_nexus_monitor,
    load_nexus_sample,
    load_nexus_source,
    to_detector,
    to_monitor,
    to_path,
)
"""Providers for loading single files."""
event_merging_providers = (
    merge_sample_runs,
    merge_background_runs,
    merge_sample_monitor_runs,
    merge_background_monitor_runs,
)
"""Providers to merge events from multiples files."""
