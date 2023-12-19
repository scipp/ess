# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""
from functools import reduce
from pathlib import Path
from typing import Optional, Union

import sciline
import scipp as sc
import scippnexus as snx

from ..types import (
    Filename,
    LoadedDetectorContents,
    LoadedMonitorContents,
    MonitorType,
    NexusDetectorName,
    NexusInstrumentPath,
    NeXusMonitorName,
    NexusSampleName,
    NexusSourceName,
    RawData,
    RawMonitor,
    RunID,
    RunType,
    SamplePosition,
    SourcePosition,
    TransformationChainPath,
    UnmergedPatchedData,
    UnmergedPatchedMonitor,
    UnmergedRawData,
    UnmergedRawMonitor,
)


def _load_file_entry(
    filename: str, entry: Union[str, Path], transform_path: Union[str, Path]
) -> sc.DataArray:
    from .data import get_path

    with snx.File(get_path(filename)) as f:
        dg = f[str(entry)][()]
    dg = snx.compute_positions(dg, store_transform=transform_path)

    return dg


def load_data_run(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
    transform_path: TransformationChainPath,
) -> LoadedDetectorContents[RunType]:
    entry = Path(instrument_path) / Path(detector_name)
    dg = _load_file_entry(filename=filename, entry=entry, transform_path=transform_path)
    return LoadedDetectorContents[RunType](dg)


def get_detector_data(
    dg: LoadedDetectorContents[RunType],
    detector_name: NexusDetectorName,
) -> UnmergedRawData[RunType]:
    da = dg[f'{detector_name}_events']
    return UnmergedRawData[RunType](da)


def _merge_events(a, b):
    # Note: the concatenate operation will check that all coordinates are the same.
    return a.bins.concatenate(b)


def merge_detector_events(
    runs: sciline.Series[RunID[RunType], UnmergedPatchedData[RunType]]
) -> RawData[RunType]:
    return RawData[RunType](reduce(_merge_events, runs.values()))


def merge_monitor_events(
    runs: sciline.Series[RunID[RunType], UnmergedPatchedMonitor[RunType, MonitorType]]
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor[RunType, MonitorType](reduce(_merge_events, runs.values()))


def load_monitor(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    monitor_name: NeXusMonitorName[MonitorType],
    transform_path: TransformationChainPath,
) -> LoadedMonitorContents[RunType, MonitorType]:
    entry = Path(instrument_path) / Path(monitor_name)
    dg = _load_file_entry(filename=filename, entry=entry, transform_path=transform_path)
    return LoadedMonitorContents[RunType, MonitorType](dg)


def get_monitor_data(
    dg: LoadedMonitorContents[RunType, MonitorType],
    monitor_name: NeXusMonitorName[MonitorType],
) -> UnmergedRawMonitor[RunType, MonitorType]:
    out = dg[f'{monitor_name}_events']
    out.coords['position'] = dg['position']
    return UnmergedRawMonitor[RunType, MonitorType](out)


def load_sample_position(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    sample_name: Optional[NexusSampleName],
    transform_path: TransformationChainPath,
) -> SamplePosition[RunType]:
    # TODO: sample_name is optional for now because it is not found in all the files.
    if sample_name is None:
        out = sc.vector(value=[0, 0, 0], unit='m')
    else:
        entry = Path(instrument_path) / Path(sample_name)
        dg = _load_file_entry(
            filename=filename, entry=entry, transform_path=transform_path
        )
        out = SamplePosition[RunType](dg['position'])
    return SamplePosition[RunType](out)


def load_source_position(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    source_name: NexusSourceName,
    transform_path: TransformationChainPath,
) -> SourcePosition[RunType]:
    entry = Path(instrument_path) / Path(source_name)
    dg = _load_file_entry(filename=filename, entry=entry, transform_path=transform_path)
    return SourcePosition[RunType](dg['position'])


providers = (
    get_detector_data,
    get_monitor_data,
    load_monitor,
    load_data_run,
    load_sample_position,
    load_source_position,
    merge_detector_events,
    merge_monitor_events,
)
