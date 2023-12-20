# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""

from collections.abc import Iterable
from functools import partial, reduce
from pathlib import Path
from typing import Optional, Union

import sciline
import scipp as sc
import scippnexus as snx

from ..common import gravity_vector
from ..types import (
    Filelist,
    Incident,
    LoadedFileContents,
    # LoadedMonitorContents,
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
    Transmission,
    UnmergedPatchedData,
    UnmergedPatchedMonitor,
    UnmergedRawData,
    UnmergedRawMonitor,
)


def _patch_data(
    da: sc.DataArray,
    sample_position: sc.Variable,
    source_position: sc.Variable,
) -> sc.DataArray:
    out = da.copy(deep=False)
    out.coords['sample_position'] = sample_position
    out.coords['source_position'] = source_position
    out.coords['gravity'] = gravity_vector()
    return out


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da.bin(tof=1)


def _preprocess_data(
    da: sc.DataArray,
    sample_position: sc.Variable,
    source_position: sc.Variable,
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
    data_groups: Iterable[sc.DataGroup],
    instrument_path: NexusInstrumentPath,
    entries: Iterable[str],
):
    # TODO: we need some additional checks that the data is compatible. For example,
    # the sample and the source positions should be the same for all runs. Also, the
    # detector geometry (pixel_shapes, lab transform) should be the same for all runs.
    out = data_groups[0].copy(deep=False)
    for name in entries:
        data_arrays = [
            dg[instrument_path][name][f'{name}_events'] for dg in data_groups
        ]
        out[instrument_path][name][f'{name}_events'] = reduce(
            _merge_events, data_arrays
        )
    return out


def load_nexus(
    filelist: Filelist[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
    incident_monitor_name: NeXusMonitorName[Incident],
    transmission_monitor_name: NeXusMonitorName[Transmission],
    transform_path: TransformationChainPath,
    source_name: NexusSourceName,
    sample_name: Optional[NexusSampleName],
) -> LoadedFileContents[RunType]:
    from .data import get_path

    data_groups = []
    for filename in filelist:
        with snx.File(get_path(filename)) as f:
            dg = f['entry'][()]
        dg = snx.compute_positions(dg, store_transform=transform_path)
        data_groups.append(dg)

    data_entries = (detector_name, incident_monitor_name, transmission_monitor_name)
    out = _merge_runs(
        data_groups=data_groups, instrument_path=instrument_path, entries=data_entries
    )

    if sample_name is None:
        sample_position = sc.vector(value=[0, 0, 0], unit='m')
    else:
        sample_position = out[instrument_path][sample_name]['position']
    source_position = out[instrument_path][source_name]['position']

    for name in data_entries:
        # da = out[instrument_path][name][f'{name}_events']
        out[instrument_path][name][f'{name}_events'] = _preprocess_data(
            out[instrument_path][name][f'{name}_events'],
            sample_position=sample_position,
            source_position=source_position,
        )

    return LoadedFileContents[RunType](out)


# def load_data_run(
#     filename: Filename[RunType],
#     instrument_path: NexusInstrumentPath,
#     detector_name: NexusDetectorName,
#     transform_path: TransformationChainPath,
# ) -> LoadedDetectorContents[RunType]:
#     entry = Path(instrument_path) / Path(detector_name)
#     dg = _load_file_entry(filename=filename, entry=entry, transform_path=transform_path)
#     return LoadedDetectorContents[RunType](dg)


def get_detector_data(
    dg: LoadedFileContents[RunType],
    detector_name: NexusDetectorName,
    instrument_path: NexusInstrumentPath,
) -> RawData[RunType]:
    da = dg[instrument_path][detector_name][f'{detector_name}_events']
    return RawData[RunType](da)


# def _merge_events(a, b):
#     # Note: the concatenate operation will check that all coordinates are the same.
#     return a.bins.concatenate(b)


# def merge_detector_events(
#     runs: sciline.Series[RunID[RunType], UnmergedPatchedData[RunType]]
# ) -> RawData[RunType]:
#     return RawData[RunType](reduce(_merge_events, runs.values()))


# def merge_monitor_events(
#     runs: sciline.Series[RunID[RunType], UnmergedPatchedMonitor[RunType, MonitorType]]
# ) -> RawMonitor[RunType, MonitorType]:
#     return RawMonitor[RunType, MonitorType](reduce(_merge_events, runs.values()))


# def load_monitor(
#     filename: Filename[RunType],
#     instrument_path: NexusInstrumentPath,
#     monitor_name: NeXusMonitorName[MonitorType],
#     transform_path: TransformationChainPath,
# ) -> LoadedMonitorContents[RunType, MonitorType]:
#     entry = Path(instrument_path) / Path(monitor_name)
#     dg = _load_file_entry(filename=filename, entry=entry, transform_path=transform_path)
#     return LoadedMonitorContents[RunType, MonitorType](dg)


def get_monitor_data(
    dg: LoadedFileContents[RunType],
    monitor_name: NeXusMonitorName[MonitorType],
    instrument_path: NexusInstrumentPath,
) -> RawMonitor[RunType, MonitorType]:
    mon_dg = dg[instrument_path][monitor_name]
    out = mon_dg[f'{monitor_name}_events']
    out.coords['position'] = mon_dg['position']
    return RawMonitor[RunType, MonitorType](out)


# def load_sample_position(
#     filename: Filename[RunType],
#     instrument_path: NexusInstrumentPath,
#     sample_name: Optional[NexusSampleName],
#     transform_path: TransformationChainPath,
# ) -> SamplePosition[RunType]:
#     # TODO: sample_name is optional for now because it is not found in all the files.
#     if sample_name is None:
#         out = sc.vector(value=[0, 0, 0], unit='m')
#     else:
#         entry = Path(instrument_path) / Path(sample_name)
#         dg = _load_file_entry(
#             filename=filename, entry=entry, transform_path=transform_path
#         )
#         out = SamplePosition[RunType](dg['position'])
#     return SamplePosition[RunType](out)


# def load_source_position(
#     filename: Filename[RunType],
#     instrument_path: NexusInstrumentPath,
#     source_name: NexusSourceName,
#     transform_path: TransformationChainPath,
# ) -> SourcePosition[RunType]:
#     entry = Path(instrument_path) / Path(source_name)
#     dg = _load_file_entry(filename=filename, entry=entry, transform_path=transform_path)
#     return SourcePosition[RunType](dg['position'])


providers = (
    get_detector_data,
    get_monitor_data,
    load_nexus,
    # get_detector_data,
    # get_monitor_data,
    # load_monitor,
    # load_data_run,
    # load_sample_position,
    # load_source_position,
    # merge_detector_events,
    # merge_monitor_events,
)
