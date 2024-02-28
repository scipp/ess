# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""
from functools import reduce
from typing import Optional

import sciline
import scipp as sc
import scippnexus as snx

from ..common import gravity_vector
from ..types import (
    BackgroundRun,
    DataFolder,
    Filename,
    FilenameType,
    FilePath,
    Incident,
    LoadedFileContents,
    LoadedSingleFileContents,
    NeXusDetectorName,
    NeXusMonitorName,
    NeXusSampleName,
    NeXusSourceName,
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
    da: sc.DataArray, sample_position: sc.Variable, source_position: sc.Variable
) -> sc.DataArray:
    out = da.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents['data']
        if content.variances is None:
            content.variances = content.values
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
    da: sc.DataArray, sample_position: sc.Variable, source_position: sc.Variable
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
    data_groups: sciline.Series[
        Filename[ScatteringRunType], LoadedSingleFileContents[ScatteringRunType]
    ],
    detector_name: NeXusDetectorName,
    incident_monitor_name: NeXusMonitorName[Incident],
    transmission_monitor_name: NeXusMonitorName[Transmission],
) -> LoadedFileContents[ScatteringRunType]:
    """
    Merge detector and monitor events from multiple runs into a single run.
    """
    data_entries = (detector_name, incident_monitor_name, transmission_monitor_name)
    # TODO: we need some additional checks that the data is compatible. For example,
    # the sample and the source positions should be the same for all runs. Also, the
    # detector geometry (pixel_shapes, lab transform) should be the same for all runs.
    out = list(data_groups.values())[0].copy(deep=False)
    for name in data_entries:
        data_arrays = []
        for dg in data_groups.values():
            events = dg[NEXUS_INSTRUMENT_PATH][name][f'{name}_events']
            if 'event_time_zero' in events.dims:
                events = events.bins.concat('event_time_zero')
            data_arrays.append(events)
        out[NEXUS_INSTRUMENT_PATH][name][f'{name}_events'] = reduce(
            _merge_events, data_arrays
        )
    return out


def merge_sample_runs(
    data_groups: sciline.Series[
        Filename[SampleRun], LoadedSingleFileContents[SampleRun]
    ],
    detector_name: NeXusDetectorName,
    incident_monitor_name: NeXusMonitorName[Incident],
    transmission_monitor_name: NeXusMonitorName[Transmission],
) -> LoadedFileContents[SampleRun]:
    """
    Merge detector and monitor events from multiple sample runs into a single sample
    run.
    """

    out = _merge_runs(
        data_groups=data_groups,
        detector_name=detector_name,
        incident_monitor_name=incident_monitor_name,
        transmission_monitor_name=transmission_monitor_name,
    )
    return LoadedFileContents[SampleRun](out)


def merge_background_runs(
    data_groups: sciline.Series[
        Filename[BackgroundRun], LoadedSingleFileContents[BackgroundRun]
    ],
    detector_name: NeXusDetectorName,
    incident_monitor_name: NeXusMonitorName[Incident],
    transmission_monitor_name: NeXusMonitorName[Transmission],
) -> LoadedFileContents[BackgroundRun]:
    """
    Merge detector and monitor events from multiple background runs into a single
    background run.
    """

    out = _merge_runs(
        data_groups=data_groups,
        detector_name=detector_name,
        incident_monitor_name=incident_monitor_name,
        transmission_monitor_name=transmission_monitor_name,
    )
    return LoadedFileContents[BackgroundRun](out)


def load_nexus(
    filename: FilePath[Filename[RunType]],
    detector_name: NeXusDetectorName,
    incident_monitor_name: NeXusMonitorName[Incident],
    transmission_monitor_name: NeXusMonitorName[Transmission],
    transform_path: TransformationPath,
    source_name: NeXusSourceName,
    sample_name: Optional[NeXusSampleName],
) -> LoadedSingleFileContents[RunType]:
    data_entries = (detector_name, incident_monitor_name, transmission_monitor_name)

    with snx.File(filename) as f:
        dg = f['entry'][()]
    dg = snx.compute_positions(dg, store_transform=transform_path)

    if sample_name is None:
        sample_position = sc.vector(value=[0, 0, 0], unit='m')
    else:
        sample_position = dg[NEXUS_INSTRUMENT_PATH][sample_name]['position']
    source_position = dg[NEXUS_INSTRUMENT_PATH][source_name]['position']

    for name in data_entries:
        data = _preprocess_data(
            dg[NEXUS_INSTRUMENT_PATH][name][f'{name}_events'],
            sample_position=sample_position,
            source_position=source_position,
        )
        if name in DETECTOR_BANK_RESHAPING:
            data = DETECTOR_BANK_RESHAPING[name](data)

        dg[NEXUS_INSTRUMENT_PATH][name][f'{name}_events'] = data

    return LoadedSingleFileContents[RunType](dg)


def to_file_contents(
    data: LoadedSingleFileContents[RunType],
) -> LoadedFileContents[RunType]:
    """Dummy provider to convert single-file contents to file contents."""
    return LoadedFileContents[RunType](data)


def to_path(filename: FilenameType, path: DataFolder) -> FilePath[FilenameType]:
    return f'{path}/{filename}'


providers = (load_nexus, to_file_contents, to_path)
