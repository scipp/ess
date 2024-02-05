# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""
from collections.abc import Iterable
from functools import reduce
from typing import Optional

import scipp as sc
import scippnexus as snx

from ..common import gravity_vector
from ..types import (
    FileList,
    Incident,
    LoadedFileContents,
    NeXusDetectorName,
    NeXusMonitorName,
    NeXusSampleName,
    NeXusSourceName,
    RunType,
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


def _merge_runs(data_groups: Iterable[sc.DataGroup], entries: Iterable[str]):
    # TODO: we need some additional checks that the data is compatible. For example,
    # the sample and the source positions should be the same for all runs. Also, the
    # detector geometry (pixel_shapes, lab transform) should be the same for all runs.
    out = data_groups[0].copy(deep=False)
    for name in entries:
        data_arrays = []
        for dg in data_groups:
            events = dg[NEXUS_INSTRUMENT_PATH][name][f'{name}_events']
            if 'event_time_zero' in events.dims:
                events = events.bins.concat('event_time_zero')
            data_arrays.append(events)
        out[NEXUS_INSTRUMENT_PATH][name][f'{name}_events'] = reduce(
            _merge_events, data_arrays
        )
    return out


def load_nexus(
    filelist: FileList[RunType],
    detector_name: NeXusDetectorName,
    incident_monitor_name: NeXusMonitorName[Incident],
    transmission_monitor_name: NeXusMonitorName[Transmission],
    transform_path: TransformationPath,
    source_name: NeXusSourceName,
    sample_name: Optional[NeXusSampleName],
) -> LoadedFileContents[RunType]:
    from .data import get_path

    data_entries = (detector_name, incident_monitor_name, transmission_monitor_name)

    data_groups = []
    for filename in filelist:
        with snx.File(get_path(filename)) as f:
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

        data_groups.append(dg)

    out = _merge_runs(data_groups=data_groups, entries=data_entries)
    return LoadedFileContents[RunType](out)


providers = (load_nexus,)
