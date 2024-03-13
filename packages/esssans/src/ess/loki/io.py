# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""
# from collections.abc import Mapping
# from itertools import groupby
# from functools import reduce
# from typing import Optional, Union

# import sciline
import scipp as sc
from ess.reduce import nexus

# from ..sans.common import gravity_vector
from ..sans.types import (
    # BackgroundRun,
    DataFolder,
    # DetectorPixelShape,
    Filename,
    FilenameType,
    FilePath,
    # Incident,
    # LabFrameTransform,
    LoadedNeXusDetector,
    LoadedNeXusMonitor,
    # LoadedSingleFileDetector,
    # LoadedSingleFileMonitor,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    # PixelShapePath,
    RawSample,
    RawSource,
    RunType,
    # SampleRun,
    # ScatteringRunType,
    # TransformationPath,
    # Transmission,
)

# from .general import default_parameters as params

# DETECTOR_BANK_RESHAPING = {
#     params[NeXusDetectorName]: lambda x: x.fold(
#         dim='detector_number', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
#     )
# }


# def add_variances_and_coordinates(
#     da: sc.DataArray,
#     source_position: sc.Variable,
#     sample_position: Optional[sc.Variable] = None,
# ) -> sc.DataArray:
#     out = da.copy(deep=False)
#     if out.bins is not None:
#         content = out.bins.constituents['data']
#         if content.variances is None:
#             content.variances = content.values
#     # Sample position is not needed in the case of a monitor.
#     if sample_position is not None:
#         out.coords['sample_position'] = sample_position
#     out.coords['source_position'] = source_position
#     out.coords['gravity'] = gravity_vector()
#     return out


# def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
#     da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
#     if 'event_time_zero' in da.dims:
#         da = da.bins.concat('event_time_zero')
#     return da


# def _preprocess_data(
#     da: sc.DataArray,
#     source_position: sc.Variable,
#     sample_position: Optional[sc.Variable] = None,
# ) -> sc.DataArray:
#     out = _patch_data(
#         da=da, sample_position=sample_position, source_position=source_position
#     )
#     out = _convert_to_tof(out)
#     return out


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
    # raw_source: RawSource[RunType],
    # raw_sample: RawSample[RunType],
) -> LoadedNeXusDetector[RunType]:
    # out = nexus.load_detector(file_path=file_path, detector_name=detector_name)
    # # Note here we specify the name instead of using
    # # ess.reduce.nexus.extract_detector_data because we need the name to put the
    # # events back into the original data group.
    # key = f'{detector_name}_events'
    # events = _preprocess_data(
    #     out[key],
    #     sample_position=raw_sample['position'],
    #     source_position=raw_source['position'],
    # )
    # if detector_name in DETECTOR_BANK_RESHAPING:
    #     events = DETECTOR_BANK_RESHAPING[detector_name](events)
    # out[key] = events
    return LoadedNeXusDetector[RunType](
        nexus.load_detector(file_path=file_path, detector_name=detector_name)
    )


def load_nexus_monitor(
    file_path: FilePath[Filename[RunType]],
    monitor_name: NeXusMonitorName[MonitorType],
    # raw_source: RawSource[RunType],
) -> LoadedNeXusMonitor[RunType, MonitorType]:
    # out = nexus.load_monitor(file_path=file_path, monitor_name=monitor_name)
    # key = f'{monitor_name}_events'
    # out[key] = _preprocess_data(out[key], source_position=raw_source['position'])
    return LoadedNeXusMonitor[RunType, MonitorType](
        nexus.load_monitor(file_path=file_path, monitor_name=monitor_name)
    )


# def to_detector(
#     data: LoadedSingleFileDetector[RunType],
# ) -> LoadedDetector[RunType]:
#     """Dummy provider to convert a single-file detector to a combined detector."""
#     return LoadedDetector[RunType](data)


# def to_monitor(
#     data: LoadedSingleFileMonitor[RunType, MonitorType],
# ) -> LoadedMonitor[RunType, MonitorType]:
#     """Dummy provider to convert a single-file monitor to a combined monitor."""
#     return LoadedMonitor[RunType, MonitorType](data)


def to_path(filename: FilenameType, path: DataFolder) -> FilePath[FilenameType]:
    return FilePath[FilenameType](f'{path}/{filename}')


providers = (
    load_nexus_detector,
    load_nexus_monitor,
    load_nexus_sample,
    load_nexus_source,
    # to_detector,
    # to_monitor,
    to_path,
)
"""Providers for loading single files."""
