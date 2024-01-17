# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""


from ..types import (
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedFileContents,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    NeXusSourceName,
    RawData,
    RawMonitor,
    RunType,
    TransformationPath,
    Transmission,
)

NEXUS_INSTRUMENT_PATH = 'instrument'

default_parameters = {
    NeXusDetectorName: 'larmor_detector',
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    NeXusSourceName: 'source',
    # TODO: sample is not in the files, so by not adding the name here, we use the
    # default value of [0, 0, 0] when loading the sample position.
    TransformationPath: 'esssans_loki_transformation',
}


def get_detector_data(
    dg: LoadedFileContents[RunType], detector_name: NeXusDetectorName
) -> RawData[RunType]:
    da = dg[NEXUS_INSTRUMENT_PATH][detector_name][f'{detector_name}_events']
    return RawData[RunType](da)


def get_monitor_data(
    dg: LoadedFileContents[RunType], monitor_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    mon_dg = dg[NEXUS_INSTRUMENT_PATH][monitor_name]
    out = mon_dg[f'{monitor_name}_events']
    out.coords['position'] = mon_dg['position']
    return RawMonitor[RunType, MonitorType](out)


def detector_pixel_shape(
    dg: LoadedFileContents[RunType], detector_name: NeXusDetectorName
) -> DetectorPixelShape[RunType]:
    return DetectorPixelShape[RunType](
        dg[NEXUS_INSTRUMENT_PATH][detector_name]['pixel_shape']
    )


def detector_lab_frame_transform(
    dg: LoadedFileContents[RunType],
    detector_name: NeXusDetectorName,
    transform_path: TransformationPath,
) -> LabFrameTransform[RunType]:
    return LabFrameTransform[RunType](
        dg[NEXUS_INSTRUMENT_PATH][detector_name][transform_path]
    )


providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    get_detector_data,
    get_monitor_data,
)
