# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""
from typing import Optional

import scipp as sc
from ess.reduce import nexus

from ..sans.common import gravity_vector
from ..sans.types import (
    ConfiguredReducibleDataData,
    ConfiguredReducibleMonitor,
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedNeXusDetector,
    LoadedNeXusMonitor,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    PixelShapePath,
    RawData,
    RawMonitor,
    RawSample,
    RawSource,
    RunType,
    SamplePosition,
    ScatteringRunType,
    SourcePosition,
    TofData,
    TofMonitor,
    TransformationPath,
    Transmission,
)

default_parameters = {
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    TransformationPath: 'transform',
    PixelShapePath: 'pixel_shape',
}


DETECTOR_BANK_RESHAPING = {
    'larmor_detector': lambda x: x.fold(
        dim='detector_number', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
    )
}


def get_source_position(
    raw_source: RawSource[RunType],
) -> SourcePosition[RunType]:
    return SourcePosition[RunType](raw_source['position'])


def get_sample_position(
    raw_sample: RawSample[RunType],
) -> SamplePosition[RunType]:
    return SamplePosition[RunType](raw_sample['position'])


def get_detector_data(
    detector: LoadedNeXusDetector[ScatteringRunType],
    detector_name: NeXusDetectorName,
) -> RawData[ScatteringRunType]:
    da = nexus.extract_detector_data(detector)
    if detector_name in DETECTOR_BANK_RESHAPING:
        da = DETECTOR_BANK_RESHAPING[detector_name](da)
    return RawData[ScatteringRunType](da)


def get_monitor_data(
    monitor: LoadedNeXusMonitor[RunType, MonitorType],
) -> RawMonitor[RunType, MonitorType]:
    out = nexus.extract_monitor_data(monitor).copy(deep=False)
    out.coords['position'] = monitor['position']
    return RawMonitor[RunType, MonitorType](out)


def _add_variances_and_coordinates(
    da: sc.DataArray,
    source_position: sc.Variable,
    sample_position: Optional[sc.Variable] = None,
) -> sc.DataArray:
    out = da.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents['data']
        if content.variances is None:
            content.variances = content.values
    # Sample position is not needed in the case of a monitor.
    if sample_position is not None:
        out.coords['sample_position'] = sample_position
    out.coords['source_position'] = source_position
    out.coords['gravity'] = gravity_vector()
    return out


def patch_detector_data(
    detector_data: RawData[ScatteringRunType],
    source_position: SourcePosition[ScatteringRunType],
    sample_position: SamplePosition[ScatteringRunType],
) -> ConfiguredReducibleDataData[ScatteringRunType]:
    return ConfiguredReducibleDataData[ScatteringRunType](
        _add_variances_and_coordinates(
            da=detector_data,
            source_position=source_position,
            sample_position=sample_position,
        )
    )


def patch_monitor_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> ConfiguredReducibleMonitor[RunType, MonitorType]:
    return ConfiguredReducibleMonitor[RunType, MonitorType](
        _add_variances_and_coordinates(da=monitor_data, source_position=source_position)
    )


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def data_to_tof(
    da: ConfiguredReducibleDataData[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    return TofData[ScatteringRunType](_convert_to_tof(da))


def monitor_to_tof(
    da: ConfiguredReducibleMonitor[RunType, MonitorType],
) -> TofMonitor[RunType, MonitorType]:
    return TofMonitor[RunType, MonitorType](_convert_to_tof(da))


def detector_pixel_shape(
    detector: LoadedNeXusDetector[ScatteringRunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[ScatteringRunType]:
    return DetectorPixelShape[ScatteringRunType](detector[pixel_shape_path])


def detector_lab_frame_transform(
    detector: LoadedNeXusDetector[ScatteringRunType],
    transform_path: TransformationPath,
) -> LabFrameTransform[ScatteringRunType]:
    return LabFrameTransform[ScatteringRunType](detector[transform_path])


providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    get_detector_data,
    get_monitor_data,
    get_sample_position,
    get_source_position,
    patch_detector_data,
    patch_monitor_data,
    data_to_tof,
    monitor_to_tof,
)
