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
    CalibratedMonitor,
    DataWithVariancesAndCoordinates,
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    # LoadedDetector,
    # LoadedMonitor,
    LoadedSingleFileDetector,
    LoadedSingleFileMonitor,
    LogicalDimsData,
    MonitorType,
    MonitorWithVariancesAndCoordinates,
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
    SingleFileDetectorData,
    SingleFileMonitorData,
    SourcePosition,
    TofData,
    TransformationPath,
    Transmission,
)

default_parameters = {
    NeXusDetectorName: 'larmor_detector',
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    TransformationPath: 'transform',
    PixelShapePath: 'pixel_shape',
}


DETECTOR_BANK_RESHAPING = {
    default_parameters[NeXusDetectorName]: lambda x: x.fold(
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
    detector: LoadedSingleFileDetector[ScatteringRunType],
) -> SingleFileDetectorData[ScatteringRunType]:
    return SingleFileDetectorData[ScatteringRunType](
        nexus.extract_detector_data(detector)
    )


def to_detector_data(
    data: SingleFileDetectorData[ScatteringRunType],
) -> RawData[ScatteringRunType]:
    """Dummy provider to convert a single-file detector to a combined detector."""
    return RawData[ScatteringRunType](data)


def get_monitor_data(
    monitor: LoadedSingleFileMonitor[RunType, MonitorType],
) -> SingleFileMonitorData[RunType, MonitorType]:
    out = nexus.extract_monitor_data(monitor).copy(deep=False)
    out.coords['position'] = monitor['position']
    return SingleFileMonitorData[RunType, MonitorType](out)


def to_monitor_data(
    data: SingleFileMonitorData[RunType, MonitorType],
) -> RawMonitor[RunType, MonitorType]:
    """Dummy provider to convert a single-file monitor to a combined monitor."""
    return RawMonitor[RunType, MonitorType](data)


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
) -> DataWithVariancesAndCoordinates[ScatteringRunType]:
    return DataWithVariancesAndCoordinates[ScatteringRunType](
        _add_variances_and_coordinates(
            da=detector_data,
            source_position=source_position,
            sample_position=sample_position,
        )
    )


def patch_monitor_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> MonitorWithVariancesAndCoordinates[RunType, MonitorType]:
    return MonitorWithVariancesAndCoordinates[RunType, MonitorType](
        _add_variances_and_coordinates(da=monitor_data, source_position=source_position)
    )


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def data_to_tof(
    da: DataWithVariancesAndCoordinates[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    return TofData[ScatteringRunType](_convert_to_tof(da))


def monitor_to_tof(
    da: MonitorWithVariancesAndCoordinates[RunType, MonitorType],
) -> CalibratedMonitor[RunType, MonitorType]:
    return CalibratedMonitor[RunType, MonitorType](_convert_to_tof(da))


def to_logical_dims(
    da: TofData[ScatteringRunType],
    detector_name: NeXusDetectorName,
) -> LogicalDimsData[ScatteringRunType]:
    if detector_name in DETECTOR_BANK_RESHAPING:
        da = DETECTOR_BANK_RESHAPING[detector_name](da)
    return LogicalDimsData[ScatteringRunType](da)


def detector_pixel_shape(
    detector: LoadedSingleFileDetector[ScatteringRunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[ScatteringRunType]:
    return DetectorPixelShape[ScatteringRunType](detector[pixel_shape_path])


def detector_lab_frame_transform(
    detector: LoadedSingleFileDetector[ScatteringRunType],
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
    to_detector_data,
    to_logical_dims,
    to_monitor_data,
    data_to_tof,
    monitor_to_tof,
)
