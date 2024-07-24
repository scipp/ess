# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""

import sciline
import scipp as sc
from ess.reduce import nexus
from ess.sans import providers as sans_providers

from ..sans.common import gravity_vector
from ..sans.types import (
    CorrectForGravity,
    DetectorEventData,
    DetectorPixelShape,
    DimsToKeep,
    Incident,
    LabFrameTransform,
    MonitorEventData,
    MonitorType,
    NeXusDetector,
    NeXusMonitor,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelShapePath,
    RawDetector,
    RawDetectorData,
    RawMonitor,
    RawMonitorData,
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
    WavelengthBands,
    WavelengthMask,
)
from .io import dummy_load_sample


def default_parameters() -> dict:
    return {
        CorrectForGravity: False,
        DimsToKeep: (),
        NeXusMonitorName[Incident]: 'monitor_1',
        NeXusMonitorName[Transmission]: 'monitor_2',
        TransformationPath: 'transform',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
        WavelengthMask: None,
        WavelengthBands: None,
    }


def LokiAtLarmorWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for Loki test at Larmor.

    This version of the Loki workflow:

    - Uses ISIS XML files to define masks.
    - Sets a dummy sample position [0,0,0] since files do not contain this information.

    Returns
    -------
    :
        Loki workflow as a sciline.Pipeline
    """
    from ess.isissans.io import read_xml_detector_masking

    from . import providers as loki_providers

    params = default_parameters()
    loki_providers = sans_providers + loki_providers
    workflow = sciline.Pipeline(providers=loki_providers, params=params)
    workflow.insert(read_xml_detector_masking)
    # No sample information in the Loki@Larmor files, so we use a dummy sample provider
    workflow.insert(dummy_load_sample)
    return workflow


DETECTOR_BANK_RESHAPING = {
    'larmor_detector': lambda x: x.fold(
        dim='detector_number', sizes={'layer': 4, 'tube': 32, 'straw': 7, 'pixel': 512}
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
    detector: NeXusDetector[ScatteringRunType],
) -> RawDetector[ScatteringRunType]:
    da = nexus.extract_detector_data(detector)
    if (reshape := DETECTOR_BANK_RESHAPING.get(detector['detector_name'])) is not None:
        da = reshape(da)
    return RawDetector[ScatteringRunType](da)


def get_monitor_data(
    monitor: NeXusMonitor[RunType, MonitorType],
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor[RunType, MonitorType](
        nexus.extract_monitor_data(monitor).assign_coords(position=monitor['position'])
    )


def _add_variances_and_coordinates(
    da: sc.DataArray,
    source_position: sc.Variable,
    sample_position: sc.Variable | None = None,
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


def assemble_detector_data(
    detector_data: RawDetector[ScatteringRunType],
    event_data: DetectorEventData[ScatteringRunType],
    source_position: SourcePosition[ScatteringRunType],
    sample_position: SamplePosition[ScatteringRunType],
) -> RawDetectorData[ScatteringRunType]:
    grouped = nexus.group_event_data(
        event_data=event_data, detector_number=detector_data.coords['detector_number']
    )
    detector_data.data = grouped.data
    return RawDetectorData[ScatteringRunType](
        _add_variances_and_coordinates(
            da=detector_data,
            source_position=source_position,
            sample_position=sample_position,
        )
    )


def assemble_monitor_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    event_data: MonitorEventData[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> RawMonitorData[RunType, MonitorType]:
    meta = monitor_data.drop_coords('event_time_zero')
    da = event_data.assign_coords(meta.coords).assign_masks(meta.masks)
    return RawMonitorData[RunType, MonitorType](
        _add_variances_and_coordinates(da=da, source_position=source_position)
    )


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def data_to_tof(
    da: RawDetectorData[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    return TofData[ScatteringRunType](_convert_to_tof(da))


def monitor_to_tof(
    da: RawMonitorData[RunType, MonitorType],
) -> TofMonitor[RunType, MonitorType]:
    return TofMonitor[RunType, MonitorType](_convert_to_tof(da))


def detector_pixel_shape(
    detector: NeXusDetector[ScatteringRunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[ScatteringRunType]:
    return DetectorPixelShape[ScatteringRunType](detector[pixel_shape_path])


def detector_lab_frame_transform(
    detector: NeXusDetector[ScatteringRunType],
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
    assemble_detector_data,
    assemble_monitor_data,
    data_to_tof,
    monitor_to_tof,
)
