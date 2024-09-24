# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""

import sciline
import scipp as sc

from ess.reduce.nexus.generic_workflow import GenericNeXusWorkflow
from ess.sans import providers as sans_providers
from ess.sans.io import read_xml_detector_masking

from ..sans.types import (
    CorrectForGravity,
    DetectorBankSizes,
    DetectorData,
    DetectorPixelShape,
    DimsToKeep,
    DirectBeam,
    DirectBeamFilename,
    Incident,
    LabFrameTransform,
    MonitorData,
    MonitorType,
    NeXusDetector,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelShapePath,
    RunType,
    ScatteringRunType,
    TofData,
    TofMonitor,
    TransformationPath,
    Transmission,
    WavelengthBands,
    WavelengthMask,
)

DETECTOR_BANK_SIZES = {
    'larmor_detector': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': 512}
}


def default_parameters() -> dict:
    return {
        CorrectForGravity: False,
        DetectorBankSizes: DETECTOR_BANK_SIZES,
        DimsToKeep: (),
        NeXusMonitorName[Incident]: 'monitor_1',
        NeXusMonitorName[Transmission]: 'monitor_2',
        TransformationPath: 'transform',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
        WavelengthMask: None,
        WavelengthBands: None,
    }


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def data_to_tof(
    da: DetectorData[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    return TofData[ScatteringRunType](_convert_to_tof(da))


def monitor_to_tof(
    da: MonitorData[RunType, MonitorType],
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


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    """Load direct beam from file."""
    return DirectBeam(sc.io.load_hdf5(filename))


loki_providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    data_to_tof,
    load_direct_beam,
    monitor_to_tof,
)


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
    workflow = GenericNeXusWorkflow()
    for provider in sans_providers + loki_providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    workflow.insert(read_xml_detector_masking)
    return workflow
