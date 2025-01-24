# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the Loki workflow.
"""

import sciline
import scipp as sc
import scippnexus as snx

from ess import sans
from ess.reduce.workflow import register_workflow
from ess.sans.io import read_xml_detector_masking
from ess.sans.parameters import typical_outputs

from ..sans.types import (
    BackgroundRun,
    BeamCenter,
    DetectorBankSizes,
    DetectorData,
    DetectorPixelShape,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    MonitorData,
    MonitorType,
    NeXusComponent,
    NeXusDetectorName,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelMaskFilename,
    PixelShapePath,
    RunType,
    SampleRun,
    ScatteringRunType,
    TofData,
    TofMonitor,
    Transmission,
    TransmissionRun,
)

DETECTOR_BANK_SIZES = {
    'larmor_detector': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': 512}
}


def default_parameters() -> dict:
    return {
        DetectorBankSizes: DETECTOR_BANK_SIZES,
        NeXusMonitorName[Incident]: 'monitor_1',
        NeXusMonitorName[Transmission]: 'monitor_2',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
    }


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    event_time_offset = da.bins.coords['event_time_offset']
    da = da.bins.drop_coords('event_time_offset')
    da.bins.coords['tof'] = event_time_offset
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
    detector: NeXusComponent[snx.NXdetector, ScatteringRunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[ScatteringRunType]:
    return DetectorPixelShape[ScatteringRunType](detector[pixel_shape_path])


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    """Load direct beam from file."""
    return DirectBeam(sc.io.load_hdf5(filename))


loki_providers = (detector_pixel_shape, data_to_tof, load_direct_beam, monitor_to_tof)


@register_workflow
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
    workflow = sans.SansWorkflow()
    for provider in loki_providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    workflow.insert(read_xml_detector_masking)
    workflow[NeXusDetectorName] = 'larmor_detector'
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
def LokiAtLarmorTutorialWorkflow() -> sciline.Pipeline:
    from ess.loki import data

    workflow = LokiAtLarmorWorkflow()

    workflow[PixelMaskFilename] = data.loki_tutorial_mask_filenames()
    workflow[Filename[SampleRun]] = data.loki_tutorial_sample_run_60339()
    workflow[Filename[BackgroundRun]] = data.loki_tutorial_background_run_60393()
    workflow[Filename[TransmissionRun[SampleRun]]] = (
        data.loki_tutorial_sample_transmission_run()
    )
    workflow[Filename[TransmissionRun[BackgroundRun]]] = data.loki_tutorial_run_60392()
    workflow[Filename[EmptyBeamRun]] = data.loki_tutorial_run_60392()
    workflow[BeamCenter] = sc.vector(value=[-0.02914868, -0.01816138, 0.0], unit='m')
    return workflow
