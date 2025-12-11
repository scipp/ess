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
    DetectorPixelShape,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    MonitorType,
    NeXusComponent,
    NeXusDetectorName,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelMaskFilename,
    PixelShapePath,
    RawDetector,
    RawMonitor,
    RunType,
    SampleRun,
    TofDetector,
    TofMonitor,
    Transmission,
    TransmissionRun,
)

DETECTOR_BANK_SIZES = {
    'larmor_detector': {'layer': 4, 'tube': -1, 'straw': 7, 'pixel': 512},
    'loki_detector_0': {'layer': 4, 'tube': 56, 'straw': 7, 'pixel': -1},
    'loki_detector_1': {'layer': 4, 'tube': 16, 'straw': 7, 'pixel': -1},
    'loki_detector_2': {'layer': 4, 'tube': 12, 'straw': 7, 'pixel': -1},
    'loki_detector_3': {'layer': 4, 'tube': 16, 'straw': 7, 'pixel': -1},
    'loki_detector_4': {'layer': 4, 'tube': 12, 'straw': 7, 'pixel': -1},
    'loki_detector_5': {'layer': 4, 'tube': 28, 'straw': 7, 'pixel': -1},
    'loki_detector_6': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': -1},
    'loki_detector_7': {'layer': 4, 'tube': 20, 'straw': 7, 'pixel': -1},
    'loki_detector_8': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': -1},
}


def larmor_default_parameters() -> dict:
    return {
        DetectorBankSizes: DETECTOR_BANK_SIZES,
        NeXusMonitorName[Incident]: 'monitor_1',
        NeXusMonitorName[Transmission]: 'monitor_2',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
    }


def loki_default_parameters() -> dict:
    return {
        DetectorBankSizes: DETECTOR_BANK_SIZES,
        NeXusMonitorName[Incident]: 'beam_monitor_1',
        NeXusMonitorName[Transmission]: 'beam_monitor_3',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
    }


def _larmor_convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    event_time_offset = da.bins.coords['event_time_offset']
    da = da.bins.drop_coords('event_time_offset')
    da.bins.coords['tof'] = event_time_offset
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da


def larmor_data_to_tof(da: RawDetector[RunType]) -> TofDetector[RunType]:
    """
    Compute time-of-flight coordinate for Loki detector data at Larmor.
    This is different from the standard conversion from the GenericTofWorkflow because
    the detector test was conducted as ISIS where the pulse has a different time
    structure.
    The conversion here is much simpler: the event_time_offset coordinate is directly
    renamed as time-of-flight.
    """
    return TofDetector[RunType](_larmor_convert_to_tof(da))


def larmor_monitor_to_tof(
    da: RawMonitor[RunType, MonitorType],
) -> TofMonitor[RunType, MonitorType]:
    """
    Compute time-of-flight coordinate for Loki monitor data at Larmor.
    This is different from the standard conversion from the GenericTofWorkflow because
    the detector test was conducted as ISIS where the pulse has a different time
    structure.
    The conversion here is much simpler: the event_time_offset coordinate is directly
    renamed as time-of-flight.
    """
    return TofMonitor[RunType, MonitorType](_larmor_convert_to_tof(da))


def detector_pixel_shape(
    detector: NeXusComponent[snx.NXdetector, RunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[RunType]:
    return DetectorPixelShape[RunType](detector[pixel_shape_path])


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    """Load direct beam from file."""
    return DirectBeam(sc.io.load_hdf5(filename))


loki_providers = (detector_pixel_shape, load_direct_beam)


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
    for key, param in larmor_default_parameters().items():
        workflow[key] = param
    workflow.insert(larmor_data_to_tof)
    workflow.insert(larmor_monitor_to_tof)
    workflow.insert(read_xml_detector_masking)
    workflow[NeXusDetectorName] = 'larmor_detector'
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
def LokiAtLarmorTutorialWorkflow() -> sciline.Pipeline:
    from ess.loki import data

    workflow = LokiAtLarmorWorkflow()

    workflow[PixelMaskFilename] = list(map(str, data.loki_tutorial_mask_filenames()))
    workflow[Filename[SampleRun]] = str(data.loki_tutorial_sample_run_60339())
    workflow[Filename[BackgroundRun]] = str(data.loki_tutorial_background_run_60393())
    workflow[Filename[TransmissionRun[SampleRun]]] = str(
        data.loki_tutorial_sample_transmission_run()
    )
    workflow[Filename[TransmissionRun[BackgroundRun]]] = str(
        data.loki_tutorial_run_60392()
    )
    workflow[Filename[EmptyBeamRun]] = str(data.loki_tutorial_run_60392())
    workflow[BeamCenter] = sc.vector(value=[-0.02914868, -0.01816138, 0.0], unit='m')
    return workflow


@register_workflow
def LokiWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for Loki.

    Returns
    -------
    :
        Loki workflow as a sciline.Pipeline
    """
    workflow = sans.SansWorkflow()
    for provider in loki_providers:
        workflow.insert(provider)
    for key, param in loki_default_parameters().items():
        workflow[key] = param
    workflow.typical_outputs = typical_outputs
    return workflow
