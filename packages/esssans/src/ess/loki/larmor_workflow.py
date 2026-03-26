# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the Loki workflow.
"""

import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.workflow import register_workflow
from scippneutron.conversion.graph import beamline, tof

from ess import sans
from ess.sans.io import read_xml_detector_masking
from ess.sans.parameters import typical_outputs

from ..sans.conversions import ElasticCoordTransformGraph, sans_elastic
from ..sans.types import (
    BackgroundRun,
    BeamCenter,
    CorrectForGravity,
    DetectorBankSizes,
    EmptyBeamRun,
    Filename,
    GravityVector,
    Incident,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelMaskFilename,
    PixelShapePath,
    Position,
    RawDetector,
    RawMonitor,
    RunType,
    SampleRun,
    Transmission,
    TransmissionRun,
    WavelengthDetector,
    WavelengthMonitor,
)
from .workflow import loki_providers

DETECTOR_BANK_SIZES = {
    'larmor_detector': {'layer': 4, 'tube': -1, 'straw': 7, 'pixel': 512},
}


class MonitorCoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """
    Coordinate transformation graph that allows to compute wavelength from tof for
    Larmor monitors (no scattering).
    """


def larmor_default_parameters() -> dict:
    return {
        DetectorBankSizes: DETECTOR_BANK_SIZES,
        NeXusMonitorName[Incident]: 'monitor_1',
        NeXusMonitorName[Transmission]: 'monitor_2',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
    }


def _larmor_convert_to_wavelength(da: sc.DataArray, graph: dict) -> sc.DataArray:
    event_time_offset = da.bins.coords['event_time_offset']
    da = da.bins.drop_coords('event_time_offset')
    da.bins.coords['tof'] = event_time_offset
    if 'event_time_zero' in da.dims:
        da = da.bins.concat('event_time_zero')
    return da.transform_coords('wavelength', graph=graph, keep_intermediate=False)


def larmor_data_to_wavelength(
    da: RawDetector[RunType], graph: ElasticCoordTransformGraph[RunType]
) -> WavelengthDetector[RunType]:
    """
    Compute wavelength coordinate for Loki detector data at Larmor.
    This is different from the standard conversion from the GenericUnwrapWorkflow
    because the detector test was conducted as ISIS where the pulse has a different time
    structure.
    The conversion here is much simpler: the event_time_offset coordinate is directly
    renamed as time-of-flight, and a wavelength is computed from that.
    """
    return WavelengthDetector[RunType](_larmor_convert_to_wavelength(da, graph=graph))


def larmor_monitor_to_wavelength(
    da: RawMonitor[RunType, MonitorType], graph: MonitorCoordTransformGraph[RunType]
) -> WavelengthMonitor[RunType, MonitorType]:
    """
    Compute wavelength coordinate for Loki monitor data at Larmor.
    This is different from the standard conversion from the GenericUnwrapWorkflow
    because the detector test was conducted as ISIS where the pulse has a different time
    structure.
    The conversion here is much simpler: the event_time_offset coordinate is directly
    renamed as time-of-flight, and a wavelength is computed from that.
    """
    return WavelengthMonitor[RunType, MonitorType](
        _larmor_convert_to_wavelength(da, graph=graph)
    )


def larmor_detector_coord_transform_graph(
    correct_for_gravity: CorrectForGravity,
    *,
    sample_position: Position[snx.NXsample, RunType],
    source_position: Position[snx.NXsource, RunType],
    gravity: GravityVector,
) -> ElasticCoordTransformGraph[RunType]:
    graph = sans_elastic(
        correct_for_gravity=correct_for_gravity,
        sample_position=sample_position,
        source_position=source_position,
        gravity=gravity,
    )
    return ElasticCoordTransformGraph[RunType]({**graph, **tof.elastic_Q('tof')})


def larmor_monitor_coord_transform_graph(
    source_position: Position[snx.NXsource, RunType],
) -> MonitorCoordTransformGraph[RunType]:
    """
    Generate a coordinate transformation graph for SANS monitor (no scattering).
    """
    return MonitorCoordTransformGraph(
        {
            **beamline.beamline(scatter=False),
            **tof.elastic_wavelength('tof'),
            'source_position': lambda: source_position,
        }
    )


loki_at_larmor_providers = (
    larmor_data_to_wavelength,
    larmor_monitor_to_wavelength,
    larmor_detector_coord_transform_graph,
    larmor_monitor_coord_transform_graph,
    read_xml_detector_masking,
)


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
    for provider in loki_at_larmor_providers:
        workflow.insert(provider)
    for key, param in larmor_default_parameters().items():
        workflow[key] = param
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
