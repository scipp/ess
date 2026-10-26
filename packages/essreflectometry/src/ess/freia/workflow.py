# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import Literal

import sciline
import scipp as sc
from ess.reduce.nexus.types import DetectorBankSizes
from ess.reduce.uncertainty import UncertaintyBroadcastMode
from ess.reduce.unwrap import WavelengthLutMode
from ess.reduce.unwrap.types import KeepEventTimeOffset
from ess.reduce.unwrap.workflow import GenericUnwrapWorkflow
from ess.reduce.workflow import register_workflow

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry.types import (
    DetectorSpatialResolution,
    LookupTableRelativeErrorThreshold,
    NeXusDetectorName,
    NeXusMonitorName,
    ReferenceRun,
    SampleRun,
)
from . import conversions, corrections, mcstas, normalization, orso
from .corrections import RunNormalization, insert_run_normalization
from .types import (
    DetectorRegionOfInterest,
    NormalizationMonitor,
    PSCMonitor,
    ShutterMonitor,
    WBC1Monitor,
    WBC2Monitor,
    WBC3Monitor,
)

providers = (
    *reflectometry_providers,
    *conversions.providers,
    *corrections.providers,
    *normalization.providers,
    *orso.providers,
)

"""List of providers for setting up a Sciline pipeline data.

This provides a default Freia workflow including providers for loading files.
"""


def mcstas_default_parameters() -> dict:
    """Return default parameters for the McStas Freia workflow."""
    return default_parameters() | {
        NeXusDetectorName: "Multiblade",
        LookupTableRelativeErrorThreshold: {
            "Multiblade": 0.06,
        },
    }


def default_parameters() -> dict:
    """Return default parameters for the NeXus Freia workflow."""
    return {
        NeXusDetectorName: "multiblade_detector",
        NeXusMonitorName[WBC1Monitor]: 'monitor_1',
        NeXusMonitorName[PSCMonitor]: 'monitor_2',
        NeXusMonitorName[WBC2Monitor]: 'monitor_3',
        NeXusMonitorName[WBC3Monitor]: 'monitor_4',
        NeXusMonitorName[ShutterMonitor]: 'monitor_5',
        NeXusMonitorName[NormalizationMonitor]: 'monitor_6',
        DetectorBankSizes: {
            "multiblade_detector": {"strip": 64, "blade": 32, "wire": 32},
        },
        DetectorSpatialResolution: 0.0025 * sc.units.m,
        LookupTableRelativeErrorThreshold: {
            "multiblade_detector": float('inf'),
            "monitor_1": float('inf'),
            "monitor_2": float('inf'),
            "monitor_3": float('inf'),
            "monitor_4": float('inf'),
            "monitor_5": float('inf'),
            "monitor_6": float('inf'),
        },
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    }


def FreiaMcStasWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.none,
    wavelength_from: WavelengthLutMode = "analytical",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for reducing FREIA McStas events with a no-sample direct beam.

    Loads geometry and uses the default WFM chopper settings. Reduction inputs
    and outputs are described in :func:`FreiaWorkflow`.

    Parameters
    ----------
    run_norm:
        Normalization procedure to be used. See :class:`RunNormalization`.
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
    """
    workflow = FreiaWorkflow(
        run_norm=run_norm, wavelength_from=wavelength_from, **kwargs
    )
    for provider in mcstas.providers:
        workflow.insert(provider)
    for name, param in mcstas_default_parameters().items():
        workflow[name] = param
    return workflow


def FreiaWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.proton_charge,
    wavelength_from: WavelengthLutMode = "file",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for reduction of data for the Freia instrument.

    The coordinate transformation graph computes the signed, gravity-corrected
    scattering angle above the laboratory x-z plane for both runs, with reflection
    angle and Q for the sample. The direct beam is mapped to Q when building
    ``Reference``. Reflectivity
    requires separate sample/direct-beam ROIs, wavelength and Q bins, and beam
    and sample sizes for the footprint correction. The reference run must be a
    measurement without a sample, taken with matching slit and chopper settings.
    Set its ``SampleSurfaceNormal`` to the sample run's orientation.

    To skip footprint correction, set
    ``workflow[Sample] = workflow[ReducibleData[SampleRun]]``.

    Monitor normalization requires an incident monitor selected through
    ``NeXusName[NormalizationMonitor]``, or supplied as ``WavelengthMonitor[RunType]``.

    Parameters
    ----------
    run_norm:
        Normalization procedure to be used. See :class:`RunNormalization`.
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
    """
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun, ReferenceRun],
        monitor_types=[
            WBC1Monitor,
            PSCMonitor,
            WBC2Monitor,
            WBC3Monitor,
            ShutterMonitor,
            NormalizationMonitor,
        ],
        wavelength_from=wavelength_from,
        **kwargs,
    )
    for provider in providers:
        workflow.insert(provider)
    insert_run_normalization(workflow, run_norm)
    for name, param in default_parameters().items():
        workflow[name] = param
    return workflow


@register_workflow
def FreiaOffspecWorkflow(
    *,
    incident_beam_from: Literal['closest', 'slit_logs'] = 'closest',
    wavelength_from: WavelengthLutMode = "file",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for unnormalized, coplanar off-specular FREIA data.

    Supply UpstreamSlitCenters and DownstreamSlitCenters for SampleRun as
    dictionaries pairing the fixed openings of each candidate incident beam.
    The assigned ``incident_angle`` and measured ``reflection_angle`` determine
    ``Qx`` and ``Qz``. The output is ``CorrectedDetector[SampleRun]``.

    Parameters
    ----------
    incident_beam_from:
        ``'closest'`` assigns the beam whose specular reflection angle is closest
        to the measured reflection angle. ``'slit_logs'`` instead uses
        ``SlitStatusLogs[SampleRun]`` for all candidate beams, evaluating all
        logs at a shared ``time_at_slit`` coordinate. The common approximate
        position is the mean downstream slit center. Exactly one open slit
        determines the incident beam. Otherwise the incidence is NaN and the
        event is masked as ``incident_beam_unassigned``; no closest-beam fallback
        is applied.
    wavelength_from:
        Wavelength lookup table mode: 'analytical', 'simulation', or 'file'.
    """
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun],
        monitor_types=[],
        wavelength_from=wavelength_from,
        **kwargs,
    )
    assignment_provider = {
        'closest': conversions.closest_beam_incident_angle_graph,
        'slit_logs': conversions.slit_log_incident_angle_graph,
    }[incident_beam_from]
    for provider in (
        conversions.offspecular_coordinate_transformation_graph,
        assignment_provider,
        corrections.add_coords_and_masks,
    ):
        workflow.insert(provider)
    for name, param in default_parameters().items():
        workflow[name] = param
    workflow[KeepEventTimeOffset] = incident_beam_from == 'slit_logs'
    workflow[DetectorRegionOfInterest[SampleRun]] = {}
    return workflow


@register_workflow
def FreiaMcStasUnnormalizedWorkflow() -> sciline.Pipeline:
    """Workflow for Freia McStas data without run normalization."""
    return FreiaMcStasWorkflow(run_norm=RunNormalization.none)


@register_workflow
def FreiaMcStasMonitorHistogramWorkflow() -> sciline.Pipeline:
    """Workflow for Freia McStas data using histogrammed monitor normalization."""
    return FreiaMcStasWorkflow(run_norm=RunNormalization.monitor_histogram)


@register_workflow
def FreiaMcStasMonitorIntegratedWorkflow() -> sciline.Pipeline:
    """Workflow for Freia McStas data using integrated monitor normalization."""
    return FreiaMcStasWorkflow(run_norm=RunNormalization.monitor_integrated)


@register_workflow
def FreiaMcStasProtonChargeWorkflow() -> sciline.Pipeline:
    """Workflow for Freia McStas data using proton charge normalization."""
    return FreiaMcStasWorkflow(run_norm=RunNormalization.proton_charge)


@register_workflow
def FreiaUnnormalizedWorkflow() -> sciline.Pipeline:
    """Workflow for Freia NeXus data without run normalization."""
    return FreiaWorkflow(run_norm=RunNormalization.none)


@register_workflow
def FreiaMonitorHistogramWorkflow() -> sciline.Pipeline:
    """Workflow for Freia NeXus data using histogrammed monitor normalization."""
    return FreiaWorkflow(run_norm=RunNormalization.monitor_histogram)


@register_workflow
def FreiaMonitorIntegratedWorkflow() -> sciline.Pipeline:
    """Workflow for Freia NeXus data using integrated monitor normalization."""
    return FreiaWorkflow(run_norm=RunNormalization.monitor_integrated)


@register_workflow
def FreiaProtonChargeWorkflow() -> sciline.Pipeline:
    """Workflow for Freia NeXus data using proton charge normalization."""
    return FreiaWorkflow(run_norm=RunNormalization.proton_charge)


__all__ = [
    'FreiaMcStasMonitorHistogramWorkflow',
    'FreiaMcStasMonitorIntegratedWorkflow',
    'FreiaMcStasProtonChargeWorkflow',
    'FreiaMcStasUnnormalizedWorkflow',
    'FreiaMcStasWorkflow',
    'FreiaMonitorHistogramWorkflow',
    'FreiaMonitorIntegratedWorkflow',
    'FreiaOffspecWorkflow',
    'FreiaProtonChargeWorkflow',
    'FreiaUnnormalizedWorkflow',
    'FreiaWorkflow',
    'default_parameters',
    'mcstas_default_parameters',
]
