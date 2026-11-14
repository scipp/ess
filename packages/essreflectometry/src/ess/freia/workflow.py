# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import Literal

import sciline
import scipp as sc
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.uncertainty import UncertaintyBroadcastMode
from ess.reduce.unwrap import WavelengthLutMode
from ess.reduce.unwrap.types import KeepEventTimeOffset
from ess.reduce.unwrap.workflow import GenericUnwrapWorkflow
from ess.reduce.workflow import register_workflow
from scippnexus import NXmirror

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry.types import (
    DetectorSpatialResolution,
    LookupTableRelativeErrorThreshold,
    NeXusDetectorName,
    NeXusMonitorName,
    ReferenceRun,
    SampleRun,
)
from . import conversions, corrections, mcstas, mirror, normalization, orso
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
            "Multiblade": 0.035,
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
    inversion: bool = False,
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
    inversion:
        Use the inversion-mirror geometry described in :func:`FreiaWorkflow`.
        Supply ``MirrorPosition`` and ``MirrorNormal`` for both runs; the McStas
        adapter does not infer a mirror from guide walls. The loaded sample
        normal is reversed to describe reflection from the underside.
    """
    workflow = FreiaWorkflow(
        run_norm=run_norm,
        wavelength_from=wavelength_from,
        inversion=inversion,
        **kwargs,
    )
    for provider in mcstas.providers:
        workflow.insert(provider)
    for name, param in mcstas_default_parameters().items():
        workflow[name] = param
    if inversion:
        workflow.insert(mcstas.mcstas_inverted_sample_surface_normal)
    return workflow


def FreiaWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.proton_charge,
    wavelength_from: WavelengthLutMode = "file",
    inversion: bool = False,
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
    inversion:
        Enable a planar mirror upstream of the sample for illumination from
        below. Load its pose from the NXmirror named ``inversion_mirror``
        (configurable with ``NeXusName[NXmirror]``), or supply per-run
        ``MirrorPosition`` and ``MirrorNormal`` in laboratory coordinates.
        The detector flight path includes the mirror reflection. Supply the
        actual sample/detector positions and a ``SampleSurfaceNormal`` pointing
        out of the underside (normally -y) for both runs. The reference must be
        measured with the same mirror configuration.
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
    if inversion:
        _insert_mirror(workflow)
    return workflow


@register_workflow
def FreiaOffspecWorkflow(
    *,
    incident_beam_from: Literal['closest', 'fixed_slits', 'shutter_logs'] = 'closest',
    wavelength_from: WavelengthLutMode = "file",
    inversion: bool = False,
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for unnormalized, coplanar off-specular FREIA data.

    Supply UpstreamSlitCenters and DownstreamSlitCenters for SampleRun as
    dictionaries pairing the fixed openings of each candidate incident beam.
    The assigned ``incident_angle`` and measured ``reflection_angle`` determine
    ``Qx`` and ``Qz``. The output is ``CorrectedDetector[SampleRun]``.

    All three modes assign a nominal angle and signed effective gravity length.
    These are computed once per candidate from the slit centres, sample plane,
    and (in inversion mode) mirror pose. A coordinate kernel then corrects the
    assigned angle using the event's wavelength. Gravity stays in the lab frame.
    Setting ``GravityVector`` to zero disables both incident and outgoing
    gravity corrections. Flight lengths and shutter crossing times retain
    their straight-segment approximations.

    Parameters
    ----------
    incident_beam_from:
        ``'closest'`` assigns the beam whose nominal specular reflection angle
        is closest to the geometrical sample-to-pixel angle. This heuristic uses
        no gravity correction; gravity is applied after the beam is selected.
        ``'fixed_slits'`` uses the single beam defined by one matching pair of
        slit centers, independently of the reflection angle and without shutter
        positions or status logs.
        ``'shutter_logs'`` uses ``ShutterStatusLogs[SampleRun]`` and
        ``ShutterPositions[SampleRun]`` for all candidate beams.
        Each log is evaluated at the neutron's crossing
        time at that shutter, independently of the slit positions. Exactly one
        open shutter, with all states known, determines the incident beam.
        Otherwise the incidence is NaN and the event is masked as
        ``incident_beam_unassigned``.
    wavelength_from:
        Wavelength lookup table mode: 'analytical', 'simulation', or 'file'.
    inversion:
        Enable the mirror geometry described in :func:`FreiaWorkflow`.
        The mirror is between the slit assemblies, after the shutters and
        before the sample. Supply all positions in laboratory coordinates.
        Incident angles and shutter crossing times include the reflection.
        Use the actual sample position and a surface normal pointing out of
        the underside.
    """
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun],
        monitor_types=[],
        wavelength_from=wavelength_from,
        **kwargs,
    )
    assignment_provider = {
        'closest': conversions.closest_beam_incident_angle_graph,
        'fixed_slits': conversions.fixed_slits_incident_angle_graph,
        'shutter_logs': conversions.shutter_log_incident_angle_graph,
    }[incident_beam_from]
    for provider in (
        mirror.no_mirror,
        conversions.offspecular_coordinate_transformation_graph,
        assignment_provider,
        corrections.add_offspecular_coords_and_masks,
    ):
        workflow.insert(provider)
    for name, param in default_parameters().items():
        workflow[name] = param
    workflow[KeepEventTimeOffset] = incident_beam_from == 'shutter_logs'
    workflow[DetectorRegionOfInterest[SampleRun]] = {}
    if inversion:
        _insert_mirror(workflow)
    return workflow


def _insert_mirror(workflow: sciline.Pipeline) -> None:
    for provider in mirror.providers:
        workflow.insert(provider)
    workflow[NeXusName[NXmirror]] = 'inversion_mirror'


@register_workflow
def FreiaMcStasUnnormalizedWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia McStas data without run normalization."""
    return FreiaMcStasWorkflow(run_norm=RunNormalization.none, inversion=inversion)


@register_workflow
def FreiaMcStasMonitorHistogramWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia McStas data using histogrammed monitor normalization."""
    return FreiaMcStasWorkflow(
        run_norm=RunNormalization.monitor_histogram, inversion=inversion
    )


@register_workflow
def FreiaMcStasMonitorIntegratedWorkflow(
    *, inversion: bool = False
) -> sciline.Pipeline:
    """Workflow for Freia McStas data using integrated monitor normalization."""
    return FreiaMcStasWorkflow(
        run_norm=RunNormalization.monitor_integrated, inversion=inversion
    )


@register_workflow
def FreiaMcStasProtonChargeWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia McStas data using proton charge normalization."""
    return FreiaMcStasWorkflow(
        run_norm=RunNormalization.proton_charge, inversion=inversion
    )


@register_workflow
def FreiaUnnormalizedWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia NeXus data without run normalization."""
    return FreiaWorkflow(run_norm=RunNormalization.none, inversion=inversion)


@register_workflow
def FreiaMonitorHistogramWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia NeXus data using histogrammed monitor normalization."""
    return FreiaWorkflow(
        run_norm=RunNormalization.monitor_histogram, inversion=inversion
    )


@register_workflow
def FreiaMonitorIntegratedWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia NeXus data using integrated monitor normalization."""
    return FreiaWorkflow(
        run_norm=RunNormalization.monitor_integrated, inversion=inversion
    )


@register_workflow
def FreiaProtonChargeWorkflow(*, inversion: bool = False) -> sciline.Pipeline:
    """Workflow for Freia NeXus data using proton charge normalization."""
    return FreiaWorkflow(run_norm=RunNormalization.proton_charge, inversion=inversion)


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
