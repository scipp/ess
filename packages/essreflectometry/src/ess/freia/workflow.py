# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import Literal

import sciline
import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode
from ess.reduce.workflow import register_workflow

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CorrectionsToApply,
    DetectorSpatialResolution,
    LookupTableRelativeErrorThreshold,
    NeXusDetectorName,
    RunType,
    SampleRotationOffset,
)
from . import (
    beamline,
    conversions,
    corrections,
    load,
    maskings,
    mcstas,
    normalization,
    orso,
)
from .corrections import RunNormalization, insert_run_normalization

_general_providers = (
    *reflectometry_providers,
    *conversions.providers,
    *corrections.providers,
    *maskings.providers,
    *normalization.providers,
    *orso.providers,
    *load.providers,
)

mcstas_providers = (
    *_general_providers,
    *mcstas.providers,
)
"""List of providers for setting up a Sciline pipeline for McStas data.

This provides a default Freia workflow including providers for loadings files.
"""

providers = (*_general_providers,)
"""List of providers for setting up a Sciline pipeline data.

This provides a default Freia workflow including providers for loadings files.
"""


def mcstas_default_parameters() -> dict:
    """Return default parameters for the McStas Freia workflow."""
    return {
        DetectorSpatialResolution: 0.0025 * sc.units.m,
        NeXusDetectorName: "detector",
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
        SampleRotationOffset[RunType]: sc.scalar(0.0, unit='deg'),
        CorrectionsToApply: corrections.default_corrections,
        LookupTableRelativeErrorThreshold: {
            "detector": 0.06,
        },
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    }


def default_parameters() -> dict:
    """Return default parameters for the NeXus Freia workflow."""
    return {
        NeXusDetectorName: "multiblade_detector",
        SampleRotationOffset[RunType]: sc.scalar(0.0, unit='deg'),
        CorrectionsToApply: corrections.default_corrections,
        DetectorSpatialResolution: 0.0025 * sc.units.m,
        LookupTableRelativeErrorThreshold: {
            "multiblade_detector": float('inf'),
        },
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    }


def FreiaMcStasWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.none,
    mode: Literal["analytical", "simulation", "file"] = "file",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for reduction of McStas data for the Freia instrument.

    Parameters
    ----------
    run_norm:
        Normalization procedure to be used. See :class:`RunNormalization`.
    mode:
        Mode for creating the wavelength lookup table. The 'analytical' mode uses
        analytical calculations to propagate and chop a pulse through the chopper
        cascade and build the lookup table. The 'simulation' mode uses ``tof`` to trace
        individual neutrons through the chopper system and build the table.
        The 'file' mode loads a pre-computed table from a file.
    """
    workflow = beamline.LoadNeXusWorkflow(mode=mode, **kwargs)
    for provider in mcstas_providers:
        workflow.insert(provider)
    insert_run_normalization(workflow, run_norm)
    for name, param in mcstas_default_parameters().items():
        workflow[name] = param
    return workflow


def FreiaWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.proton_charge,
    mode: Literal["analytical", "simulation", "file"] = "file",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for reduction of data for the Freia instrument.

    Parameters
    ----------
    run_norm:
        Normalization procedure to be used. See :class:`RunNormalization`.
    mode:
        Mode for creating the wavelength lookup table. The 'analytical' mode uses
        analytical calculations to propagate and chop a pulse through the chopper
        cascade and build the lookup table. The 'simulation' mode uses ``tof`` to trace
        individual neutrons through the chopper system and build the table.
        The 'file' mode loads a pre-computed table from a file.
    """
    workflow = beamline.LoadNeXusWorkflow(mode=mode, **kwargs)
    for provider in providers:
        workflow.insert(provider)
    insert_run_normalization(workflow, run_norm)
    for name, param in default_parameters().items():
        workflow[name] = param
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
    'FreiaProtonChargeWorkflow',
    'FreiaUnnormalizedWorkflow',
    'FreiaWorkflow',
    'default_parameters',
    'mcstas_default_parameters',
]
