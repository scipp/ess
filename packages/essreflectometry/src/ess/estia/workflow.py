# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import Literal

import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import TransformationTimeFilter
from ess.reduce.uncertainty import UncertaintyBroadcastMode
from ess.reduce.workflow import register_workflow

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry import supermirror
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

This provides a default Estia workflow including providers for loadings files.
"""

providers = (*_general_providers,)
"""List of providers for setting up a Sciline pipeline data.

This provides a default Estia workflow including providers for loadings files.
"""


def mcstas_default_parameters() -> dict:
    """Return default parameters for the McStas Estia workflow."""
    return {
        supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
        # The reference sample in the McStas simulation has R=1 everywhere
        supermirror.CriticalEdge: sc.scalar(float('inf'), unit='1/angstrom'),
        supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
        DetectorSpatialResolution: 0.0025 * sc.units.m,
        NeXusDetectorName: "multiblade_detector",
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
        SampleRotationOffset[RunType]: sc.scalar(0.0, unit='deg'),
        CorrectionsToApply: corrections.default_corrections,
        LookupTableRelativeErrorThreshold: {
            "multiblade_detector": 0.06,
        },
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    }


def default_parameters() -> dict:
    """Return default parameters for the NeXus Estia workflow."""
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


def EstiaMcStasWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.none,
    wavelength_from: Literal["analytical", "simulation", "file"] = "file",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for reduction of McStas data for the Estia instrument.

    Parameters
    ----------
    run_norm:
        Normalization procedure to be used. See :class:`RunNormalization`.
    wavelength_from:
        Mode for creating the wavelength lookup table. The 'analytical' mode uses
        analytical calculations to propagate and chop a pulse through the chopper
        cascade and build the lookup table. The 'simulation' mode uses ``tof`` to trace
        individual neutrons through the chopper system and build the table.
        The 'file' mode loads a pre-computed table from a file.
    """
    workflow = beamline.LoadNeXusWorkflow(wavelength_from=wavelength_from, **kwargs)
    for provider in mcstas_providers:
        workflow.insert(provider)
    insert_run_normalization(workflow, run_norm)
    for name, param in mcstas_default_parameters().items():
        workflow[name] = param
    return workflow


def EstiaWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.proton_charge,
    wavelength_from: Literal["analytical", "simulation", "file"] = "file",
    **kwargs,
) -> sciline.Pipeline:
    """Workflow for reduction of data for the Estia instrument.

    Parameters
    ----------
    run_norm:
        Normalization procedure to be used. See :class:`RunNormalization`.
    wavelength_from:
        Mode for creating the wavelength lookup table. The 'analytical' mode uses
        analytical calculations to propagate and chop a pulse through the chopper
        cascade and build the lookup table. The 'simulation' mode uses ``tof`` to trace
        individual neutrons through the chopper system and build the table.
        The 'file' mode loads a pre-computed table from a file.
    """
    workflow = beamline.LoadNeXusWorkflow(wavelength_from=wavelength_from, **kwargs)
    for provider in providers:
        workflow.insert(provider)
    insert_run_normalization(workflow, run_norm)
    for name, param in default_parameters().items():
        workflow[name] = param

    workflow[TransformationTimeFilter[snx.NXdetector, RunType]] = (
        # Default to zero detector rotation if the log is empty.
        # In practice it should never be empty, and it cannot be reduced,
        # but this default makes it possible to at least load the data
        # for visualization.
        corrections.assume_time_series_constant_with_zero_default_value_if_empty
    )
    return workflow


@register_workflow
def EstiaMcStasUnnormalizedWorkflow() -> sciline.Pipeline:
    """Workflow for Estia McStas data without run normalization."""
    return EstiaMcStasWorkflow(run_norm=RunNormalization.none)


@register_workflow
def EstiaMcStasMonitorHistogramWorkflow() -> sciline.Pipeline:
    """Workflow for Estia McStas data using histogrammed monitor normalization."""
    return EstiaMcStasWorkflow(run_norm=RunNormalization.monitor_histogram)


@register_workflow
def EstiaMcStasMonitorIntegratedWorkflow() -> sciline.Pipeline:
    """Workflow for Estia McStas data using integrated monitor normalization."""
    return EstiaMcStasWorkflow(run_norm=RunNormalization.monitor_integrated)


@register_workflow
def EstiaMcStasProtonChargeWorkflow() -> sciline.Pipeline:
    """Workflow for Estia McStas data using proton charge normalization."""
    return EstiaMcStasWorkflow(run_norm=RunNormalization.proton_charge)


@register_workflow
def EstiaUnnormalizedWorkflow() -> sciline.Pipeline:
    """Workflow for Estia NeXus data without run normalization."""
    return EstiaWorkflow(run_norm=RunNormalization.none)


@register_workflow
def EstiaMonitorHistogramWorkflow() -> sciline.Pipeline:
    """Workflow for Estia NeXus data using histogrammed monitor normalization."""
    return EstiaWorkflow(run_norm=RunNormalization.monitor_histogram)


@register_workflow
def EstiaMonitorIntegratedWorkflow() -> sciline.Pipeline:
    """Workflow for Estia NeXus data using integrated monitor normalization."""
    return EstiaWorkflow(run_norm=RunNormalization.monitor_integrated)


@register_workflow
def EstiaProtonChargeWorkflow() -> sciline.Pipeline:
    """Workflow for Estia NeXus data using proton charge normalization."""
    return EstiaWorkflow(run_norm=RunNormalization.proton_charge)


__all__ = [
    'EstiaMcStasMonitorHistogramWorkflow',
    'EstiaMcStasMonitorIntegratedWorkflow',
    'EstiaMcStasProtonChargeWorkflow',
    'EstiaMcStasUnnormalizedWorkflow',
    'EstiaMcStasWorkflow',
    'EstiaMonitorHistogramWorkflow',
    'EstiaMonitorIntegratedWorkflow',
    'EstiaProtonChargeWorkflow',
    'EstiaUnnormalizedWorkflow',
    'EstiaWorkflow',
    'default_parameters',
    'mcstas_default_parameters',
]
