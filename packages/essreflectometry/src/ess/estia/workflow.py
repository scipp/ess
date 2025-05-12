# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc

from ess.reduce import nexus

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry import supermirror
from ..reflectometry.types import (
    BeamDivergenceLimits,
    DetectorSpatialResolution,
    NeXusDetectorName,
    ReferenceRun,
    RunType,
    SampleRotationOffset,
    SampleRun,
)
from . import conversions, corrections, load, maskings, normalization, orso

_general_providers = (
    *reflectometry_providers,
    *conversions.providers,
    *corrections.providers,
    *maskings.providers,
    *normalization.providers,
    *orso.providers,
)

mcstas_providers = (
    *_general_providers,
    *load.providers,
)
"""List of providers for setting up a Sciline pipeline for McStas data.

This provides a default Estia workflow including providers for loadings files.
"""

providers = (*_general_providers,)
"""List of providers for setting up a Sciline pipeline data.

This provides a default Estia workflow including providers for loadings files.
"""


def mcstas_default_parameters() -> dict:
    return {
        supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
        supermirror.CriticalEdge: 0.022 * sc.Unit("1/angstrom"),
        supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
        DetectorSpatialResolution[RunType]: 0.0025 * sc.units.m,
        NeXusDetectorName: "detector",
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
        SampleRotationOffset[RunType]: sc.scalar(0.0, unit='deg'),
    }


def default_parameters() -> dict:
    return {
        NeXusDetectorName: "multiblade_detector",
    }


def EstiaMcStasWorkflow() -> sciline.Pipeline:
    """Workflow for reduction of McStas data for the Estia instrument."""
    return sciline.Pipeline(
        providers=mcstas_providers, params=mcstas_default_parameters()
    )


def EstiaWorkflow() -> sciline.Pipeline:
    """Workflow for reduction of data for the Estia instrument."""
    workflow = nexus.GenericNeXusWorkflow(
        run_types=[SampleRun, ReferenceRun], monitor_types=[]
    )
    for provider in providers:
        workflow.insert(provider)
    for name, param in default_parameters().items():
        workflow[name] = param
    return workflow
