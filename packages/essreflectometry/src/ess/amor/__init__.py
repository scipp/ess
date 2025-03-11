# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import importlib.metadata

import sciline
import scipp as sc

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry import supermirror
from ..reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    DetectorSpatialResolution,
    NeXusDetectorName,
    RunType,
    SamplePosition,
)
from . import (
    conversions,
    figures,
    load,
    normalization,
    orso,
    resolution,
    utils,
    workflow,
)
from .instrument_view import instrument_view
from .types import (
    AngularResolution,
    ChopperFrequency,
    ChopperPhase,
    GravityToggle,
    QThetaFigure,
    ReflectivityDiagnosticsView,
    SampleSizeResolution,
    ThetaBins,
    WavelengthResolution,
    WavelengthThetaFigure,
    WavelengthZIndexFigure,
)

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

providers = (
    *reflectometry_providers,
    *load.providers,
    *conversions.providers,
    *normalization.providers,
    *utils.providers,
    *figures.providers,
    *orso.providers,
    *workflow.providers,
)
"""
List of providers for setting up a Sciline pipeline.

This provides a default Amor workflow including providers for loadings files.
"""


def default_parameters() -> dict:
    return {
        supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
        supermirror.CriticalEdge: 0.022 * sc.Unit("1/angstrom"),
        supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
        BeamSize[RunType]: 2.0 * sc.units.mm,
        DetectorSpatialResolution[RunType]: 0.0025 * sc.units.m,
        SamplePosition[RunType]: sc.vector([0, 0, 0], unit="m"),
        NeXusDetectorName[RunType]: "detector",
        ChopperPhase[RunType]: sc.scalar(7.0, unit="deg"),
        ChopperFrequency[RunType]: sc.scalar(8.333, unit="Hz"),
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
        GravityToggle: True,
    }


def AmorWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Amor PSI instrument.
    """
    return sciline.Pipeline(providers=providers, params=default_parameters())


__all__ = [
    "AmorWorkflow",
    "AngularResolution",
    "Chopper1Position",
    "Chopper2Position",
    "ChopperFrequency",
    "ChopperPhase",
    "QThetaFigure",
    "ReflectivityDiagnosticsView",
    "SampleSizeResolution",
    "ThetaBins",
    "WavelengthResolution",
    "WavelengthThetaFigure",
    "WavelengthZIndexFigure",
    "conversions",
    "default_parameters",
    "instrument_view",
    "load",
    "orso",
    "providers",
    "resolution",
    "supermirror",
]
