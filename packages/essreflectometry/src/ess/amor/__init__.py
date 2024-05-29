# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import importlib.metadata

import scipp as sc

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry import supermirror
from ..reflectometry.types import (
    BeamSize,
    DetectorSpatialResolution,
    Gravity,
    NeXusDetectorName,
    Run,
    SamplePosition,
    SampleSize,
)
from . import conversions, data, load, orso, resolution, utils
from .types import (
    AngularResolution,
    Chopper1Position,
    Chopper2Position,
    ChopperFrequency,
    ChopperPhase,
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
    *resolution.providers,
    *utils.providers,
)
"""
List of providers for setting up a Sciline pipeline.

This provides a default Amor workflow including providers for loadings files.
"""

default_parameters = {
    supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
    supermirror.CriticalEdge: 0.022 * sc.Unit('1/angstrom'),
    supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
    BeamSize[Run]: 2.0 * sc.units.mm,
    SampleSize[Run]: 10.0 * sc.units.mm,
    DetectorSpatialResolution[Run]: 0.0025 * sc.units.m,
    Gravity: sc.vector(value=[0, -1, 0]) * sc.constants.g,
    SamplePosition[Run]: sc.vector([0, 0, 0], unit='m'),
    NeXusDetectorName[Run]: 'detector',
    ChopperPhase[Run]: sc.scalar(-5.0, unit='deg'),
    ChopperFrequency[Run]: sc.scalar(8.333, unit='Hz'),
}

del sc

__all__ = [
    "supermirror",
    "beamline",
    "conversions",
    "data",
    "load",
    "orso",
    "resolution",
    "instrument_view",
    "providers",
    "default_parameters",
    "WavelengthResolution",
    "AngularResolution",
    "SampleSizeResolution",
    "ChopperFrequency",
    "ChopperPhase",
    "Chopper1Position",
    "Chopper2Position",
    "ThetaBins",
    "WavelengthThetaFigure",
    "WavelengthZIndexFigure",
    "QThetaFigure",
    "ReflectivityDiagnosticsView",
]
