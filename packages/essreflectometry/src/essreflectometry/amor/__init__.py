# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F401
import itertools

import scipp as sc

from .. import providers as reflectometry_providers
from .. import supermirror
from ..types import BeamSize, DetectorSpatialResolution, Gravity, Run, SampleSize
from . import beamline, conversions, load, resolution
from .beamline import instrument_view_components
from .instrument_view import instrument_view
from .types import (
    AngularResolution,
    BeamlineParams,
    Chopper1Position,
    Chopper2Position,
    ChopperFrequency,
    ChopperPhase,
    SampleSizeResolution,
    WavelengthResolution,
)

providers = list(
    itertools.chain(
        reflectometry_providers,
        load.providers,
        conversions.providers,
        resolution.providers,
        beamline.providers,
    )
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
    ChopperFrequency[Run]: sc.scalar(20 / 3, unit='Hz'),
    ChopperPhase[Run]: sc.scalar(-8.0, unit='deg'),
    Chopper1Position[Run]: sc.vector(value=[0, 0, -15.5], unit='m'),
    Chopper2Position[Run]: sc.vector(value=[0, 0, -14.5], unit='m'),
}

del sc
del itertools
