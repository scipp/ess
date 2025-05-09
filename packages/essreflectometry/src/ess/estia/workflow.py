# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc
import scippnexus as snx

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry import supermirror
from ..reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    DetectorSpatialResolution,
    NeXusDetectorName,
    RunType,
    Position,
)
from . import conversions, load, maskings, normalization, orso, corrections


mcstas_providers = (
    *reflectometry_providers,
    *load.providers,
    *conversions.providers,
    *corrections.providers,
    *maskings.providers,
    *normalization.providers,
    *orso.providers,
)
"""List of providers for setting up a Sciline pipeline for McStas data.

This provides a default Estia workflow including providers for loadings files.
"""


def mcstas_default_parameters() -> dict:
    return {
        supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
        supermirror.CriticalEdge: 0.022 * sc.Unit("1/angstrom"),
        supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
        BeamSize[RunType]: 2.0 * sc.units.mm,
        DetectorSpatialResolution[RunType]: 0.0025 * sc.units.m,
        Position[snx.NXsample, RunType]: sc.vector([0, 0, 0], unit="m"),
        NeXusDetectorName: "detector",
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
    }


def EstiaMcStasWorkflow() -> sciline.Pipeline:
    """Workflow for reduction of McStas data for the Estia instrument."""
    return sciline.Pipeline(providers=mcstas_providers, params=mcstas_default_parameters())
