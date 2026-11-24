# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import importlib.metadata
from typing import Literal

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
    Position,
    ReferenceRun,
    RunType,
    SampleRotationOffset,
    SampleRun,
)
from . import (
    conversions,
    corrections,
    legacy,
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
    ChopperPhaseOffset,
    GravityToggle,
    ProtonCurrentThreshold,
    SampleSizeResolution,
    WavelengthResolution,
)

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

providers = (
    *reflectometry_providers,
    *load.providers,
    *conversions.providers,
    *corrections.providers,
    *normalization.providers,
    *utils.providers,
    *orso.providers,
    *workflow.providers,
)
"""
List of providers for setting up a Sciline pipeline.

This provides a default Amor workflow including providers for loadings files.
The providers that depend on the kind of reference measurement are
inserted by :func:`AmorWorkflow`.
"""


def default_parameters() -> dict:
    """Return the default parameters for the Amor workflow."""
    return {
        supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
        supermirror.CriticalEdge: 0.022 * sc.Unit("1/angstrom"),
        supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
        BeamSize[RunType]: 2.0 * sc.units.mm,
        DetectorSpatialResolution: 0.0025 * sc.units.m,
        Position[snx.NXsample, RunType]: sc.vector([0, 0, 0], unit="m"),
        NeXusDetectorName: "detector",
        ChopperPhaseOffset[RunType]: sc.scalar(-5.0, unit="deg"),
        ProtonCurrentThreshold: sc.scalar(50.0, unit="uA"),
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
        GravityToggle: True,
        SampleRotationOffset[RunType]: sc.scalar(0.0, unit='deg'),
    }


def AmorWorkflow(
    *,
    normalized_by: Literal[
        'supermirror', 'direct_beam', 'direct_beam_with_aperture'
    ] = 'supermirror',
) -> sciline.Pipeline:
    """
    Workflow with default parameters for the Amor PSI instrument.

    The workflow reads the files written by Amor since 2026.
    Use :func:`AmorWorkflow2025` for older files.

    Parameters
    ----------
    normalized_by:
        The kind of measurement that ``Filename[ReferenceRun]`` refers to.

        - ``'supermirror'``: a reflection from a supermirror. The reference is
          divided by the reflectivity of the supermirror.
        - ``'direct_beam'``: the unrestricted beam without a sample. Only the
          sample measurement receives a footprint correction.
        - ``'direct_beam_with_aperture'``: the direct beam through an aperture
          matching the sample footprint at the reference measurement's angle.
          Both measurements receive a footprint correction at their own angles.
          ``SampleSize[ReferenceRun]`` defaults to ``SampleSize[SampleRun]``.

    Returns
    -------
    :
        The workflow.
    """
    match normalized_by:
        case 'supermirror':
            reference_providers = normalization.supermirror_providers
        case 'direct_beam' | 'direct_beam_with_aperture':
            reference_providers = (
                *normalization.direct_beam_providers,
                conversions.direct_beam_coordinate_transformation_graph,
                orso.orso_amor_direct_beam_corrections,
            )
            if normalized_by == 'direct_beam_with_aperture':
                reference_providers += (
                    normalization.reduce_direct_beam_reference_with_aperture,
                )
        case _:
            raise ValueError(
                "normalized_by must be 'supermirror', 'direct_beam', "
                "or 'direct_beam_with_aperture', "
                f"got {normalized_by!r}"
            )
    workflow = sciline.Pipeline(
        providers=providers,
        params=default_parameters(),
        constraints={RunType: [SampleRun, ReferenceRun]},
    )
    for provider in reference_providers:
        workflow.insert(provider)
    return workflow


def AmorWorkflow2025() -> sciline.Pipeline:
    """
    Workflow with default parameters for the files written by Amor until 2025.

    The reference is a reflection from a supermirror.
    """
    return sciline.Pipeline(
        providers=(
            *reflectometry_providers,
            load.amor_chopper,
            load.load_amor_chopper_distance,
            load.load_amor_chopper_separation,
            load.load_beamline_metadata,
            load.load_measurement_metadata,
            normalization.evaluate_reference_at_sample_coords,
            normalization.mask_events_where_supermirror_does_not_cover,
            *utils.providers,
            *orso.providers,
            *legacy.providers,
        ),
        params=legacy.default_parameters(),
        constraints={RunType: [SampleRun, ReferenceRun]},
    )


__all__ = [
    "AmorWorkflow",
    "AmorWorkflow2025",
    "AngularResolution",
    "ChopperFrequency",
    "ChopperPhase",
    "ChopperPhaseOffset",
    "ProtonCurrentThreshold",
    "SampleSizeResolution",
    "WavelengthResolution",
    "conversions",
    "corrections",
    "default_parameters",
    "instrument_view",
    "legacy",
    "load",
    "normalization",
    "orso",
    "providers",
    "resolution",
    "supermirror",
]
