# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from scippneutron.conversion import graph, tof
from scippnexus import NXsample, NXsource

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import CoordTransformationGraph, RunType, WavelengthDetector
from .types import QDetector, SampleSurfaceNormal


def theta(
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
    sample_surface_normal: sc.Variable,
) -> sc.Variable:
    """Signed, gravity-corrected angle above the sample plane.

    Approximate the flight time using the straight sample-to-detector distance,
    as in ScippNeutron's gravity correction. Positive angles point toward the
    sample surface normal.
    """
    flight_time = tof.tof_from_wavelength(
        wavelength=wavelength, Ltotal=sc.norm(scattered_beam)
    ).to(unit='s')
    outgoing_beam = scattered_beam - (0.5 * gravity * flight_time**2).to(
        unit=scattered_beam.unit
    )
    normal = sample_surface_normal / sc.norm(sample_surface_normal)
    return sc.asin(sc.dot(outgoing_beam, normal) / sc.norm(outgoing_beam))


def coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    gravity: GravityVector,
) -> CoordTransformationGraph[RunType]:
    """Build a specular conversion graph."""
    length = sc.norm(sample_surface_normal)
    if (
        not sc.isfinite(length).value
        or not (length > sc.scalar(0.0, unit=length.unit)).value
    ):
        raise ValueError('SampleSurfaceNormal must be a finite, nonzero vector.')
    return {
        **graph.beamline.beamline(scatter=True),
        'theta': theta,
        'Q': reflectometry_q,
        'source_position': lambda: source_position,
        'sample_position': lambda: sample_position,
        'sample_surface_normal': lambda: sample_surface_normal,
        'gravity': lambda: gravity,
    }


def add_coords(
    da: WavelengthDetector[RunType],
    graph: CoordTransformationGraph[RunType],
) -> QDetector[RunType]:
    """Add specular Q and the gravity-corrected angle to detector events."""
    return QDetector[RunType](
        da.transform_coords(
            (
                'theta',
                'Q',
                'L1',
                'L2',
                'incident_beam',
                'sample_position',
                'sample_surface_normal',
            ),
            graph,
            rename_dims=False,
            keep_intermediate=False,
            keep_aliases=False,
        )
    )


providers = (coordinate_transformation_graph, add_coords)
