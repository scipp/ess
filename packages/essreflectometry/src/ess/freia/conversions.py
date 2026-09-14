# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from scippneutron.conversion import beamline, graph
from scippnexus import NXsample, NXsource

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import CoordTransformationGraph, RunType, WavelengthDetector
from .types import QDetector, SampleSurfaceNormal


def theta(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
    sample_surface_normal: sc.Variable,
) -> sc.Variable:
    """Signed, gravity-corrected exit angle above the sample plane.

    ScippNeutron reconstructs the outgoing direction at the sample. Project
    that direction onto the sample normal to retain the sign and support a
    tilted sample. Its reflectometry-specific scattering_angle_in_yz_plane
    returns an unsigned angle, which cannot distinguish the direct beam.

    The horizontal beam direction only defines a coordinate basis here; it
    does not specify an incident angle. For Q and footprint we still assume
    specular reflection, so the incidence angle equals this exit angle.
    """
    basis = beamline.beam_aligned_unit_vectors(incident_beam, gravity)
    x, y, z = (basis[f'beam_aligned_unit_{axis}'] for axis in 'xyz')
    # Use the horizontal reference axis: the source-to-sample line in FREIA
    # is tilted and does not describe the incident direction at the sample.
    angles = beamline.scattering_angles_with_gravity(
        incident_beam=z * sc.scalar(1.0, unit='m'),
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    polar, azimuth = angles['two_theta'], angles['phi']
    normal = sample_surface_normal / sc.norm(sample_surface_normal)
    return sc.asin(
        sc.sin(polar)
        * (sc.dot(normal, x) * sc.cos(azimuth) + sc.dot(normal, y) * sc.sin(azimuth))
        + sc.dot(normal, z) * sc.cos(polar)
    )


def coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    gravity: GravityVector,
) -> CoordTransformationGraph[RunType]:
    """Build a specular conversion graph independent of detector pixel layout."""
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
    """Add Q without requiring an ROI, monitor, reference, or footprint inputs."""
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
