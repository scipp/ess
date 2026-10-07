# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from scippneutron.conversion import graph, tof
from scippnexus import NXsample, NXsource

from ..reflectometry.conversions import (
    reflectometry_q,
    reflectometry_q_x,
    reflectometry_q_z,
)
from ..reflectometry.types import (
    CoordTransformationGraph,
    RunType,
    SampleRun,
)
from .types import (
    DownstreamSlitCenters,
    SampleSurfaceNormal,
    UpstreamSlitCenters,
)


def outgoing_direction(
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> sc.Variable:
    """Unit direction of the outgoing ray at the sample, corrected for gravity.

    Approximate the flight time using the straight sample-to-detector distance,
    as in ScippNeutron's gravity correction.
    """
    flight_time = tof.tof_from_wavelength(
        wavelength=wavelength, Ltotal=sc.norm(scattered_beam)
    ).to(unit='s')
    outgoing_beam = scattered_beam - (0.5 * gravity * flight_time**2).to(
        unit=scattered_beam.unit
    )
    return outgoing_beam / sc.norm(outgoing_beam)


def scattering_angle(outgoing_direction: sc.Variable) -> sc.Variable:
    """Signed elevation of the outgoing ray above the laboratory x-z plane."""
    return sc.asin(outgoing_direction.fields.y)


def theta(
    outgoing_direction: sc.Variable,
    sample_surface_normal: sc.Variable,
) -> sc.Variable:
    """Specular reflection angle between the outgoing ray and the sample surface.

    Use the full three-dimensional direction. Under specular reflection this
    also determines the incidence angle, without requiring the incoming ray.
    """
    normal = sample_surface_normal / sc.norm(sample_surface_normal)
    return sc.asin(sc.dot(outgoing_direction, normal))


def incident_beam_directions(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
) -> dict[str, sc.Variable]:
    """Compute candidate incident directions from corresponding slit openings.

    The two inputs map the names of open channels to their centers in the
    upstream and downstream slit assemblies. Matching keys describe one
    possible incident beam.
    """
    if upstream_slit_centers.keys() != downstream_slit_centers.keys():
        raise ValueError(
            'UpstreamSlitCenters and DownstreamSlitCenters must have the same keys.'
        )
    directions = {}
    for key, upstream in upstream_slit_centers.items():
        beam = downstream_slit_centers[key] - upstream
        directions[key] = beam / sc.norm(beam)
    return directions


def assign_incident_angle(
    reflection_angle: sc.Variable,
    expected_reflection_angles: dict[str, sc.Variable],
) -> sc.Variable:
    """Assign incidence using the closest expected specular reflection angle."""
    angles = iter(expected_reflection_angles.values())
    closest = next(angles)
    if len(expected_reflection_angles) == 1:
        return closest

    min_distance = sc.abs(reflection_angle - closest)
    for angle in angles:
        distance = sc.abs(reflection_angle - angle)
        is_closer = distance < min_distance
        closest = sc.where(is_closer, angle, closest)
        min_distance = sc.where(is_closer, distance, min_distance)
    return closest


def incident_angle(
    incident_direction: sc.Variable,
    sample_surface_normal: sc.Variable,
) -> sc.Variable:
    """Angle at which the assigned incident beam approaches the sample."""
    normal = sample_surface_normal / sc.norm(sample_surface_normal)
    return sc.asin(-sc.dot(incident_direction, normal))


def coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    gravity: GravityVector,
) -> CoordTransformationGraph[RunType]:
    """Build the scattering coordinates shared by sample and direct-beam runs."""
    length = sc.norm(sample_surface_normal)
    if (
        not sc.isfinite(length).value
        or not (length > sc.scalar(0.0, unit=length.unit)).value
    ):
        raise ValueError('SampleSurfaceNormal must be a finite, nonzero vector.')
    return {
        **graph.beamline.L1(),
        **graph.beamline.L2(),
        'outgoing_direction': outgoing_direction,
        'scattering_angle': scattering_angle,
        'source_position': lambda: source_position,
        'sample_position': lambda: sample_position,
        'sample_surface_normal': lambda: sample_surface_normal,
        'gravity': lambda: gravity,
    }


def sample_coordinate_transformation_graph(
    source_position: Position[NXsource, SampleRun],
    sample_position: Position[NXsample, SampleRun],
    sample_surface_normal: SampleSurfaceNormal[SampleRun],
    gravity: GravityVector,
) -> CoordTransformationGraph[SampleRun]:
    """Extend the scattering graph with the sample's reflection angle and Q."""
    return coordinate_transformation_graph(
        source_position, sample_position, sample_surface_normal, gravity
    ) | {'theta': theta, 'Q': reflectometry_q}


def offspecular_sample_coordinate_transformation_graph(
    source_position: Position[NXsource, SampleRun],
    sample_position: Position[NXsample, SampleRun],
    sample_surface_normal: SampleSurfaceNormal[SampleRun],
    gravity: GravityVector,
    upstream_slit_centers: UpstreamSlitCenters[SampleRun],
    downstream_slit_centers: DownstreamSlitCenters[SampleRun],
) -> CoordTransformationGraph[SampleRun]:
    """Assign incidence by comparing measured and expected reflection angles."""
    transformation_graph = coordinate_transformation_graph(
        source_position, sample_position, sample_surface_normal, gravity
    )
    directions = incident_beam_directions(
        upstream_slit_centers, downstream_slit_centers
    )
    expected_reflection_angles = {
        key: incident_angle(direction, sample_surface_normal)
        for key, direction in directions.items()
    }

    def select_incident_angle(
        reflection_angle: sc.Variable,
    ) -> sc.Variable:
        return assign_incident_angle(reflection_angle, expected_reflection_angles)

    return transformation_graph | {
        'incident_angle': select_incident_angle,
        'reflection_angle': theta,
        'Qx': reflectometry_q_x,
        'Qz': reflectometry_q_z,
    }


def add_coords(
    da: sc.DataArray,
    graph: dict,
) -> sc.DataArray:
    """Add the scattering coordinates provided by the run's transformation graph."""
    return da.transform_coords(rename_dims=False, **graph)


providers = (
    coordinate_transformation_graph,
    sample_coordinate_transformation_graph,
)
