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
    IncidentAngleGraph,
    SampleSurfaceNormal,
    SlitStatusLogs,
    UpstreamSlitCenters,
)


def time_at_slit(
    event_time_zero: sc.Variable,
    event_time_offset: sc.Variable,
    wavelength: sc.Variable,
    Ltotal: sc.Variable,
    source_position: sc.Variable,
    slit_position: sc.Variable,
) -> sc.Variable:
    """Estimate absolute event times at the slit's distance from the source.

    Back-propagate from detector arrival using the reconstructed wavelength.
    This retains source emission delays and does not assume that the neutron
    left the source at the pulse timestamp.

    Parameters
    ----------
    event_time_zero:
        Absolute pulse timestamps associated with detector events.
    event_time_offset:
        Detector arrival offsets relative to those timestamps.
    wavelength:
        Reconstructed neutron wavelengths.
    Ltotal:
        Total source-to-detector flight paths.
    source_position:
        Source position in global coordinates.
    slit_position:
        Fixed slit position in the same global frame as the source.

    Returns
    -------
    :
        Absolute timestamps at the source-to-slit distance.
    """
    source_to_slit_distance = sc.norm(
        slit_position.to(unit=source_position.unit) - source_position
    ).to(unit=Ltotal.unit)
    slit_to_detector_flight_time = tof.tof_from_wavelength(
        wavelength=wavelength, Ltotal=Ltotal - source_to_slit_distance
    ).to(unit='ns', dtype='float64')
    detector_arrival_offset = event_time_offset.to(unit='ns', dtype='float64')
    slit_crossing_offset = sc.round(
        detector_arrival_offset - slit_to_detector_flight_time
    ).astype('int64')
    return event_time_zero.to(unit='ns') + slit_crossing_offset


def assign_incident_angle_from_slit_logs(
    time_at_slit: sc.Variable,
    incident_angles: dict[str, sc.Variable],
    slit_status_logs: dict[str, sc.DataArray],
) -> sc.Variable:
    """Assign the incident angle when all slit states are known and one is open.

    All logs are evaluated at the same neutron crossing time, using a common
    approximate slit position. Unknown status, no open slit or multiple open slits
    yields NaN. The measured reflection angle does not enter this assignment.
    """
    unassigned_incident_angle = sc.scalar(float('nan'), unit='rad')
    assigned_incident_angle = unassigned_incident_angle
    open_slit_count = sc.scalar(0, dtype='int32', unit=None)
    all_slit_states_known = sc.scalar(True)
    # Encode unknown separately from the boolean states: -1 unknown, 0 closed, 1 open.
    slit_crossing_time = time_at_slit.to(unit='ns')
    for beam_name, beam_incident_angle in incident_angles.items():
        slit_log = slit_status_logs[beam_name]
        unknown_slit_state = sc.scalar(-1, dtype='int32', unit=slit_log.unit)
        open_slit_state = sc.scalar(1, dtype='int32', unit=slit_log.unit)
        slit_state = sc.lookup(
            slit_log.astype('int32').assign_coords(
                time=slit_log.coords['time'].to(unit='ns')
            ),
            dim='time',
            mode='previous',
            fill_value=unknown_slit_state,
        )(slit_crossing_time)
        all_slit_states_known = all_slit_states_known & (
            slit_state != unknown_slit_state
        )
        slit_is_open = slit_state == open_slit_state
        assigned_incident_angle = sc.where(
            slit_is_open,
            beam_incident_angle.to(unit='rad', dtype='float64'),
            assigned_incident_angle,
        )
        open_slit_count = open_slit_count + slit_is_open.astype('int32')
    exactly_one_slit_open = open_slit_count == sc.scalar(1, unit=None)
    return sc.where(
        all_slit_states_known & exactly_one_slit_open,
        assigned_incident_angle,
        unassigned_incident_angle,
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


def _incident_angles_from_slits(
    upstream_slit_centers, downstream_slit_centers, sample_surface_normal
):
    return {
        beam_name: incident_angle(direction, sample_surface_normal)
        for beam_name, direction in incident_beam_directions(
            upstream_slit_centers, downstream_slit_centers
        ).items()
    }


def closest_beam_incident_angle_graph(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
) -> IncidentAngleGraph[RunType]:
    """Provide incidence assignment using the closest expected reflection angle."""
    beam_incident_angles = _incident_angles_from_slits(
        upstream_slit_centers, downstream_slit_centers, sample_surface_normal
    )

    def select_incident_angle(reflection_angle: sc.Variable) -> sc.Variable:
        return assign_incident_angle(reflection_angle, beam_incident_angles)

    return {'incident_angle': select_incident_angle}


def slit_log_incident_angle_graph(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    slit_status_logs: SlitStatusLogs[RunType],
) -> IncidentAngleGraph[RunType]:
    """Provide one crossing-time coordinate and use it to look up all slit logs."""
    beam_incident_angles = _incident_angles_from_slits(
        upstream_slit_centers, downstream_slit_centers, sample_surface_normal
    )

    slit_position = sc.mean(
        sc.concat(list(downstream_slit_centers.values()), dim='slit')
    )

    def select_incident_angle(time_at_slit: sc.Variable) -> sc.Variable:
        return assign_incident_angle_from_slit_logs(
            time_at_slit, beam_incident_angles, slit_status_logs
        )

    return {
        'slit_position': lambda: slit_position,
        'time_at_slit': time_at_slit,
        'incident_angle': select_incident_angle,
    }


def offspecular_coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    gravity: GravityVector,
    incident_angle_graph: IncidentAngleGraph[RunType],
) -> CoordTransformationGraph[RunType]:
    """Combine an incidence assignment with the shared offspecular coordinates."""
    return (
        coordinate_transformation_graph(
            source_position, sample_position, sample_surface_normal, gravity
        )
        | incident_angle_graph
        | {
            'reflection_angle': theta,
            'Qx': reflectometry_q_x,
            'Qz': reflectometry_q_z,
        }
    )


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
