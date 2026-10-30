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
    ShutterPositions,
    ShutterStatusLogs,
    UpstreamSlitCenters,
)


def time_at_detector(
    event_time_zero: sc.Variable,
    event_time_offset: sc.Variable,
) -> sc.Variable:
    """Detector arrival as float64 seconds since the Unix epoch."""
    pulse_time = event_time_zero.to(unit='ns', copy=False) - sc.epoch(unit='ns')
    return pulse_time.to(unit='s', dtype='float64') + event_time_offset.to(
        unit='s', dtype='float64', copy=False
    )


def time_at_shutter(
    time_at_detector: sc.Variable,
    wavelength: sc.Variable,
    L2: sc.Variable,
    sample_position: sc.Variable,
    shutter_position: sc.Variable,
) -> sc.Variable:
    """Estimate the time the event passed the shutter position.

    Parameters
    ----------
    time_at_detector:
        Detector arrival as float64 seconds since the Unix epoch.
    wavelength:
        Reconstructed neutron wavelengths.
    L2:
        Sample-to-detector flight paths.
    sample_position:
        Sample position in global coordinates.
    shutter_position:
        Fixed shutter position in the same global frame as the sample.

    Returns
    -------
    :
        Time the neutron passed the shutter, as float64 seconds since the Unix epoch.
    """
    # Approximate the path by straight segments from shutter to sample to detector.
    shutter_to_sample_distance = sc.norm(
        shutter_position.to(unit=sample_position.unit, copy=False) - sample_position
    ).to(unit=L2.unit)
    shutter_to_detector_flight_time = tof.tof_from_wavelength(
        wavelength=wavelength, Ltotal=shutter_to_sample_distance + L2
    ).to(unit='s', copy=False)
    return time_at_detector - shutter_to_detector_flight_time


def _shutter_state_at_time(
    crossing_time: sc.Variable, shutter_log: sc.DataArray
) -> sc.Variable:
    """Look up one shutter's state: NaN unknown, 0 closed, 1 open."""
    log_time = shutter_log.coords['time'].to(unit='ns', copy=False) - sc.epoch(
        unit='ns'
    )
    return sc.lookup(
        shutter_log.astype('float32').assign_coords(
            time=log_time.to(unit='s', dtype='float64')
        ),
        dim='time',
        mode='previous',
    )(crossing_time)


def assign_incident_angle_from_shutter_logs(
    event_time_zero: sc.Variable,
    event_time_offset: sc.Variable,
    wavelength: sc.Variable,
    L2: sc.Variable,
    sample_position: sc.Variable,
    *,
    incident_angles: dict[str, sc.Variable],
    shutter_positions: dict[str, sc.Variable],
    shutter_status_logs: dict[str, sc.DataArray],
) -> sc.Variable:
    """Assign incidence when all shutter states are known and exactly one is open.

    Each beam's log is evaluated at the neutron's crossing time at that beam's
    shutter. Process shutters sequentially, discarding each crossing-time array
    after looking up its status. Keys in all three dictionaries must match.
    An unknown state, no open shutter, or multiple open shutters yields NaN.
    """
    if not (
        shutter_positions.keys() == incident_angles.keys() == shutter_status_logs.keys()
    ):
        raise ValueError(
            'Shutter positions, incident angles and logs must have the same keys.'
        )

    detector_time = time_at_detector(event_time_zero, event_time_offset)
    unassigned_incident_angle = sc.scalar(float('nan'), unit='rad')
    assigned_incident_angle = unassigned_incident_angle
    open_shutter_count = sc.scalar(0.0, dtype='float32', unit=None)
    for beam_name, beam_incident_angle in incident_angles.items():
        shutter_log = shutter_status_logs[beam_name]
        shutter_state = _shutter_state_at_time(
            time_at_shutter(
                detector_time,
                wavelength,
                L2,
                sample_position,
                shutter_positions[beam_name],
            ),
            shutter_log,
        )
        shutter_is_open = shutter_state == sc.scalar(1.0, unit=shutter_log.unit)
        assigned_incident_angle = sc.where(
            shutter_is_open,
            beam_incident_angle.to(unit='rad', dtype='float64'),
            assigned_incident_angle,
        )
        # Unknown states propagate NaN, so they cannot yield exactly one open shutter.
        open_shutter_count = open_shutter_count + shutter_state
        # Release per-shutter temporaries before calculating the next crossing time.
        del shutter_state, shutter_is_open

    return sc.where(
        open_shutter_count == sc.scalar(1.0, unit=None),
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
    also determines the incidence angle.
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


def shutter_log_incident_angle_graph(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    shutter_positions: ShutterPositions[RunType],
    shutter_status_logs: ShutterStatusLogs[RunType],
) -> IncidentAngleGraph[RunType]:
    """Identify candidate beams using each shutter's position and status log."""
    beam_incident_angles = _incident_angles_from_slits(
        upstream_slit_centers, downstream_slit_centers, sample_surface_normal
    )

    if not beam_incident_angles:
        raise ValueError('At least one incident beam is required.')
    if not (
        beam_incident_angles.keys()
        == shutter_positions.keys()
        == shutter_status_logs.keys()
    ):
        raise ValueError(
            'Slit centers, ShutterPositions and ShutterStatusLogs '
            'must have the same keys.'
        )

    def select_incident_angle(
        event_time_zero: sc.Variable,
        event_time_offset: sc.Variable,
        wavelength: sc.Variable,
        L2: sc.Variable,
        sample_position: sc.Variable,
    ) -> sc.Variable:
        return assign_incident_angle_from_shutter_logs(
            event_time_zero,
            event_time_offset,
            wavelength,
            L2,
            sample_position,
            incident_angles=beam_incident_angles,
            shutter_positions=shutter_positions,
            shutter_status_logs=shutter_status_logs,
        )

    return {'incident_angle': select_incident_angle}


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


providers = (
    coordinate_transformation_graph,
    sample_coordinate_transformation_graph,
)
