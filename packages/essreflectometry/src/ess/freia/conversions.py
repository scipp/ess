# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from scipp.constants import h, m_n
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
from .mirror import no_mirror, reflected_upstream_position
from .types import (
    DownstreamSlitCenters,
    IncidentAngleGraph,
    MirrorGeometry,
    SampleSurfaceNormal,
    ShutterPositions,
    ShutterStatusLogs,
    UpstreamSlitCenters,
)


def time_at_detector(
    event_time_zero: sc.Variable,
    event_time_offset: sc.Variable,
) -> sc.Variable:
    """Detector arrival as float64 seconds since the Unix epoch.

    For timestamps around 2026, float64 spacing is about 0.24 microseconds,
    well below the hundreds of microseconds needed for shutter timing.
    """
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
        In inversion mode, use its reflection across the mirror plane so the
        distance to the sample includes the path through the mirror.

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
    incident_gravity_lengths: dict[str, sc.Variable],
    shutter_positions: dict[str, sc.Variable],
    shutter_status_logs: dict[str, sc.DataArray],
) -> dict[str, sc.Variable]:
    """Assign nominal angle and gravity length when exactly one known shutter is open.

    Each beam's log is evaluated at the neutron's crossing time at that beam's
    shutter. Process shutters sequentially, discarding each crossing-time array
    after looking up its status. Keys in all dictionaries must match.
    An unknown state, no open shutter, or multiple open shutters yields NaN.
    """
    if not (
        shutter_positions.keys()
        == incident_angles.keys()
        == incident_gravity_lengths.keys()
        == shutter_status_logs.keys()
    ):
        raise ValueError(
            'Incident angles, gravity lengths, shutter positions and logs '
            'must have the same keys.'
        )

    detector_time = time_at_detector(event_time_zero, event_time_offset)
    unassigned_angle = sc.scalar(float('nan'), unit='rad')
    unassigned_length = sc.scalar(float('nan'), unit='m')
    assigned_angle, assigned_length = unassigned_angle, unassigned_length
    open_shutter_count = sc.scalar(0.0, dtype='float32', unit=None)
    for beam_name, angle in incident_angles.items():
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
        assigned_angle = sc.where(shutter_is_open, angle.to(unit='rad'), assigned_angle)
        assigned_length = sc.where(
            shutter_is_open,
            incident_gravity_lengths[beam_name].to(unit='m'),
            assigned_length,
        )
        # Unknown states propagate NaN, so they cannot yield exactly one open shutter.
        open_shutter_count = open_shutter_count + shutter_state
        # Release per-shutter temporaries before calculating the next crossing time.
        del shutter_state, shutter_is_open

    assigned = open_shutter_count == sc.scalar(1.0, unit=None)
    return {
        'nominal_incident_angle': sc.where(assigned, assigned_angle, unassigned_angle),
        'incident_gravity_length': sc.where(
            assigned, assigned_length, unassigned_length
        ),
    }


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


def assign_incident_angle(
    reflection_angle: sc.Variable,
    expected_reflection_angles: dict[str, sc.Variable],
    incident_gravity_lengths: dict[str, sc.Variable],
) -> dict[str, sc.Variable]:
    """Select the nominal incident angle closest to the geometrical reflection angle.

    ``reflection_angle`` is measured from the sample-to-pixel direction, without
    gravity correction. Return ``nominal_incident_angle`` and the corresponding
    ``incident_gravity_length`` together, for the subsequent gravity correction.
    """
    if expected_reflection_angles.keys() != incident_gravity_lengths.keys():
        raise ValueError('Incident angles and gravity lengths must have the same keys.')
    names = iter(expected_reflection_angles)
    first = next(names)
    closest = expected_reflection_angles[first]
    length = incident_gravity_lengths[first]
    if len(expected_reflection_angles) > 1:
        min_distance = sc.abs(reflection_angle - closest)
        for name in names:
            angle = expected_reflection_angles[name]
            distance = sc.abs(reflection_angle - angle)
            is_closer = distance < min_distance
            closest = sc.where(is_closer, angle, closest)
            length = sc.where(is_closer, incident_gravity_lengths[name], length)
            min_distance = sc.where(is_closer, distance, min_distance)
    return {'nominal_incident_angle': closest, 'incident_gravity_length': length}


def _incident_angle_and_gravity_length(
    upstream: sc.Variable,
    downstream: sc.Variable,
    normal: sc.Variable,
    mirror: MirrorGeometry | None,
    sample_position: sc.Variable,
    gravity: sc.Variable,
) -> tuple[sc.Variable, sc.Variable]:
    r"""Compute the nominal incident angle and signed effective gravity length.

    Assumes FREIA's layout: distinct slits, the inversion mirror strictly
    between them, and the sample downstream. Sample normal and gravity have
    arbitrary orientations. Geometry is evaluated once per candidate.

    Returns
    -------
    :
        Nominal incident angle in radians and the geometric coefficient of the
        angular gravity correction, in metres. Both are scalars. The coefficient
        combines flight lengths and orientations; positive values mean gravity
        increases the incident angle.

    Notes
    -----
    All lengths follow the ray calculated without gravity: :math:`L` is the
    path between slit centres, :math:`a` the upstream-slit-to-mirror path, and
    :math:`c` the downstream-slit-to-sample-plane path.
    Let :math:`u` be the unit incident direction at the sample, :math:`\alpha_0`
    its angle to the surface, :math:`n_s` the outward unit sample normal,
    :math:`n_m` the unit mirror normal, and :math:`g` gravitational acceleration.

    The vector :math:`K` weights gravity by these flight lengths. Projecting
    its effect onto the sample angle gives the returned coefficient :math:`D`:

    .. math::

        K = g(c+L/2) - n_m(n_m \cdot g)\frac{a^2}{L}, \qquad
        D = -\frac{K \cdot [n_s-u(u \cdot n_s)]}{|g|\cos\alpha_0}.

    Here :math:`|g|` is the acceleration magnitude. Without the mirror, omit the
    second term in :math:`K`. For zero gravity, return zero for the coefficient.
    """
    upstream, downstream, sample_position = (
        position.to(unit='m') for position in (upstream, downstream, sample_position)
    )
    reflected_upstream = reflected_upstream_position(upstream, downstream, mirror)
    beam = downstream - reflected_upstream
    length, g = sc.norm(beam), sc.norm(gravity)
    direction, normal = beam / length, normal / sc.norm(normal)
    normal_component = sc.dot(direction, normal)
    angle = sc.asin(-normal_component)
    if g.value == 0:
        return angle, sc.scalar(0.0, unit='m')

    transverse_normal = normal - direction * normal_component
    cosine = sc.norm(sc.cross(normal, direction))
    c = sc.dot(sample_position - downstream, normal) / normal_component
    gravity_direction = gravity / g
    k = gravity_direction * (c + length / 2)
    if mirror is not None:
        a = sc.dot(
            mirror.position.to(unit='m') - reflected_upstream, mirror.normal
        ) / sc.dot(direction, mirror.normal)
        k -= mirror.normal * sc.dot(gravity_direction, mirror.normal) * a**2 / length
    return angle, -sc.dot(k, transverse_normal) / cosine


def gravity_corrected_incident_angle(
    nominal_incident_angle: sc.Variable,
    incident_gravity_length: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> sc.Variable:
    r"""Correct the selected incident angle for gravity at each event's wavelength.

    Parameters
    ----------
    nominal_incident_angle:
        Incident angle determined from slit geometry, before gravity correction.
    incident_gravity_length:
        Signed geometric coefficient, in metres, combining the selected beam's
        flight lengths and mirror, sample, and gravity orientations. Positive
        values increase the incident angle.
    wavelength:
        Reconstructed neutron wavelengths.
    gravity:
        Gravitational acceleration vector in laboratory coordinates.
        A zero vector disables the correction.

    Returns
    -------
    :
        Incident angle at the sample, in radians.

    Notes
    -----
    The first-order correction is

    .. math::

        \alpha(\lambda) = \alpha_0 + |g|D\left(\frac{m_n\lambda}{h}\right)^2.

    Here :math:`\alpha_0` is ``nominal_incident_angle``, :math:`D` is
    ``incident_gravity_length``, :math:`\lambda` is ``wavelength``, and
    :math:`|g|` is the magnitude of ``gravity``. The constants :math:`m_n` and
    :math:`h` are the neutron mass and Planck's constant. The result
    :math:`\alpha(\lambda)` is the corrected angle in radians.
    """
    g = sc.norm(gravity)
    if g.value == 0:
        return nominal_incident_angle
    factor = g * (m_n / h) ** 2
    correction = (
        factor
        * incident_gravity_length
        * wavelength.to(dtype='float64', copy=False) ** 2
    ).to(unit='dimensionless')
    correction.unit = 'rad'
    return nominal_incident_angle + correction


def coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    gravity: GravityVector,
    mirror: MirrorGeometry[RunType] | None = None,
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
        'source_position': lambda: reflected_upstream_position(
            source_position, sample_position, mirror
        ),
        'sample_position': lambda: sample_position,
        'sample_surface_normal': lambda: sample_surface_normal,
        'gravity': lambda: gravity,
    }


def sample_coordinate_transformation_graph(
    source_position: Position[NXsource, SampleRun],
    sample_position: Position[NXsample, SampleRun],
    sample_surface_normal: SampleSurfaceNormal[SampleRun],
    gravity: GravityVector,
    mirror: MirrorGeometry[SampleRun] | None = None,
) -> CoordTransformationGraph[SampleRun]:
    """Extend the scattering graph with the sample's reflection angle and Q."""
    return coordinate_transformation_graph(
        source_position, sample_position, sample_surface_normal, gravity, mirror
    ) | {'theta': theta, 'Q': reflectometry_q}


def _incident_angle_parameters(
    upstream_slit_centers: dict[str, sc.Variable],
    downstream_slit_centers: dict[str, sc.Variable],
    sample_surface_normal: sc.Variable,
    mirror: MirrorGeometry | None,
    sample_position: sc.Variable,
    gravity: sc.Variable,
) -> tuple[dict[str, sc.Variable], dict[str, sc.Variable]]:
    """Compute a nominal angle and signed effective length for each slit pair."""
    if upstream_slit_centers.keys() != downstream_slit_centers.keys():
        raise ValueError(
            'UpstreamSlitCenters and DownstreamSlitCenters must have the same keys.'
        )
    if not upstream_slit_centers:
        raise ValueError('At least one incident beam is required.')
    angles, lengths = {}, {}
    for name, upstream in upstream_slit_centers.items():
        angles[name], lengths[name] = _incident_angle_and_gravity_length(
            upstream,
            downstream_slit_centers[name],
            sample_surface_normal,
            mirror,
            sample_position,
            gravity,
        )
    return angles, lengths


def fixed_slits_incident_angle_graph(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    sample_position: Position[NXsample, RunType],
    gravity: GravityVector,
    mirror: MirrorGeometry[RunType] | None = None,
) -> IncidentAngleGraph[RunType]:
    """Use one slit pair's nominal angle and effective length, then apply gravity."""
    angles, lengths = _incident_angle_parameters(
        upstream_slit_centers,
        downstream_slit_centers,
        sample_surface_normal,
        mirror,
        sample_position,
        gravity,
    )
    (angle,), (length,) = angles.values(), lengths.values()
    return {
        'nominal_incident_angle': lambda: angle,
        'incident_gravity_length': lambda: length,
        'incident_angle': gravity_corrected_incident_angle,
    }


def closest_beam_incident_angle_graph(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    sample_position: Position[NXsample, RunType],
    gravity: GravityVector,
    mirror: MirrorGeometry[RunType] | None = None,
) -> IncidentAngleGraph[RunType]:
    """Select the closest nominal angle using detector geometry, then apply gravity.

    Compare the sample-to-pixel angle with each candidate's nominal incident
    angle, without gravity correction. Select its effective length along with
    the angle for the subsequent wavelength-dependent correction.
    """
    if len(upstream_slit_centers) == 1:
        return fixed_slits_incident_angle_graph(
            upstream_slit_centers,
            downstream_slit_centers,
            sample_surface_normal,
            sample_position,
            gravity,
            mirror,
        )
    angles, lengths = _incident_angle_parameters(
        upstream_slit_centers,
        downstream_slit_centers,
        sample_surface_normal,
        mirror,
        sample_position,
        gravity,
    )

    def select_incident_angle(geometric_reflection_angle: sc.Variable):
        return assign_incident_angle(geometric_reflection_angle, angles, lengths)

    return {
        'geometric_reflection_angle': lambda scattered_beam: theta(
            scattered_beam / sc.norm(scattered_beam), sample_surface_normal
        ),
        ('nominal_incident_angle', 'incident_gravity_length'): select_incident_angle,
        'incident_angle': gravity_corrected_incident_angle,
    }


def shutter_log_incident_angle_graph(
    upstream_slit_centers: UpstreamSlitCenters[RunType],
    downstream_slit_centers: DownstreamSlitCenters[RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    shutter_positions: ShutterPositions[RunType],
    shutter_status_logs: ShutterStatusLogs[RunType],
    sample_position: Position[NXsample, RunType],
    gravity: GravityVector,
    mirror: MirrorGeometry[RunType] | None = None,
) -> IncidentAngleGraph[RunType]:
    """Assign nominal angle and length from shutter logs, then apply gravity."""
    angles, lengths = _incident_angle_parameters(
        upstream_slit_centers,
        downstream_slit_centers,
        sample_surface_normal,
        mirror,
        sample_position,
        gravity,
    )
    if not (angles.keys() == shutter_positions.keys() == shutter_status_logs.keys()):
        raise ValueError(
            'Slit centers, ShutterPositions and ShutterStatusLogs '
            'must have the same keys.'
        )
    reflected_shutter_positions = {
        key: reflected_upstream_position(position, sample_position, mirror)
        for key, position in shutter_positions.items()
    }

    def select_incident_angle(
        event_time_zero: sc.Variable,
        event_time_offset: sc.Variable,
        wavelength: sc.Variable,
        L2: sc.Variable,
        sample_position: sc.Variable,
    ):
        return assign_incident_angle_from_shutter_logs(
            event_time_zero,
            event_time_offset,
            wavelength,
            L2,
            sample_position,
            incident_angles=angles,
            incident_gravity_lengths=lengths,
            shutter_positions=reflected_shutter_positions,
            shutter_status_logs=shutter_status_logs,
        )

    return {
        ('nominal_incident_angle', 'incident_gravity_length'): select_incident_angle,
        'incident_angle': gravity_corrected_incident_angle,
    }


def offspecular_coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_surface_normal: SampleSurfaceNormal[RunType],
    gravity: GravityVector,
    incident_angle_graph: IncidentAngleGraph[RunType],
    mirror: MirrorGeometry[RunType] | None = None,
) -> CoordTransformationGraph[RunType]:
    """Combine an incidence assignment with the shared offspecular coordinates."""
    return (
        coordinate_transformation_graph(
            source_position,
            sample_position,
            sample_surface_normal,
            gravity,
            mirror,
        )
        | incident_angle_graph
        | {
            'reflection_angle': theta,
            'Qx': reflectometry_q_x,
            'Qz': reflectometry_q_z,
        }
    )


providers = (
    no_mirror,
    coordinate_transformation_graph,
    sample_coordinate_transformation_graph,
)
