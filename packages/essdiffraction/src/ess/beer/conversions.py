import scipp as sc
import scipp.constants
import scippnexus as snx
from ess.powder.conversion import powder_coordinate_transformation_graph
from ess.powder.types import ElasticCoordTransformGraph, GravityVector, RunType
from scippneutron.conversion.tof import wavelength_from_tof

from ess.reduce.nexus.types import DiskChoppers, Position
from ess.reduce.unwrap.lut import chopper_distance_along_beam
from ess.reduce.unwrap.types import DetectorLtotal

from .types import (
    DHKLList,
    PulseLength,
    RawDetector,
    StreakClusteredData,
    WavelengthDetector,
)


def compute_wavelength_in_each_cluster(
    da: StreakClusteredData[RunType],
    choppers: DiskChoppers[RunType],
) -> WavelengthDetector[RunType]:
    """Fits a line through each cluster, the intercept of the line is t0.
    The line is fitted using linear regression with an outlier removal procedure.

    The algorithm is:

    1. Use least squares to fit line through clusters.
    2. Mask points that are "outliers" based on the criteria that they are too far
       from the line in the ``t`` variable.
       This means they don't seem to have the same time of flight origin as the rest
       of the points in the cluster, and probably should belong to another cluster or
       are part of the background.
    3. Go back to 1) and iterate until convergence. A few iterations should be enough.
    4. Finally, round the estimated dt to the closest known chopper opening time.
    """
    mod_period = _modulation_period(choppers)
    max_distance_from_streak_line = mod_period / 3
    sin_theta_L = sc.sin(da.bins.coords['two_theta'] / 2) * da.bins.coords['Ltotal']
    # Time from center of chopper opening "window".
    # The window contains all modulation sub pulses.
    t = da.bins.coords['tof']
    for _ in range(15):
        # dt is the difference between the the modulation chopper opening time
        # and the time center of the chopper opening window.
        s, dt = _linear_regression_by_bin(sin_theta_L, t, da.data)

        # Distance from point to line through cluster
        distance_to_self = sc.abs(sc.values(dt) + sc.values(s) * sin_theta_L - t)

        da = da.bins.assign_masks(
            too_far_from_center=(distance_to_self > max_distance_from_streak_line),
        )

    # The dt estimate from fitting is influenced by peak overlap, background,
    # and other factors that can make the estimate worse.
    # We know the true chopper opening times,
    # so we can round the estimate to the closest known time.
    dt = sc.values(dt)
    dt /= mod_period
    dt += 0.5
    sc.floor(dt, out=dt)
    dt *= mod_period
    # Remove the offset
    t -= dt
    da.bins.coords['wavelength'] = wavelength_from_tof(
        tof=t, Ltotal=da.bins.coords['Ltotal']
    )
    del da.bins.coords['tof']
    return da


def _modulation_period(choppers: DiskChoppers[RunType]) -> sc.Variable:
    periods = [
        1 / (sc.abs(chopper.frequency) * chopper.slit_begin.sizes['cutout'])
        for name, chopper in choppers.items()
        if not name.startswith('FC')
    ]
    return max(periods)


def _modulation_chopper_position(
    choppers: DiskChoppers[RunType],
) -> sc.Variable:
    return sc.concat(
        [
            chopper.axle_position
            for name, chopper in choppers.items()
            if not name.startswith('FC')
        ],
        dim='chopper',
    ).mean()


def _modulation_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
    detector_ltotal: DetectorLtotal[RunType],
    choppers: DiskChoppers[RunType],
) -> ElasticCoordTransformGraph[RunType]:
    """Use chopper-to-detector distance as ``Ltotal`` in modulation workflows."""
    graph = powder_coordinate_transformation_graph(
        source_position, sample_position, gravity
    )
    ltotal = detector_ltotal - chopper_distance_along_beam(
        _modulation_chopper_position(choppers), source_position
    )
    graph['Ltotal'] = lambda: ltotal
    return graph


def _linear_regression_by_bin(
    x: sc.Variable, y: sc.Variable, w: sc.Variable
) -> tuple[sc.Variable, sc.Variable]:
    """Performs a weighted linear regression of the points
    in the binned variables ``x`` and ``y`` weighted by ``w``.
    Returns ``b1`` and ``b0`` such that ``y = b1 * x + b0``.
    """
    w = sc.values(w)
    tot_w = w.bins.sum()

    avg_x = (w * x).bins.sum() / tot_w
    avg_y = (w * y).bins.sum() / tot_w

    cov_xy = (w * (x - avg_x) * (y - avg_y)).bins.sum() / tot_w
    var_x = (w * (x - avg_x) ** 2).bins.sum() / tot_w

    b1 = cov_xy / var_x
    b0 = avg_y - b1 * avg_x

    return b1, b0


def _compute_d_given_list_of_peaks(
    time_of_arrival: sc.Variable,
    theta: sc.Variable,
    dhkl_list: sc.Variable,
    pulse_length: sc.Variable,
    detector_ltotal: sc.Variable,
) -> sc.Variable:
    """Determines the ``d_hkl`` peak each event belongs to,
    given a list of known peaks."""
    # Source: https://www.mcstas.org/download/components/current/contrib/NPI_tof_dhkl_detector.comp
    sinth = sc.sin(theta)

    d = sc.full_like(
        time_of_arrival, value=float('nan'), unit=dhkl_list.unit, dtype='float64'
    )
    dtfound = sc.full_like(time_of_arrival, value=float('nan'), dtype='float64')

    const = (
        2 * sinth * detector_ltotal / (scipp.constants.h / scipp.constants.m_n)
    ).to(unit=f'{time_of_arrival.unit}/angstrom')
    for dhkl in dhkl_list:
        dt = sc.abs(time_of_arrival - dhkl * const - pulse_length / 2)
        dt_in_range = dt < pulse_length / 2
        no_dt_found = sc.isnan(dtfound)
        dtfound = sc.where(dt_in_range, sc.where(no_dt_found, dt, dtfound), dtfound)
        d = sc.where(
            dt_in_range,
            sc.where(no_dt_found, dhkl, sc.scalar(float('nan'), unit=dhkl.unit)),
            d,
        )

    return d


def modulation_time_of_arrival(
    event_time_offset: sc.Variable,
    Ltotal: sc.Variable,
    choppers: DiskChoppers[RunType],
) -> tuple[sc.Variable, sc.Variable]:
    """Unwrap detector event times for a BEER modulation mode.

    ``event_time_offset`` contains detector arrival times folded into one source
    period. The opening times of the modulation choppers and ``FC2A`` define the
    cutoff for shifting an event by one ``FC2A`` period so its time refers to the
    source pulse that produced it. Here, ``Ltotal`` is the distance from the
    modulation chopper to the detector.

    Returns
    -------
    time_of_arrival : sc.Variable
        Detector arrival time relative to the start of the source pulse that
        produced the event.
    tof_from_chopper : sc.Variable
        Approximate time of flight from the modulation chopper to the detector,
        obtained by subtracting the center time of the modulation-chopper opening.
    """

    def center_time(chopper):
        slit_center = (chopper.slit_begin[0] + chopper.slit_end[0]) / 2
        return chopper.time_offset_angle_at_beam(angle=slit_center).max()

    definition_choppers = [
        chopper for name, chopper in choppers.items() if not name.startswith('FC')
    ]

    nominal_time_at_chopper = sc.concat(
        [center_time(chopper) for chopper in definition_choppers],
        dim='chopper',
    ).mean()
    modulation_position = _modulation_chopper_position(choppers).fields.z

    frame_chopper = choppers['FC2A']
    inverse_velocity = (center_time(frame_chopper) - nominal_time_at_chopper) / (
        frame_chopper.axle_position.fields.z - modulation_position
    )
    period = (1 / sc.abs(frame_chopper.frequency)).to(unit=event_time_offset.unit)
    nominal_time_at_chopper = nominal_time_at_chopper.to(unit=event_time_offset.unit)
    cutoff = (
        nominal_time_at_chopper
        + (inverse_velocity * Ltotal.min()).to(unit=event_time_offset.unit)
        - period / 2
    )
    tof_from_source = sc.where(
        event_time_offset >= cutoff, event_time_offset, event_time_offset + period
    )
    return tof_from_source, tof_from_source - nominal_time_at_chopper


def automatic_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
    detector_ltotal: DetectorLtotal[RunType],
    choppers: DiskChoppers[RunType],
) -> ElasticCoordTransformGraph[RunType]:
    """Coordinate transformations for automatic modulation reduction."""
    graph = _modulation_coordinate_transformation_graph(
        source_position, sample_position, gravity, detector_ltotal, choppers
    )

    def tof(event_time_offset: sc.Variable, Ltotal: sc.Variable):
        return modulation_time_of_arrival(event_time_offset, Ltotal, choppers)[1]

    graph.update(
        {
            'tof': tof,
            'wavelength': wavelength_from_tof,
        }
    )
    return graph


def _tof_from_dhkl(
    tof_from_chopper: sc.Variable,
    theta: sc.Variable,
    coarse_dhkl: sc.Variable,
    Ltotal: sc.Variable,
    mod_period: sc.Variable,
) -> sc.Variable:
    """Computes tof for BEER given the dhkl peak that the event belongs to"""
    # Source: https://www.mcstas.org/download/components/current/contrib/NPI_tof_dhkl_detector.comp
    # tref = 2 * d_hkl * sin(theta) / hm * Ltotal
    # tc = tof_from_chopper - tref
    # dt = floor(tc / mod_period + 0.5) * mod_period
    # tof = tof_from_chopper - dt
    c = (-2 * 1.0 / (scipp.constants.h / scipp.constants.m_n)).to(
        unit=f'{tof_from_chopper.unit}/m/angstrom'
    )
    out = c * coarse_dhkl
    out *= sc.sin(theta)
    out *= Ltotal
    out += tof_from_chopper
    out /= mod_period
    out += 0.5
    sc.floor(out, out=out)
    out *= mod_period
    out *= -1
    out += tof_from_chopper
    return out


def known_peaks_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
    detector_ltotal: DetectorLtotal[RunType],
    choppers: DiskChoppers[RunType],
    pulse_length: PulseLength,
    dhkl_list: DHKLList,
) -> ElasticCoordTransformGraph[RunType]:
    """Coordinate transformations for modulation with known peak positions."""
    graph = _modulation_coordinate_transformation_graph(
        source_position, sample_position, gravity, detector_ltotal, choppers
    )
    mod_period = _modulation_period(choppers)

    def modulation_times(event_time_offset: sc.Variable, Ltotal: sc.Variable):
        time_of_arrival, tof_from_chopper = modulation_time_of_arrival(
            event_time_offset, Ltotal, choppers
        )
        return {
            'time_of_arrival': time_of_arrival,
            'tof_from_chopper': tof_from_chopper,
        }

    graph.update(
        {
            'detector_ltotal': lambda: detector_ltotal,
            'pulse_length': lambda: pulse_length,
            'mod_period': lambda: mod_period,
            'dhkl_list': lambda: dhkl_list,
            'tof': _tof_from_dhkl,
            'wavelength': wavelength_from_tof,
            ('time_of_arrival', 'tof_from_chopper'): modulation_times,
            'coarse_dhkl': _compute_d_given_list_of_peaks,
            'theta': lambda two_theta: two_theta / 2,
        }
    )
    return graph


def wavelength_detector(
    da: RawDetector[RunType], graph: ElasticCoordTransformGraph[RunType]
) -> WavelengthDetector[RunType]:
    """Compute wavelength using the workflow's coordinate transformations."""
    return da.transform_coords('wavelength', graph=graph, keep_intermediate=False)
