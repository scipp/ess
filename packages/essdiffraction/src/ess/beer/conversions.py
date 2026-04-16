import scipp as sc
import scipp.constants
from scippneutron.conversion import graph

from ess.powder.types import ElasticCoordTransformGraph, RunType

from .types import (
    DHKLList,
    GeometryCoordTransformGraph,
    ModulationPeriod,
    PulseLength,
    RawDetector,
    StreakClusteredData,
    WavelengthDefinitionChopperDelay,
    WavelengthDetector,
)


def compute_wavelength_in_each_cluster(
    da: StreakClusteredData[RunType],
    chopper_delay: WavelengthDefinitionChopperDelay,
    mod_period: ModulationPeriod,
    graph: GeometryCoordTransformGraph,
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
    4. Finally, round the estimated t0 to the closest known chopper opening time.
    """
    if isinstance(da, sc.DataGroup):
        return sc.DataGroup(
            {
                k: compute_wavelength_in_each_cluster(v, mod_period)
                for k, v in da.items()
            }
        )

    max_distance_from_streak_line = mod_period / 3
    sin_theta_L = sc.sin(da.bins.coords['two_theta'] / 2) * da.bins.coords['Ltotal']
    t = time_of_arrival(
        da.bins.coords['event_time_offset'],
        da.bins.coords['frame_cutoff_time'],
    )
    for _ in range(15):
        s, t0 = _linear_regression_by_bin(sin_theta_L, t, da.data)

        # Distance from point to line through cluster
        distance_to_self = sc.abs(sc.values(t0) + sc.values(s) * sin_theta_L - t)

        da = da.bins.assign_masks(
            too_far_from_center=(distance_to_self > max_distance_from_streak_line),
        )

    # The t0 estimate from fitting is influenced by peak overlap, background,
    # and other factors that can make the estimate offset from the true
    # chopper opening time that it should match.
    # We know the true chopper opening times, so instead of using the t0 estimte
    # directly we can round the estimate to the closest chopper opening time.
    # That way the t0 estimate becomes more robust and is guaranteed to correspond to
    # a true chopper opening time.
    t0 = _round_t0_to_nearest_chopper_opening(sc.values(t0), mod_period, chopper_delay)
    da = da.assign_coords(t0=t0)
    da = da.bins.assign_coords(tof=(t - t0))
    return da


def _round_t0_to_nearest_chopper_opening(
    t0: sc.Variable,
    mod_period: sc.Variable,
    chopper_delay: sc.Variable,
) -> sc.Variable:
    out = t0 - chopper_delay
    out /= mod_period
    out += 0.5
    sc.floor(out, out=out)
    out *= mod_period
    out += chopper_delay
    return out


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
    moderator_to_detector_distance: sc.Variable,
) -> sc.Variable:
    """Determines the ``d_hkl`` peak each event belongs to,
    given a list of known peaks."""
    # Source: https://www.mcstas.org/download/components/current/contrib/NPI_tof_dhkl_detector.comp
    sinth = sc.sin(theta)
    t = time_of_arrival

    d = sc.full_like(
        time_of_arrival, value=float('nan'), unit=dhkl_list[0].unit, dtype='float64'
    )
    dtfound = sc.full_like(time_of_arrival, value=float('nan'), dtype='float64')

    const = (
        2
        * sinth
        * moderator_to_detector_distance
        / (scipp.constants.h / scipp.constants.m_n)
    ).to(unit=f'{time_of_arrival.unit}/angstrom')
    for dhkl in dhkl_list:
        dt = sc.abs(t - dhkl * const)
        dt_in_range = dt < pulse_length / 2
        no_dt_found = sc.isnan(dtfound)
        dtfound = sc.where(dt_in_range, sc.where(no_dt_found, dt, dtfound), dtfound)
        d = sc.where(
            dt_in_range,
            sc.where(no_dt_found, dhkl, sc.scalar(float('nan'), unit=dhkl.unit)),
            d,
        )

    return d


def time_of_arrival(
    event_time_offset: sc.Variable,
    frame_cutoff_time: sc.Variable,
):
    """Does frame unwrapping for pulse shaping chopper modes.

    Events before the "cutoff time" are assumed to come from the previous pulse."""
    _eto = event_time_offset
    T = sc.scalar(1 / 14, unit='s').to(unit=_eto.unit)
    tc = frame_cutoff_time.to(unit=_eto.unit)
    return sc.where(_eto >= tc, _eto, _eto + T)


def _tof_from_dhkl(
    time_of_arrival: sc.Variable,
    theta: sc.Variable,
    coarse_dhkl: sc.Variable,
    Ltotal: sc.Variable,
    mod_period: sc.Variable,
    chopper_delay: sc.Variable,
) -> sc.Variable:
    """Computes tof for BEER given the dhkl peak that the event belongs to"""
    # Source: https://www.mcstas.org/download/components/current/contrib/NPI_tof_dhkl_detector.comp
    # tref = 2 * d_hkl * sin(theta) / hm * Ltotal
    # tc = time_of_arrival - chopper_delay - tref
    # dt = floor(tc / mod_period + 0.5) * mod_period + chopper_delay
    # tof = time_of_arrival - dt
    c = (-2 * 1.0 / (scipp.constants.h / scipp.constants.m_n)).to(
        unit=f'{time_of_arrival.unit}/m/angstrom'
    )
    out = c * coarse_dhkl
    out *= sc.sin(theta)
    out *= Ltotal
    out += time_of_arrival
    out -= chopper_delay
    out /= mod_period
    out += 0.5
    sc.floor(out, out=out)
    out *= mod_period
    out += chopper_delay
    out *= -1
    out += time_of_arrival
    return out


def t0_estimate(
    wavelength_estimate: sc.Variable,
    source_to_wavelength_definition_chopper_distance: sc.Variable,
) -> sc.Variable:
    """
    Computes the time a neutron reaches a chopper at
    ``source_to_wavelength_chopper_distance`` distance from the source
    if it has wavelength ``wavelength_estimate``.
    """
    return (
        sc.constants.m_n
        / sc.constants.h
        * wavelength_estimate
        * source_to_wavelength_definition_chopper_distance.to(
            unit=wavelength_estimate.unit
        )
    ).to(unit='s')


def tof_from_t0_estimate_graph(
    da: RawDetector[RunType],
    gg: GeometryCoordTransformGraph,
) -> ElasticCoordTransformGraph[RunType]:
    """Graph for computing ``wavelength`` in pulse shaping chopper modes."""
    return {
        **gg,
        't0': t0_estimate,
        'tof': lambda time_of_arrival, t0: time_of_arrival - t0,
        'time_of_arrival': time_of_arrival,
    }


def geometry_graph() -> GeometryCoordTransformGraph:
    return {
        **graph.beamline.beamline(scatter=True),
        **graph.tof.elastic("tof"),
    }


def tof_from_known_dhkl_graph(
    da: RawDetector[RunType],
    mod_period: ModulationPeriod,
    pulse_length: PulseLength,
    chopper_delay: WavelengthDefinitionChopperDelay,
    dhkl_list: DHKLList,
    gg: GeometryCoordTransformGraph,
) -> ElasticCoordTransformGraph[RunType]:
    """Graph computing ``tof`` in modulation chopper modes using
    list of peak positions."""

    def _compute_coarse_dspacing(
        time_of_arrival: sc.Variable,
        theta: sc.Variable,
        pulse_length: sc.Variable,
        moderator_to_detector_distance: sc.Variable,
    ):
        """To capture dhkl_list, otherwise it causes an error when
        ``.transform_coords`` is called unless it is called with
        ``keep_indermediates=False``.
        The error happens because data arrays do not allow coordinates
        with dimensions not present on the data.
        """
        return _compute_d_given_list_of_peaks(
            time_of_arrival=time_of_arrival,
            theta=theta,
            pulse_length=pulse_length,
            moderator_to_detector_distance=moderator_to_detector_distance,
            dhkl_list=dhkl_list,
        )

    return {
        **gg,
        'pulse_length': lambda: pulse_length,
        'mod_period': lambda: mod_period,
        'chopper_delay': lambda: chopper_delay,
        'tof': _tof_from_dhkl,
        'time_of_arrival': time_of_arrival,
        'coarse_dhkl': _compute_coarse_dspacing,
        'theta': lambda two_theta: two_theta / 2,
    }


def wavelength_detector(
    da: RawDetector[RunType], graph: ElasticCoordTransformGraph[RunType]
) -> WavelengthDetector[RunType]:
    """Applies the transformation graph to compute ``wavelength``."""
    return da.transform_coords(('wavelength',), graph=graph)


convert_from_known_peaks_providers = (
    geometry_graph,
    tof_from_known_dhkl_graph,
    wavelength_detector,
)
convert_pulse_shaping = (
    geometry_graph,
    tof_from_t0_estimate_graph,
    wavelength_detector,
)
providers = (compute_wavelength_in_each_cluster, geometry_graph)
