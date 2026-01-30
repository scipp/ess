import scipp as sc
import scipp.constants
from scippneutron.conversion import graph

from .types import (
    DHKLList,
    GeometryCoordTransformGraph,
    ModulationPeriod,
    PulseLength,
    RawDetector,
    RunType,
    StreakClusteredData,
    TofCoordTransformGraph,
    TofDetector,
    WavelengthDefinitionChopperDelay,
)


def compute_tof_in_each_cluster(
    da: StreakClusteredData[RunType],
    mod_period: ModulationPeriod,
) -> TofDetector[RunType]:
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
    """
    if isinstance(da, sc.DataGroup):
        return sc.DataGroup(
            {k: compute_tof_in_each_cluster(v, mod_period) for k, v in da.items()}
        )

    max_distance_from_streak_line = mod_period / 3
    sin_theta_L = sc.sin(da.bins.coords['two_theta'] / 2) * da.bins.coords['Ltotal']
    t = time_of_arrival(
        da.bins.coords['event_time_offset'],
        da.coords['tc'].to(unit=da.bins.coords['event_time_offset'].unit),
    )
    for _ in range(15):
        s, t0 = _linear_regression_by_bin(sin_theta_L, t, da.data)

        # Distance from point to line through cluster
        distance_to_self = sc.abs(sc.values(t0) + sc.values(s) * sin_theta_L - t)

        da = da.bins.assign_masks(
            too_far_from_center=(distance_to_self > max_distance_from_streak_line),
        )

    da = da.assign_coords(t0=sc.values(t0))
    da = da.bins.assign_coords(tof=(t - sc.values(t0)))
    return da


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
    L0: sc.Variable,
) -> sc.Variable:
    """Determines the ``d_hkl`` peak each event belongs to,
    given a list of known peaks."""
    # Source: https://www2.mcstas.org/download/components/3.4/contrib/NPI_tof_dhkl_detector.comp
    sinth = sc.sin(theta)
    t = time_of_arrival

    d = sc.full_like(
        time_of_arrival, value=float('nan'), unit=dhkl_list[0].unit, dtype='float64'
    )
    dtfound = sc.full_like(time_of_arrival, value=float('nan'), dtype='float64')

    const = (2 * sinth * L0 / (scipp.constants.h / scipp.constants.m_n)).to(
        unit=f'{time_of_arrival.unit}/angstrom'
    )
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
    tc: sc.Variable,
):
    """Does frame unwrapping for pulse shaping chopper modes.

    Events before the "cutoff time" `tc` are assumed to come from the previous pulse."""
    _eto = event_time_offset
    T = sc.scalar(1 / 14, unit='s').to(unit=_eto.unit)
    tc = tc.to(unit=_eto.unit)
    return sc.where(_eto >= tc, _eto, _eto + T)


def _tof_from_dhkl(
    time_of_arrival: sc.Variable,
    theta: sc.Variable,
    coarse_dhkl: sc.Variable,
    Ltotal: sc.Variable,
    mod_period: sc.Variable,
    time0: sc.Variable,
) -> sc.Variable:
    """Computes tof for BEER given the dhkl peak that the event belongs to"""
    # Source: https://www2.mcstas.org/download/components/3.4/contrib/NPI_tof_dhkl_detector.comp
    # tref = 2 * d_hkl * sin(theta) / hm * Ltotal
    # tc = time_of_arrival - time0 - tref
    # dt = floor(tc / mod_period + 0.5) * mod_period + time0
    # tof = time_of_arrival - dt
    c = (-2 * 1.0 / (scipp.constants.h / scipp.constants.m_n)).to(
        unit=f'{time_of_arrival.unit}/m/angstrom'
    )
    out = c * coarse_dhkl
    out *= sc.sin(theta)
    out *= Ltotal
    out += time_of_arrival
    out -= time0
    out /= mod_period
    out += 0.5
    sc.floor(out, out=out)
    out *= mod_period
    out += time0
    out *= -1
    out += time_of_arrival
    return out


def geometry_graph() -> GeometryCoordTransformGraph:
    return graph.beamline.beamline(scatter=True)


def tof_from_known_dhkl_graph(
    mod_period: ModulationPeriod,
    pulse_length: PulseLength,
    time0: WavelengthDefinitionChopperDelay,
    dhkl_list: DHKLList,
    gg: GeometryCoordTransformGraph,
) -> TofCoordTransformGraph:
    """Graph computing ``tof`` in modulation chopper modes using
    list of peak positions."""

    def _compute_coarse_dspacing(
        time_of_arrival: sc.Variable,
        theta: sc.Variable,
        pulse_length: sc.Variable,
        L0: sc.Variable,
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
            L0=L0,
            dhkl_list=dhkl_list,
        )

    return {
        **gg,
        **graph.tof.elastic("tof"),
        'pulse_length': lambda: pulse_length,
        'mod_period': lambda: mod_period,
        'time0': lambda: time0,
        'tof': _tof_from_dhkl,
        'time_of_arrival': time_of_arrival,
        'coarse_dhkl': _compute_coarse_dspacing,
        'theta': lambda two_theta: two_theta / 2,
    }


def t0_estimate(
    wavelength_estimate: sc.Variable,
    L0: sc.Variable,
    Ltotal: sc.Variable,
) -> sc.Variable:
    """Estimates the time-at-chopper by assuming the wavelength."""
    return (
        sc.constants.m_n
        / sc.constants.h
        * wavelength_estimate
        * (L0 - Ltotal).to(unit=wavelength_estimate.unit)
    ).to(unit='s')


def _tof_from_t0(
    time_of_arrival: sc.Variable,
    t0: sc.Variable,
) -> sc.Variable:
    """Computes time-of-flight by subtracting a start time."""
    return time_of_arrival - t0


def tof_from_t0_estimate_graph(
    gg: GeometryCoordTransformGraph,
) -> TofCoordTransformGraph:
    """Graph for computing ``tof`` in pulse shaping chopper modes."""
    return {
        **gg,
        **graph.tof.elastic("tof"),
        't0': t0_estimate,
        'tof': _tof_from_t0,
        'time_of_arrival': time_of_arrival,
    }


def compute_tof(
    da: RawDetector[RunType], graph: TofCoordTransformGraph
) -> TofDetector[RunType]:
    """Uses the transformation graph to compute ``tof``."""
    return da.transform_coords(('tof',), graph=graph)


convert_from_known_peaks_providers = (
    geometry_graph,
    tof_from_known_dhkl_graph,
    compute_tof,
)
convert_pulse_shaping = (
    geometry_graph,
    tof_from_t0_estimate_graph,
    compute_tof,
)
providers = (compute_tof_in_each_cluster, geometry_graph)
