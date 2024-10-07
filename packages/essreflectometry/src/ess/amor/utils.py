from collections.abc import Sequence

import numpy as np
import scipp as sc

from ess.reflectometry.types import (
    DetectorRotation,
    QBins,
    ReflectivityData,
    ReflectivityOverQ,
    RunType,
    SampleRotation,
    SampleRun,
)

from .geometry import Detector
from .types import (
    QThetaFigure,
    ReflectivityDiagnosticsView,
    ThetaBins,
    WavelengthThetaFigure,
    WavelengthZIndexFigure,
)


def theta_grid(
    nu: DetectorRotation[RunType], mu: SampleRotation[RunType]
) -> ThetaBins[RunType]:
    """Special grid used to create intensity maps over (theta, wavelength).
    The grid avoids aliasing artifacts that occur if the
    theta bins overlap the blade edges."""
    # angular offset of two blades:
    bladeAngle = 2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)
    # associate an angle with each z-coordinate on one blade
    blade_grid = sc.atan(
        sc.arange("theta", 0, 33)
        * Detector.dZ
        / (Detector.distance + sc.arange("theta", 0, 33) * Detector.dX)
    )
    # approximate angular step width on one blade
    stepWidth = blade_grid[1] - blade_grid[0]
    # shift "downwards" of the grid in order to define boundaries rather than centers
    blade_grid = blade_grid - 0.2 * stepWidth

    delta_grid = sc.array(
        dims=["theta"], values=[], unit=blade_grid.unit, dtype=blade_grid.dtype
    )
    # loop over all blades but one:
    for _ in range(Detector.nBlades.value - 1):
        # append the actual blade's grid to the array of detector-local angles
        delta_grid = sc.concat((delta_grid, blade_grid), "theta")
        # shift the blade grid by the angular offset
        blade_grid = blade_grid + bladeAngle
        # remove all entries in the detector local grid which are above the
        #  expected next value (plus some space to avoid very thin bins)
        delta_grid = delta_grid[delta_grid < blade_grid[0] - 0.5 * stepWidth]
    # append the grid of the last blade.
    delta_grid = sc.concat((delta_grid, blade_grid), "theta")

    # add angular position of the detector
    grid = (
        nu.to(unit="rad")
        - mu.to(unit="rad")
        - sc.array(
            dims=delta_grid.dims, values=delta_grid.values[::-1], unit=delta_grid.unit
        ).to(unit="rad")
        + 0.5 * Detector.nBlades * bladeAngle.to(unit="rad")
    )
    # TODO: If theta filtering is added, use it here
    # some filtering
    # theta_grid = theta_grid[theta_grid>=thetaMin]
    # theta_grid = theta_grid[theta_grid<=thetaMax]
    return grid


def wavelength_theta_figure(
    da: sc.DataArray,
    wavelength_bins: sc.Variable | None = None,
    theta_bins: sc.Variable | None = None,
    mu: sc.Variable | None = None,
    nu: sc.Variable | None = None,
    q_edges_to_display: Sequence[sc.Variable] = (),
):
    da = da.bins.concat(set(da.dims) - {"wavelength", "theta"})

    bins = {}
    if mu is not None and nu is not None:
        bins['theta'] = theta_grid(nu=nu, mu=mu)
    if theta_bins is not None:
        bins['theta'] = theta_bins
    if wavelength_bins is not None:
        bins['wavelength'] = wavelength_bins

    if 'theta' not in da.dims and 'theta' not in bins:
        raise ValueError('No theta binning provided')

    if 'wavelength' not in da.dims and 'wavelength' not in bins:
        raise ValueError('No wavelength binning provided')

    h = da.hist(**bins).transpose(('theta', 'wavelength'))
    p = h.plot(norm="log")
    for q in q_edges_to_display:
        p.ax.plot(
            [0.0, 4 * np.pi * (sc.sin(h.coords["theta"].max()) / q).value],
            [0.0, h.coords["theta"].max().value],
            linestyle="solid",
            linewidth=0.5,
            color="black",
            marker=None,
        )
    return p


def wavelength_theta_diagnostic_figure(
    da: ReflectivityData,
    thbins: ThetaBins[SampleRun],
) -> WavelengthThetaFigure:
    return wavelength_theta_figure(da, theta_bins=thbins)


def q_theta_diagnostic_figure(
    da: ReflectivityData,
    thbins: ThetaBins[SampleRun],
    qbins: QBins,
) -> QThetaFigure:
    da = da.bins.concat().hist(theta=thbins, Q=qbins)
    return da.plot(grid=True, norm="log")


def wavelength_z_diagnostic_figure(
    da: ReflectivityData,
) -> WavelengthZIndexFigure:
    return (
        da.bins.concat("stripe")
        .flatten(("blade", "wire"), to="z_index")
        .hist()
        .plot(norm="log", grid=True)
    )


def diagnostic_view(
    lath: WavelengthThetaFigure,
    laz: WavelengthZIndexFigure,
    qth: QThetaFigure,
    ioq: ReflectivityOverQ,
) -> ReflectivityDiagnosticsView:
    ioq = ioq.hist().plot(norm="log")
    return (ioq + laz) / (lath + qth)


providers = (
    theta_grid,
    wavelength_z_diagnostic_figure,
    wavelength_theta_diagnostic_figure,
    q_theta_diagnostic_figure,
    diagnostic_view,
)
