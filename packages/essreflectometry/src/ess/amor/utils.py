from collections.abc import Sequence

import numpy as np
import plopp as pp
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
    da: sc.DataArray | Sequence[sc.DataArray],
    *,
    wavelength_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    theta_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    q_edges_to_display: Sequence[sc.Variable] = (),
    linewidth: float = 1.0,
    **kwargs,
):
    if isinstance(da, sc.DataArray):
        return wavelength_theta_figure(
            (da,),
            wavelength_bins=(wavelength_bins,),
            theta_bins=(theta_bins,),
            q_edges_to_display=q_edges_to_display,
            **kwargs,
        )

    wavelength_bins, theta_bins = (
        (None,) * len(da)
        if v is None
        else (v,) * len(da)
        if isinstance(v, sc.Variable)
        else v
        for v in (wavelength_bins, theta_bins)
    )

    hs = []
    for d, wavelength_bin, theta_bin in zip(
        da, wavelength_bins, theta_bins, strict=True
    ):
        if d.bins:
            d = d.bins.concat(set(d.dims) - {"wavelength", "theta"})
        all_coords = {*d.coords, *(d.bins or d).coords}
        if 'wavelength' not in all_coords or 'theta' not in all_coords:
            raise ValueError('Data must have wavelength and theta coord')
        if d.bins or set(d.dims) != {"wavelength", "theta"}:
            bins = {}
            if 'sample_rotation' in d.coords and 'detector_rotation' in d.coords:
                bins['theta'] = theta_grid(
                    nu=d.coords['detector_rotation'], mu=d.coords['sample_rotation']
                )
            if theta_bin is not None:
                bins['theta'] = theta_bin
            if wavelength_bin is not None:
                bins['wavelength'] = wavelength_bin
            if 'theta' not in d.dims and 'theta' not in bins:
                raise ValueError('No theta binning provided')
            if 'wavelength' not in d.dims and 'wavelength' not in bins:
                raise ValueError('No wavelength binning provided')
            d = d.hist(**bins)

        hs.append(d.transpose(('theta', 'wavelength')))

    kwargs.setdefault('cbar', True)
    kwargs.setdefault('norm', 'log')
    p = pp.imagefigure(*(pp.Node(h) for h in hs), **kwargs)
    for q in q_edges_to_display:
        thmax = max(h.coords["theta"].max() for h in hs)
        p.ax.plot(
            [0.0, 4 * np.pi * (sc.sin(thmax) / q).value],
            [0.0, thmax.value],
            linestyle="solid",
            linewidth=linewidth,
            color="black",
            marker=None,
        )
    return p


def q_theta_figure(
    da: sc.DataArray | Sequence[sc.DataArray],
    *,
    q_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    theta_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    **kwargs,
):
    if isinstance(da, sc.DataArray):
        return q_theta_figure(
            (da,), q_bins=(q_bins,), theta_bins=(theta_bins,), **kwargs
        )

    q_bins, theta_bins = (
        (None,) * len(da)
        if v is None
        else (v,) * len(da)
        if isinstance(v, sc.Variable)
        else v
        for v in (q_bins, theta_bins)
    )

    hs = []
    for d, q_bin, theta_bin in zip(da, q_bins, theta_bins, strict=True):
        if d.bins:
            d = d.bins.concat(set(d.dims) - {'theta', 'Q'})

        all_coords = {*d.coords, *(d.bins or d).coords}
        if 'theta' not in all_coords or 'Q' not in all_coords:
            raise ValueError('Data must have theta and Q coord')
        if d.bins or set(d.dims) != {"theta", "Q"}:
            bins = {}
            if theta_bin is not None:
                bins['theta'] = theta_bin
            if q_bin is not None:
                bins['Q'] = q_bin
            if 'theta' not in d.dims and 'theta' not in bins:
                raise ValueError('No theta binning provided')
            if 'Q' not in d.dims and 'Q' not in bins:
                raise ValueError('No Q binning provided')
            d = d.hist(**bins)

        hs.append(d.transpose(('theta', 'Q')))

    kwargs.setdefault('cbar', True)
    kwargs.setdefault('norm', 'log')
    kwargs.setdefault('grid', True)
    return pp.imagefigure(*(pp.Node(h) for h in hs), **kwargs)


def wavelength_z_figure(
    da: sc.DataArray | Sequence[sc.DataArray],
    *,
    wavelength_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    **kwargs,
):
    if isinstance(da, sc.DataArray):
        return wavelength_z_figure((da,), wavelength_bins=(wavelength_bins,), **kwargs)

    (wavelength_bins,) = (
        (None,) * len(da)
        if v is None
        else (v,) * len(da)
        if isinstance(v, sc.Variable)
        else v
        for v in (wavelength_bins,)
    )

    hs = []
    for d, wavelength_bin in zip(da, wavelength_bins, strict=True):
        if d.bins:
            d = d.bins.concat(set(d.dims) - {'blade', 'wire', 'wavelength'})
            bins = {}
            if wavelength_bin is not None:
                bins['wavelength'] = wavelength_bin
            if 'wavelength' not in d.dims and 'wavelength' not in bins:
                raise ValueError('No wavelength binning provided')
            d = d.hist(**bins)

        d = d.flatten(("blade", "wire"), to="z_index")
        hs.append(d.transpose(('z_index', 'wavelength')))

    kwargs.setdefault('cbar', True)
    kwargs.setdefault('norm', 'log')
    kwargs.setdefault('grid', True)
    return pp.imagefigure(*(pp.Node(h) for h in hs), **kwargs)


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
    return q_theta_figure(da, q_bins=qbins, theta_bins=thbins)


def wavelength_z_diagnostic_figure(
    da: ReflectivityData,
) -> WavelengthZIndexFigure:
    return wavelength_z_figure(da)


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
