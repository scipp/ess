# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Sequence

import numpy as np
import plopp as pp
import scipp as sc

from .types import (
    QBins,
    QThetaFigure,
    ReducibleData,
    Reference,
    ReflectivityDiagnosticsView,
    ReflectivityOverQ,
    ReflectivityOverZW,
    SampleRun,
    ThetaBins,
    WavelengthBins,
    WavelengthThetaFigure,
    WavelengthZIndexFigure,
)


def _reshape_array_to_expected_shape(da, dims, **bins):
    for d in dims:
        if d in da.coords:
            shares_dimension_with_other_coord_in_dims = any(
                set(da.coords[d].dims).intersection(da.coords[b].dims)
                for b in dims
                if d != b and b in da.coords
            )
            if not shares_dimension_with_other_coord_in_dims:
                da = da.flatten(da.coords[d].dims, to=d)
    if da.bins:
        da = da.bins.concat(set(da.dims) - set(dims))
    elif set(da.dims) > set(dims):
        raise ValueError(
            f'Histogram must have exactly the dimensions'
            f' {set(dims)} but got {set(da.dims)}'
        )

    if not set(da.dims).union(set(bins)) >= set(dims):
        raise ValueError(
            f'Could not find bins for dimensions:'
            f' {set(dims) - set(da.dims).union(set(bins))}'
        )

    if da.bins or not set(da.dims) == set(dims):
        da = da.hist(**bins)

    return da.transpose(dims)


def _repeat_variable_argument(n, arg):
    return (arg,) * n if arg is None or isinstance(arg, sc.Variable | int) else arg


def wavelength_theta_figure(
    da: sc.DataArray | Sequence[sc.DataArray],
    *,
    wavelength_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    theta_bins: (sc.Variable | None) | Sequence[sc.Variable | None] = None,
    q_edges_to_display: Sequence[sc.Variable] = (),
    linewidth: float = 1.0,
    **kwargs,
):
    '''
    Creates a figure displaying a histogram over :math:`\\theta`
    and :math:`\\lambda`.

    The input can either be a single data array containing the data to display, or
    a sequence of data arrays.

    The inputs must either have coordinates called "theta"
    and "wavelength", or they must be histograms with dimensions
    "theta" and "wavelength".

    If :code:`wavelength_bins` or :code:`theta_bins` are provided,
    they are used to construct the histogram. If not provided, the function uses the
    bin edges that already exist on the data arrays.

    If :code:`q_edges_to_display` is provided, lines will be drawn in the figure
    corresponding to :math:`Q` equal to the values in :code:`q_edges_to_display`.

    Parameters
    ----------
    da : array or sequence of arrays
        Data arrays to display.
    wavelength_bins : array-like, optional
        Bins used to histogram the data in wavelength.
    theta_bins : array-like, optional
        Bins used to histogram the data in theta.
    q_edges_to_display : sequence of float, optional
        Values of :math:`Q` to be displayed as straight lines in the figure.
    linewidth : float, optional
        Thickness of the displayed :math:`Q` lines.
    **kwargs : keyword arguments, optional
        Additional parameters passed to the histogram plot function,
        used to customize colors, etc.

    Returns
    -------
        A Plopp figure displaying the histogram.
    '''

    if isinstance(da, sc.DataArray):
        return wavelength_theta_figure(
            (da,),
            wavelength_bins=(wavelength_bins,),
            theta_bins=(theta_bins,),
            q_edges_to_display=q_edges_to_display,
            **kwargs,
        )

    wavelength_bins, theta_bins = (
        _repeat_variable_argument(len(da), arg) for arg in (wavelength_bins, theta_bins)
    )

    hs = []
    for d, wavelength_bin, theta_bin in zip(
        da, wavelength_bins, theta_bins, strict=True
    ):
        bins = {}

        if wavelength_bin is not None:
            bins['wavelength'] = wavelength_bin

        if theta_bin is not None:
            bins['theta'] = theta_bin

        hs.append(_reshape_array_to_expected_shape(d, ('theta', 'wavelength'), **bins))

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
    '''
    Creates a figure displaying a histogram over :math:`\\theta`
    and :math:`Q`.

    The input can either be a single data array containing the data to display, or
    a sequence of data arrays.

    The inputs must either have coordinates called "theta" and "Q",
    or they must be histograms with dimensions "theta" and "Q".

    If :code:`theta_bins` or :code:`q_bins` are provided, they are used
    to construct the histogram. If not provided, the function uses the
    bin edges that already exist on the data arrays.

    Parameters
    ----------
    da : array or sequence of arrays
        Data arrays to display.
    q_bins : array-like, optional
        Bins used to histogram the data in Q.
    theta_bins : array-like, optional
        Bins used to histogram the data in theta.

    Returns
    -------
        A Plopp figure displaying the histogram.
    '''

    if isinstance(da, sc.DataArray):
        return q_theta_figure(
            (da,),
            q_bins=(q_bins,),
            theta_bins=(theta_bins,),
            **kwargs,
        )

    q_bins, theta_bins = (
        _repeat_variable_argument(len(da), arg) for arg in (q_bins, theta_bins)
    )

    hs = []
    for d, q_bin, theta_bin in zip(da, q_bins, theta_bins, strict=True):
        bins = {}

        if q_bin is not None:
            bins['Q'] = q_bin

        if theta_bin is not None:
            bins['theta'] = theta_bin

        hs.append(_reshape_array_to_expected_shape(d, ('theta', 'Q'), **bins))

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
    '''
    Creates a figure displaying a histogram over the detector "Z"-direction,
    corresponding to the combination of the logical detector coordinates
    :code:`blade` and :code:`wire`.

    The input can either be a single data array containing the data to display, or
    a sequence of data arrays.

    The inputs must either have coordinates called "blade" and "wire" and "wavelength",
    or they must be histograms with dimensions "blade", "wire" and "wavelength".

    If :code:`wavelength_bins` is provided, it is used
    to construct the histogram. If not provided, the function uses the
    bin edges that already exist on the data arrays.

    Parameters
    ----------
    da : array or sequence of arrays
        Data arrays to display.
    wavelength_bins : array-like, optional
        Bins used to histogram the data in wavelength.

    Returns
    -------
        A Plopp figure displaying the histogram.
    '''

    if isinstance(da, sc.DataArray):
        return wavelength_z_figure((da,), wavelength_bins=(wavelength_bins,), **kwargs)

    wavelength_bins = _repeat_variable_argument(len(da), wavelength_bins)

    hs = []
    for d, wavelength_bin in zip(da, wavelength_bins, strict=True):
        bins = {}
        if wavelength_bin is not None:
            bins['wavelength'] = wavelength_bin

        d = _reshape_array_to_expected_shape(d, ("blade", "wire", "wavelength"), **bins)
        d = d.flatten(("blade", "wire"), to="z_index")
        hs.append(d)

    kwargs.setdefault('cbar', True)
    kwargs.setdefault('norm', 'log')
    kwargs.setdefault('grid', True)
    p = pp.imagefigure(*(pp.Node(h) for h in hs), **kwargs)
    p.ax.invert_yaxis()
    return p


def wavelength_theta_diagnostic_figure(
    da: ReducibleData[SampleRun],
    ref: Reference,
    wbins: WavelengthBins,
    thbins: ThetaBins[SampleRun],
) -> WavelengthThetaFigure:
    s = da.hist(wavelength=wbins, theta=thbins)
    r = ref.hist(theta=s.coords['theta'], wavelength=s.coords['wavelength']).data
    return wavelength_theta_figure(s / r)


def q_theta_diagnostic_figure(
    da: ReducibleData[SampleRun],
    ref: Reference,
    thbins: ThetaBins[SampleRun],
    qbins: QBins,
) -> QThetaFigure:
    s = da.hist(theta=thbins, Q=qbins)
    r = ref.hist(theta=s.coords['theta'], Q=s.coords['Q']).data
    return q_theta_figure(s / r)


def wavelength_z_diagnostic_figure(
    da: ReflectivityOverZW,
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
    wavelength_z_diagnostic_figure,
    wavelength_theta_diagnostic_figure,
    q_theta_diagnostic_figure,
    diagnostic_view,
)
