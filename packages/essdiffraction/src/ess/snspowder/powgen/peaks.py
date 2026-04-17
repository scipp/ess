# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Peak fitting and removal.

This module is specialized to POWGEN.
"""

import math
from collections.abc import Iterable
from itertools import combinations_with_replacement

import scipp as sc
from scippneutron.peaks import FitParameters, FitRequirements, FitResult, fit_peaks
from scippneutron.peaks.model import Model


def theoretical_vanadium_dspacing(
    *, hkl_range: int = 10, min_d: sc.Variable | None = None
) -> sc.Variable:
    r"""Return the d-spacing values for vanadium in an ideal case.

    Based on the bcc structure of vanadium, the values are

    .. math::

        d = \frac{a}{\sqrt{h^2 + k^2 + l^2}}

    where :math:`a = 3.0272` Å is the lattice constant
    of vanadium :cite:`Arblaster:2018` and :math:`h+k+l` is even.

    Parameters
    ----------
    hkl_range:
        h, k, l are each limited to the integer interval ``[0, hkl_range]``.
    min_d:
        If given, only return values greater than this.

    Returns
    -------
    :
        Array of vanadium d-spacing values.
        Has dimension ``'dspacing'``.
    """
    a = 3.0272
    d_values = {
        a / math.sqrt(h**2 + k**2 + l**2)
        for h, k, l in combinations_with_replacement(range(hkl_range), 3)  # noqa: E741
        if (h + k + l) % 2 == 0 and (h + k + l) > 0
    }
    d = sc.array(dims=['dspacing'], values=sorted(d_values), unit='angstrom')
    if min_d is not None:
        return d[d > min_d]
    return d


def fit_vanadium_peaks(
    data: sc.DataArray,
    *,
    peak_estimates: sc.Variable | None = None,
    windows: sc.Variable | None = None,
    background: Model | str | Iterable[Model] | Iterable[str] | None = None,
    peak: Model | str | Iterable[Model] | Iterable[str] | None = None,
    fit_parameters: FitParameters | None = None,
    fit_requirements: FitRequirements | None = None,
) -> list[FitResult]:
    """Fit coherent scattering peaks of vanadium.

    This function wraps :func:`scippneutron.peaks.fit_peaks` and provides
    default parameters for vanadium at POWGEN.

    Parameters
    ----------
    data:
        A 1d data array where ``data.data`` is the dependent variable
        and ``data.coords[data.dim]`` is the independent variable for the fit.
        Must be 1-dimensional and not binned.
    peak_estimates:
        Initial estimates of peak locations.
        A peak will be fitted for each estimate.
        Must be a 1d variable with dimension ``data.dim``.
        If ``None``, estimates are derived using :func:`theoretical_vanadium_dspacing`.
    windows:
        If a scalar, the size of fit windows.
        A window is constructed for each peak estimate centered on the estimate
        with a width equal to ``windows`` (adjusted to the data range and to maintain
        a separation between peaks, see
        :attr:`scippneutron.peaks.FitParameters.neighbor_separation_factor`).

        If a 2d array, the windows for each peak.
        Must have sizes ``{data.dim: len(data), 'range': 2}`` where
        ``windows['range', 0]`` and ``windows['range', 1]`` are the lower and upper
        bounds of the fit windows, respectively.
        The windows are not adjusted automatically in this case.

        Defaults to ``sc.scalar(0.02, unit='angstrom')``.
    background:
        The background model or models.
        Defaults to ``('linear', 'quadratic')``.
        That is, a fit with a linear background is attempted, and if the fit fails,
        a quadratic background is tried.
    peak:
        The peak model or models.
        Defaults to ``'gaussian'``.
    fit_parameters:
        Parameters for the fit not otherwise listed as function arguments.
    fit_requirements:
        Constraints on the fit result.

    Returns
    -------
    :
        A :class:`FitResult` for each peak.
    """
    if peak_estimates is None:
        peak_estimates = theoretical_vanadium_dspacing(
            hkl_range=10, min_d=sc.scalar(0.41, unit='angstrom')
        )
    if windows is None:
        windows = sc.scalar(0.02, unit='angstrom')
    if background is None:
        background = ('linear', 'quadratic')
    if peak is None:
        peak = 'gaussian'

    fits = fit_peaks(
        data,
        peak_estimates=peak_estimates,
        windows=windows,
        background=background,
        peak=peak,
        fit_parameters=fit_parameters,
        fit_requirements=fit_requirements,
    )
    return fits
