# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

import numpy as np
import scipp as sc

from .types import (
    CorrectedDetector,
    CorrectedDspacing,
    DspacingBins,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    KeepEvents,
    NormalizedDspacing,
    RunType,
    TwoThetaBins,
)


def _reconstruct_wavelength(
    dspacing_bins: sc.Variable, two_theta_bins: sc.Variable
) -> sc.Variable:
    dspacing = dspacing_bins
    two_theta = sc.midpoints(two_theta_bins)
    return (2 * dspacing * sc.sin(two_theta / 2)).to(unit='angstrom')


_BASE_TWO_THETA_RESOLUTION = 1024
"""Number of bins used to cover the full two-theta range of a detector.

This sets the resolution of the wavelength reconstructed from bin centers, see
:func:`_reconstruct_wavelength`. It is independent of the two-theta binning
requested for the final result, see :func:`_focussing_two_theta_bins`.

It only matters when focussing produces a histogram *and* the run is normalized
by a monitor histogram. The monitor is then looked up once per (d-spacing,
two-theta) bin and the lookup is piecewise constant on the bins of the monitor,
so the wavelength spread within a bin should stay below the width of a monitor
bin. Since

.. math::

    \\frac{\\Delta\\lambda}{\\lambda}
    = \\frac{\\Delta 2\\theta}{2} \\cot\\theta

this is most demanding at small scattering angles. If the focussed data keeps
its events, they carry their own wavelength and this number has no influence on
the result.
"""


def _extend_edges(start: float, step: float, limit: float) -> np.ndarray:
    """Ascending edges reaching from ``start`` (exclusive) past ``limit``."""
    n = max(int(np.ceil((limit - start) / step)), 0)
    edges = start + step * np.arange(1, n + 1)
    return edges if step > 0 else edges[::-1]


def _focussing_two_theta_bins(
    two_theta: sc.Variable, requested: sc.Variable | None
) -> sc.Variable:
    """Return the two-theta bin edges to use for focussing.

    The edges cover the full range of ``two_theta`` with bins no wider than that
    range divided by :py:data:`_BASE_TWO_THETA_RESOLUTION`.

    If ``requested`` is given, its edges are a subset of the returned edges: each
    requested bin is subdivided into equally wide sub-bins. Grouping the focussed
    data into ``requested`` is then an exact sum over whole sub-bins. Without
    this alignment, each focussing bin is assigned as a whole to the requested
    bin containing its center, so the number of bins per group varies
    periodically and produces large spikes in the result.
    """
    lo = two_theta.nanmin()
    hi = two_theta.nanmax()
    # Make the upper edge inclusive of the largest two-theta value.
    hi.value = np.nextafter(hi.value, np.inf)
    base_width = ((hi - lo) / _BASE_TWO_THETA_RESOLUTION).value
    if requested is None:
        return sc.linspace(
            'two_theta',
            start=lo,
            stop=hi,
            num=_BASE_TWO_THETA_RESOLUTION + 1,
            unit=two_theta.unit,
        )
    edges = requested.to(unit=two_theta.unit, dtype='float64').values
    widths = np.diff(edges)
    n_sub = np.ceil(widths / base_width).astype(int)
    # Index of the requested bin each sub-bin belongs to, and its position within it.
    # The first sub-bin of a requested bin reproduces its lower edge exactly.
    offset = np.concatenate([[0], np.cumsum(n_sub)])
    index = np.repeat(np.arange(len(n_sub)), n_sub)
    position = (np.arange(offset[-1]) - offset[index]) / n_sub[index]
    return sc.array(
        dims=['two_theta'],
        values=np.concatenate(
            [
                _extend_edges(edges[0], -base_width, lo.value),
                edges[index] + position * widths[index],
                edges[-1:],
                _extend_edges(edges[-1], base_width, hi.value),
            ]
        ),
        unit=two_theta.unit,
    )


def focus_data_dspacing_and_two_theta(
    data: CorrectedDetector[RunType],
    dspacing_bins: DspacingBins,
    two_theta_bins: TwoThetaBins,
    keep_events: KeepEvents[RunType],
) -> CorrectedDspacing[RunType]:
    """
    Reduce the pixel-based data to d-spacing and two-theta dimensions.

    The two-theta binning is finer than :py:class:`TwoThetaBins` and covers the full
    two-theta range of the detector, not only the requested range. Both are necessary
    to have sufficient wavelength resolution when performing a monitor normalization
    in a follow-up workflow step. The bins are nevertheless aligned with
    :py:class:`TwoThetaBins` such that :func:`group_two_theta` can produce the
    requested binning exactly, see :func:`_focussing_two_theta_bins`.

    Parameters
    ----------
    data:
        The input data to be reduced, which must have 'wavelength', 'dspacing',
        'two_theta' coordinates.
    dspacing_bins:
        The bins to use for the d-spacing dimension.
    two_theta_bins:
        The two-theta bins requested for the final result, or ``None`` if the data
        will not be grouped by two-theta.
    keep_events:
        Whether to keep the events in the output. If `False`, the output will be
        histogrammed instead of binned.

    Returns
    -------
    :
        The reduced data with 'dspacing' and 'two_theta' dimensions.
    """
    twotheta_bins = _focussing_two_theta_bins(data.coords['two_theta'], two_theta_bins)
    args = {twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins}
    if keep_events.value:
        result = data.bin(args)
    else:
        # Reconstructing the wavelength results in an inconsistency if dspacing was
        # computed with a calibration table. Another option would be to use, e.g., the
        # mean wavelength in each bin, but this leads to random wavelength values that
        # break stream processing.
        result = data.hist(args).assign_coords(
            wavelength=_reconstruct_wavelength(
                dspacing_bins=dspacing_bins, two_theta_bins=twotheta_bins
            )
        )

    return CorrectedDspacing[RunType](result)


def integrate_two_theta(
    data: NormalizedDspacing[RunType],
) -> FocussedDataDspacing[RunType]:
    """Integrate the two-theta dimension of the data."""
    if 'two_theta' not in data.dims:
        raise sc.DimensionError("Data does not have a 'two_theta' dimension.")
    return FocussedDataDspacing[RunType](
        data.nansum(dim='two_theta')
        if data.bins is None
        else data.bins.concat('two_theta')
    )


def group_two_theta(
    data: NormalizedDspacing[RunType],
    two_theta_bins: TwoThetaBins,
) -> FocussedDataDspacingTwoTheta[RunType]:
    """Group the data by two-theta bins.

    ``data`` was focussed onto a finer two-theta grid that is aligned with
    ``two_theta_bins``, see :func:`focus_data_dspacing_and_two_theta`. Grouping is
    therefore an exact sum over whole sub-bins.
    """
    if two_theta_bins is None:
        raise ValueError("Cannot group by two-theta, no 'TwoThetaBins' were set.")
    if 'two_theta' not in data.dims:
        raise ValueError("Data does not have a 'two_theta' dimension.")
    data = data.assign_coords(
        two_theta=sc.midpoints(data.coords['two_theta']).to(unit=two_theta_bins.unit)
    )
    return FocussedDataDspacingTwoTheta[RunType](
        data.groupby('two_theta', bins=two_theta_bins).nansum('two_theta')
        if data.bins is None
        else data.bin(two_theta=two_theta_bins)
    )


def collect_detectors(*detectors: sc.DataArray) -> sc.DataGroup:
    """Store all inputs in a single data group.

    This function is intended to be used to reduce a workflow which
    was mapped over detectors.

    Parameters
    ----------
    detectors:
        Data arrays for each detector bank.
        All arrays must have a scalar "detector" coord containing a ``str``.

    Returns
    -------
    :
        The inputs as a data group with the "detector" coord as the key.
    """
    return sc.DataGroup({da.coords.pop('detector').value: da for da in detectors})


providers = (
    focus_data_dspacing_and_two_theta,
    integrate_two_theta,
    group_two_theta,
)
"""Sciline providers for grouping pixels."""
