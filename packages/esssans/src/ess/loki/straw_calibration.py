# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Calibration of the position along LoKI straws using a slit mask.

A mask with narrow transparent slits perpendicular to the straws is placed in front
of a detector bank and illuminated by a neutron source.
Every straw records a comb of peaks, one peak per slit.
Comparing the measured peak positions with the known positions of the slit images
yields, for each straw, a polynomial that maps the nominal position along the straw
(as computed from the charge division) to the true position.

The procedure is:

1. :func:`projected_slit_positions` computes where the slit images fall on each
   straw. For a source close to the mask, the images are magnified and the
   magnification depends on the distance of the straw behind the mask.
2. :func:`find_slit_peaks` locates the peaks in the counts of each straw.
3. :func:`fit_straw_calibration` fits a polynomial per straw.
4. :func:`calibrate_positions` evaluates the polynomials, e.g., at pixel positions.

:func:`estimate_source_distance` is a cross-check of the source position, based on
the magnification of the slit pattern increasing with depth.

The slits are always indexed by a dimension called ``slit``.
"""

import numpy as np
import scipp as sc
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, least_squares


def projected_slit_positions(
    slits: sc.Variable,
    *,
    source_to_mask: sc.Variable,
    mask_to_straw: sc.Variable,
    source_offset: sc.Variable | None = None,
) -> sc.Variable:
    """Return the positions of the slit images on the straws for a point source.

    The mask and the straw axes are assumed to lie in parallel planes, the slits
    to be perpendicular to the straws, and the straws parallel to the mask.
    A ray from the source through a slit then hits the straw at

    .. code-block:: text

        x = x_source + (x_slit - x_source) * (source_to_mask + mask_to_straw)
                                           / source_to_mask

    All positions along the straw are relative to the same origin, typically the
    centre of the mask, which is assumed to coincide with the centre of the straws.

    Parameters
    ----------
    slits:
        Positions of the slit centres along the mask, with dim ``slit``.
    source_to_mask:
        Distance from the source to the mask plane.
    mask_to_straw:
        Distance from the mask plane to the axis of each straw, e.g., with
        dims ``(layer, tube, straw)``.
    source_offset:
        Position of the source along the straw direction.
        Defaults to zero, i.e., the source faces the centre of the mask.

    Returns
    -------
    :
        Expected peak positions with dims ``mask_to_straw.dims + ('slit',)``.
    """
    if source_offset is None:
        source_offset = sc.scalar(0.0, unit=slits.unit)
    source_offset = source_offset.to(unit=slits.unit)
    magnification = (
        source_to_mask + mask_to_straw.to(unit=source_to_mask.unit)
    ) / source_to_mask
    return source_offset + magnification * (slits - source_offset)


def find_slit_peaks(
    counts: sc.DataArray,
    expected: sc.Variable,
    *,
    dim: str = 'pixel',
    max_offset: sc.Variable | None = None,
    min_significance: float = 5.0,
    fit_half_width: float = 0.3,
) -> sc.DataArray:
    """Find the peaks of the slit images in the counts of each straw.

    For each straw:

    1. Isolated hot pixels are replaced by the median of their neighbours.
    2. The active region of the straw is determined as the range of pixels with
       counts above 20% of the median of the non-zero counts.
       The ends of the active region are sharp and correspond to fixed positions
       on the straw, which makes their midpoint an estimate of the straw centre.
    3. A comb with teeth at ``scale * expected + offset`` is aligned with the
       background-subtracted counts. The scale is searched in [0.7, 1.2] and the
       offset within ``max_offset`` of the centre of the active region.
       Without this anchor, the comb could be shifted by one slit when the
       outermost slit images fall outside the active region.
    4. A Gaussian on a linear background is fitted around each tooth of the comb,
       within ``fit_half_width`` times the spacing of the teeth.
       The fit range is restricted to the active region. Peaks are rejected if the
       fit fails, the amplitude is not significant, the width is at the limits of
       the fit, or the centre is less than one width away from the end of the fit
       range, i.e., the peak is truncated.

    Parameters
    ----------
    counts:
        Counts per pixel with dims ``(..., dim)``, without masks.
        The coordinate for ``dim`` holds the nominal position of the pixel centres
        along the straw, either 1-D or with the same dims as ``counts``.
    expected:
        Expected (true) peak positions with dims ``(..., slit)``, typically
        from :func:`projected_slit_positions`. True and nominal positions must
        increase in the same direction, and the true position zero must be within
        ``max_offset`` of the centre of the active region of each straw.
    dim:
        Dimension along the straw.
    max_offset:
        Maximum offset between the centre of the active region and the position
        of the comb centre. Defaults to 30 mm, which is less than half of the
        spacing of LoKI's calibration mask.
    min_significance:
        Minimum ratio of the fitted peak amplitude to its standard deviation.
    fit_half_width:
        Half-width of the fit window around each peak, as a fraction of the spacing
        of the peaks, at most 0.5.
        The default suits peaks with a standard deviation of up to about a tenth of
        the spacing. Wider peaks need a wider window, since otherwise the window
        holds too little background on either side and the fit rejects the peak as
        not significant.

    Returns
    -------
    :
        Measured peak positions in the nominal coordinate with dims
        ``(..., slit)``, with variances from the fit.
        Coords: ``expected`` (true position), ``sigma`` (standard deviation of the
        Gaussian), ``active_begin`` and ``active_end`` (ends of the active region
        of each straw).
        Peaks that were not found are masked with mask ``not_found``.
    """
    if counts.masks:
        raise ValueError('Masks on the counts are not supported.')
    if counts.coords.is_edges(dim):
        raise ValueError(f"The coordinate for '{dim}' must hold pixel centres.")
    if 'slit' not in expected.dims:
        raise ValueError("Expected positions must have a 'slit' dimension.")
    if not 0 < fit_half_width <= 0.5:
        raise ValueError(f'fit_half_width must be in (0, 0.5], got {fit_half_width}.')
    if max_offset is None:
        max_offset = sc.scalar(30.0, unit='mm')
    x = counts.coords[dim]
    unit = x.unit
    straw_sizes = {d: s for d, s in counts.sizes.items() if d != dim}
    straw_shape = list(straw_sizes.values())
    peak_dims = [*straw_sizes, 'slit']
    expected = (
        expected.to(unit=unit, dtype='float64')
        .broadcast(sizes={**straw_sizes, 'slit': expected.sizes['slit']})
        .transpose(peak_dims)
        .copy()
    )
    npix = counts.sizes[dim]
    nslit = expected.sizes['slit']
    results = [
        _find_straw_peaks(
            x_i,
            y_i.astype(float),
            e_i,
            max_offset=max_offset.to(unit=unit).value,
            min_significance=min_significance,
            fit_half_width=fit_half_width,
        )
        for x_i, y_i, e_i in zip(
            x.broadcast(sizes=counts.sizes)
            .transpose([*straw_sizes, dim])
            .values.reshape(-1, npix),
            counts.data.transpose([*straw_sizes, dim]).values.reshape(-1, npix),
            expected.values.reshape(-1, nslit),
            strict=True,
        )
    ]
    # Shape (..., slit, 3) with position, stddev, and sigma of each peak.
    peaks = np.stack([r[0] for r in results]).reshape([*straw_shape, nslit, 3])
    active = np.array([r[1] for r in results]).reshape([*straw_shape, 2])

    def straw_var(values: np.ndarray) -> sc.Variable:
        return sc.array(dims=list(straw_sizes), values=values, unit=unit)

    return sc.DataArray(
        sc.array(
            dims=peak_dims,
            values=peaks[..., 0],
            variances=peaks[..., 1] ** 2,
            unit=unit,
        ),
        coords={
            'expected': expected,
            'sigma': sc.array(dims=peak_dims, values=peaks[..., 2], unit=unit),
            'active_begin': straw_var(active[..., 0]),
            'active_end': straw_var(active[..., 1]),
        },
        masks={'not_found': sc.array(dims=peak_dims, values=np.isnan(peaks[..., 0]))},
    )


def _despike(y: np.ndarray) -> np.ndarray:
    """Replace pixels with more than twice the counts of their neighbourhood."""
    neighbourhood = median_filter(y, size=5, mode='nearest')
    spike = y > 2 * neighbourhood + 5 * np.sqrt(neighbourhood + 1)
    return np.where(spike, neighbourhood, y)


def _active_region(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    nonzero = y[y > 0]
    if nonzero.size < 0.1 * y.size:
        return None
    above = np.flatnonzero(y > 0.2 * np.median(nonzero))
    return x[above[0]], x[above[-1]]


def _expected_spacing(expected: np.ndarray) -> float:
    return np.median(np.diff(np.sort(expected)))


def _align_comb(
    x: np.ndarray,
    y: np.ndarray,
    expected: np.ndarray,
    *,
    begin: float,
    end: float,
    max_offset: float,
) -> tuple[float, float]:
    """Return scale and offset such that peaks are near ``scale * expected + offset``.

    The score of a comb is the sum of the background-subtracted counts at its teeth.
    """
    pitch = x[1] - x[0]
    width = max(3, round(0.6 * _expected_spacing(expected) / pitch))
    signal = y - median_filter(y, size=width, mode='nearest')
    signal[(x < begin) | (x > end)] = 0.0
    scales = np.linspace(0.7, 1.2, 101)
    offsets = 0.5 * (begin + end) + np.arange(-max_offset, max_offset + pitch, pitch)
    teeth = scales[:, None, None] * expected + offsets[None, :, None]
    score = np.interp(teeth, x, signal, left=0.0, right=0.0).sum(axis=-1)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return scales[i], offsets[j]


def _gaussian_on_line(
    x: np.ndarray, amplitude: float, mu: float, sigma: float, b0: float, b1: float
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x


def _fit_peak(
    x: np.ndarray,
    y: np.ndarray,
    *,
    centre: float,
    half_width: float,
    begin: float,
    end: float,
    min_significance: float,
) -> tuple[float, float, float] | None:
    """Fit a Gaussian on a linear background, return ``(mu, stddev(mu), sigma)``."""
    pitch = x[1] - x[0]
    # The outermost pixels of the active region often show a bump; exclude them.
    lo = max(centre - half_width, begin + 2 * pitch)
    hi = min(centre + half_width, end - 2 * pitch)
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) < 8:
        return None
    # Fit relative to the window centre to decorrelate background and peak position.
    xw, yw = x[sel] - centre, y[sel]
    background = 0.5 * (yw[:2].mean() + yw[-2:].mean())
    imax = np.argmax(yw)
    p0 = [max(yw[imax] - background, 1.0), xw[imax], 3 * pitch, background, 0.0]
    bounds = (
        [0.0, xw[0], 0.5 * pitch, -np.inf, -np.inf],
        [np.inf, xw[-1], half_width, np.inf, np.inf],
    )
    try:
        popt, pcov = curve_fit(
            _gaussian_on_line,
            xw,
            yw,
            p0=p0,
            bounds=bounds,
            sigma=np.sqrt(np.maximum(yw, 1.0)),
        )
    except (RuntimeError, ValueError):
        return None
    amplitude, mu, sigma = popt[:3]
    stddev = np.sqrt(np.diag(pcov))
    if not np.all(np.isfinite(stddev)):
        return None
    if amplitude < min_significance * stddev[0]:
        return None
    if sigma < pitch or sigma > 0.9 * half_width:
        return None  # a single-pixel spike or no peak within the window
    if mu - sigma < xw[0] or mu + sigma > xw[-1]:
        return None  # truncated by the end of the straw or the fit window
    return mu + centre, stddev[1], sigma


def _find_straw_peaks(
    x: np.ndarray,
    y: np.ndarray,
    expected: np.ndarray,
    *,
    max_offset: float,
    min_significance: float,
    fit_half_width: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return ``(peaks, (begin, end))`` where peaks has shape ``(nslit, 3)``."""
    peaks = np.full((len(expected), 3), np.nan)
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    y = _despike(y)
    region = _active_region(x, y)
    if region is None:
        return peaks, (np.nan, np.nan)
    begin, end = region
    scale, offset = _align_comb(
        x, y, expected, begin=begin, end=end, max_offset=max_offset
    )
    half_width = fit_half_width * scale * _expected_spacing(expected)
    for k, e in enumerate(expected):
        fit = _fit_peak(
            x,
            y,
            centre=scale * e + offset,
            half_width=half_width,
            begin=begin,
            end=end,
            min_significance=min_significance,
        )
        if fit is not None:
            peaks[k] = fit
    return peaks, region


def _combined_mask(da: sc.DataArray) -> sc.Variable:
    mask = sc.zeros(sizes=da.sizes, dtype=bool)
    for m in da.masks.values():
        mask |= m
    return mask


def fit_straw_calibration(peaks: sc.DataArray, *, max_degree: int = 3) -> sc.DataArray:
    """Fit a polynomial per straw mapping measured to expected peak positions.

    The degree of the polynomial is ``min(max_degree, n - 3)`` where ``n`` is the
    number of peaks found in the straw, so the fit always has at least two degrees
    of freedom. Straws with fewer than four peaks are not calibrated.
    Peaks are excluded if any mask of ``peaks`` is set.
    The fit is unweighted since the deviations from the polynomial are dominated by
    systematic effects rather than by the statistical uncertainties of the peak
    positions.

    The polynomial variable is the nominal position divided by
    ``reference_length``, which is the largest expected peak position.
    The coefficients therefore all have the unit of a length, and so does the
    result.

    Parameters
    ----------
    peaks:
        Measured peak positions with dims ``(..., slit)`` and coord ``expected``,
        typically from :func:`find_slit_peaks`.
    max_degree:
        Maximum degree of the polynomial.

    Returns
    -------
    :
        Polynomial coefficients with dims ``(..., power)``.
        Coords:

        - ``reference_length``: Divisor of the nominal position.
        - ``n_peaks``: Number of peaks used per straw.
        - ``degree``: Degree of the polynomial per straw.
        - ``residual_stddev``: Standard deviation of the difference between expected
          and calibrated peak positions, ``sqrt(sum(residual**2) / (n - degree - 1))``.
        - ``fit_begin``, ``fit_end``: Range of measured peak positions used in the
          fit. Outside this range, the polynomial is extrapolated.

        Straws without calibration are masked with mask ``uncalibrated`` and have
        NaN coefficients.
    """
    if 'slit' not in peaks.dims:
        raise ValueError("Peaks must have a 'slit' dimension.")
    expected = peaks.coords['expected']
    unit = expected.unit
    reference_length = sc.scalar(
        np.nanmax(np.abs(expected.values)), unit=unit, dtype='float64'
    )
    straw_sizes = {d: s for d, s in peaks.sizes.items() if d != 'slit'}
    straw_shape = list(straw_sizes.values())
    order = [*straw_sizes, 'slit']
    nslit = peaks.sizes['slit']
    measured = (peaks.data / reference_length).to(unit='dimensionless')
    measured = measured.transpose(order).values.reshape(-1, nslit)
    expected = expected.transpose(order).values.reshape(-1, nslit)
    valid = ~_combined_mask(peaks).transpose(order).values.reshape(-1, nslit)

    nstraw = len(measured)
    coefficients = np.full((nstraw, max_degree + 1), np.nan)
    degree = np.full(nstraw, -1)
    residual_stddev = np.full(nstraw, np.nan)
    fit_range = np.full((nstraw, 2), np.nan)
    for i, (m, e, v) in enumerate(zip(measured, expected, valid, strict=True)):
        n = np.count_nonzero(v)
        deg = min(max_degree, n - 3)
        if deg < 1:
            continue
        coefficients[i] = 0.0
        coefficients[i, : deg + 1] = np.polynomial.polynomial.polyfit(m[v], e[v], deg)
        degree[i] = deg
        residual = e[v] - np.polynomial.polynomial.polyval(m[v], coefficients[i])
        residual_stddev[i] = np.sqrt(np.sum(residual**2) / (n - deg - 1))
        fit_range[i] = m[v].min(), m[v].max()

    def straw_var(values: np.ndarray, unit: str | None) -> sc.Variable:
        return sc.array(
            dims=list(straw_sizes), values=values.reshape(straw_shape), unit=unit
        )

    degree = straw_var(degree, None)
    fit_range = fit_range * reference_length.value
    return sc.DataArray(
        sc.array(
            dims=[*straw_sizes, 'power'],
            values=coefficients.reshape([*straw_shape, max_degree + 1]),
            unit=unit,
        ),
        coords={
            'power': sc.arange('power', max_degree + 1, unit=None),
            'reference_length': reference_length,
            'n_peaks': straw_var(valid.sum(axis=-1), None),
            'degree': degree,
            'residual_stddev': straw_var(residual_stddev, unit),
            'fit_begin': straw_var(fit_range[:, 0], unit),
            'fit_end': straw_var(fit_range[:, 1], unit),
        },
        masks={'uncalibrated': degree < sc.index(1)},
    )


def calibrate_positions(coefficients: sc.DataArray, x: sc.Variable) -> sc.DataArray:
    """Evaluate the calibration polynomials at nominal positions ``x``.

    Parameters
    ----------
    coefficients:
        Output of :func:`fit_straw_calibration`.
    x:
        Nominal positions along the straws, e.g., pixel positions with dims
        ``(pixel,)`` or measured peak positions with dims ``(..., slit)``.

    Returns
    -------
    :
        True positions, with the mask of uncalibrated straws.
    """
    t = (sc.values(x) / coefficients.coords['reference_length']).to(
        unit='dimensionless', dtype='float64'
    )
    c = coefficients.data
    degree = c.sizes['power'] - 1
    result = c['power', degree]
    for power in range(degree - 1, -1, -1):
        result = result * t + c['power', power]
    return sc.DataArray(result, masks=dict(coefficients.masks))


def estimate_source_distance(
    peaks: sc.DataArray,
    *,
    slits: sc.Variable,
    depth: sc.Variable,
    max_abs_slit: sc.Variable | None = None,
) -> sc.Variable:
    """Estimate the distance of the source from the magnification of the slit images.

    For a point source at distance ``L`` in front of the plane ``depth = 0``, the
    slit pattern on a straw at ``depth`` is magnified by a factor proportional to
    ``L + depth``. The measured spacing of the peaks additionally includes the
    scale of the charge division readout.

    For each straw, a straight line is fitted to the measured peak positions as a
    function of the slit positions. Only slits close to the centre are used, since
    the response of the readout is less linear near the ends of the straws.
    The median of the logarithm of the slope over tubes is then fitted with

    .. code-block:: text

        log(slope(..., straw)) = log(g(straw)) + log(L + depth(..., straw))

    where ``...`` are the dims other than ``tube`` and ``straw``, typically
    ``layer``, and ``depth`` is the median over tubes.
    This assumes that the readout scale ``g`` depends only on the position of the
    straw within its tube, on average over tubes. A readout scale that differs
    between layers biases the result. Such systematic errors are not included in
    the returned variance.

    For a source that is large and close to the mask, the slit images overlap and
    the fitted peaks are magnified less than the images of a point source at the
    centre of the source. The result is then an effective distance, which is larger
    than the distance of the centre of the source.

    Parameters
    ----------
    peaks:
        Output of :func:`find_slit_peaks`, with dims ``tube``, ``straw``,
        and ``slit``.
    slits:
        Positions of the slits on the mask.
    depth:
        Depth of each straw, measured along the beam, relative to a reference plane.
    max_abs_slit:
        Only slits with ``abs(slits) <= max_abs_slit`` are used.
        Defaults to 250 mm.

    Returns
    -------
    :
        Distance from the source to the reference plane ``depth = 0``,
        with statistical variance.
    """
    if max_abs_slit is None:
        max_abs_slit = sc.scalar(250.0, unit='mm')
    unit = depth.unit
    inner = (abs(slits) <= max_abs_slit.to(unit=slits.unit)).values
    s = slits.to(unit=unit).values[inner]
    group_dims = [d for d in peaks.dims if d not in ('tube', 'straw', 'slit')]
    order = [*group_dims, 'tube', 'straw', 'slit']
    ngroup = int(np.prod([peaks.sizes[d] for d in group_dims]))
    shape = [ngroup, peaks.sizes['tube'], peaks.sizes['straw'], len(s)]
    measured = peaks.data.to(unit=unit).transpose(order).values[..., inner]
    valid = ~_combined_mask(peaks).transpose(order).values[..., inner]
    measured = measured.reshape(shape)
    valid = valid.reshape(shape)
    depth = (
        depth.broadcast(sizes={d: peaks.sizes[d] for d in order[:-1]})
        .transpose(order[:-1])
        .values.reshape(shape[:-1])
    )

    log_slope = np.full(shape[:-1], np.nan)
    for idx in np.ndindex(*log_slope.shape):
        v = valid[idx]
        if np.count_nonzero(v) >= max(3, len(s) - 1):
            log_slope[idx] = np.log(np.polyfit(s[v], measured[idx][v], 1)[0])
    # Median over tubes, require a few tubes per (group, straw).
    n_tubes = np.sum(np.isfinite(log_slope), axis=1)
    median_log_slope = np.full(n_tubes.shape, np.nan)
    has_data = n_tubes >= 3
    median_log_slope[has_data] = np.nanmedian(
        log_slope.transpose(0, 2, 1)[has_data], axis=-1
    )
    median_depth = np.median(depth, axis=1)
    group, straw = np.nonzero(has_data)
    straws, k = np.unique(straw, return_inverse=True)
    y = median_log_slope[group, straw]
    z = median_depth[group, straw]
    nparam = len(straws) + 1
    if len(y) <= nparam:
        raise ValueError(
            'Cannot estimate the source distance: need peaks in straws at more '
            'than one depth for the same straw index.'
        )

    def residual(p: np.ndarray) -> np.ndarray:
        return y - p[k] - np.log(p[-1] + z)

    x0 = np.r_[np.zeros(len(straws)), 10 * np.ptp(z) + 1.0]
    fit = least_squares(
        residual, x0=x0, bounds=(np.r_[np.full(len(straws), -np.inf), -z.min()], np.inf)
    )
    if np.linalg.matrix_rank(fit.jac) < nparam:
        raise ValueError(
            'Cannot estimate the source distance: need peaks in straws at more '
            'than one depth for the same straw index.'
        )
    cov = np.linalg.inv(fit.jac.T @ fit.jac) * np.sum(fit.fun**2) / (len(y) - nparam)
    return sc.scalar(fit.x[-1], variance=cov[-1, -1], unit=unit)
