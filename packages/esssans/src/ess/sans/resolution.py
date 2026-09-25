# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Q resolution of I(Q).

The resolution of a single element (pixel and wavelength bin) ``i`` is approximated as
a Gaussian with variance (Mildner and Carpenter, keeping ``cos(theta)``)

.. code-block:: text

    sigma_i**2 = (2 pi cos(theta) / lambda)**2
                 * [3 (R1/L1)**2 + 3 (R2/L3)**2 + (dR cos(2 theta) / L2)**2] / 12
               + Q_i**2 * sigma_lambda**2 / lambda**2

    1/L3 = 1/L1 + 1/L2
    sigma_lambda**2 = sigma_source**2 + d_lambda**2 / 12

with ``d_lambda`` the width of the wavelength bin. Since I(Q) in a bin is
``sum(C_i) / sum(N_i)``, with ``N_i`` the denominator, the resolution function of the
bin is the mixture of the element Gaussians weighted by ``N_i``. Its variance is
computed from sums, which can be merged across runs by addition:

.. code-block:: text

    S0 = sum(N_i)
    S1 = sum(N_i Q_i)
    S2 = sum(N_i (sigma_i**2 + Q_i**2))
    sigma_bin**2 = S2/S0 - (S1/S0)**2

``S1`` and ``S2`` are the :py:class:`ResolutionFirstMoment` and
:py:class:`ResolutionSecondMoment` parts of I(Q), processed by the same generic
providers as numerator and denominator. The variance must be computed only
after all merging. See https://github.com/scipp/ess/issues/763 for details.
"""

import numpy as np
import scipp as sc
import scippnexus as snx

from ess.reduce.unwrap import wavelength_spread

from .common import mask_range
from .conversions import ElasticCoordTransformGraph
from .types import (
    BinnedQ,
    CollimationLength,
    Denominator,
    DetectorLtotal,
    DetectorPixelSize,
    DetectorQVariance,
    LookupTable,
    MonitorTerm,
    NormalizedQ,
    QDetector,
    QResolution,
    ReducedQ,
    ResolutionFirstMoment,
    ResolutionMoment,
    ResolutionSecondMoment,
    RunType,
    SampleApertureRadius,
    SourceApertureRadius,
    SourceWavelengthSpread,
    WavelengthBins,
    WavelengthMask,
)


def source_wavelength_spread_from_lookup_table(
    table: LookupTable[RunType, snx.NXdetector],
    ltotal: DetectorLtotal[RunType],
    wavelength_bins: WavelengthBins,
) -> SourceWavelengthSpread[RunType]:
    """
    Wavelength spread from the variance stored in the wavelength lookup table.

    Uses the lookup table before masking large uncertainties, since the denominator
    includes all pixels and wavelengths.
    """
    return SourceWavelengthSpread[RunType](
        wavelength_spread(
            table, ltotal=ltotal, wavelength=sc.midpoints(wavelength_bins)
        )
    )


def _ratio_squared(a: sc.Variable, b: sc.Variable) -> sc.Variable:
    return (a / b).to(unit='') ** 2


def detector_q_variance(
    detector_term: QDetector[RunType, Denominator],
    graph: ElasticCoordTransformGraph[RunType],
    source_spread: SourceWavelengthSpread[RunType],
    wavelength_bins: WavelengthBins,
    source_aperture: SourceApertureRadius,
    sample_aperture: SampleApertureRadius,
    collimation_length: CollimationLength,
    pixel_size: DetectorPixelSize,
) -> DetectorQVariance[RunType]:
    """
    Variance ``sigma**2`` of the Q resolution of each pixel and wavelength.

    See the module docstring for the definition.
    """
    coords = detector_term.transform_coords(
        ('two_theta', 'L2'), graph=graph, keep_intermediate=False, rename_dims=False
    ).coords
    q = coords['Q']
    wavelength = coords['wavelength']
    two_theta = coords['two_theta']
    l2 = coords['L2']
    l3_inv = sc.reciprocal(collimation_length) + sc.reciprocal(l2)

    d_wavelength = wavelength_bins[1:] - wavelength_bins[:-1]
    wavelength_variance = source_spread**2 + d_wavelength**2 / 12
    angular_variance = (
        3 * _ratio_squared(source_aperture, collimation_length)
        + 3 * (sample_aperture * l3_inv).to(unit='') ** 2
        + _ratio_squared(pixel_size * sc.cos(two_theta), l2)
    ) / 12
    variance = (2 * np.pi * sc.cos(two_theta / 2) / wavelength) ** 2 * angular_variance
    variance += q**2 * (wavelength_variance / wavelength**2).to(unit='')
    return DetectorQVariance[RunType](
        variance.to(unit=q.unit**2).transpose(detector_term.dims)
    )


def resolution_first_moment(
    detector_term: QDetector[RunType, Denominator],
) -> QDetector[RunType, ResolutionFirstMoment]:
    """
    Compute ``N*Q`` for each pixel and wavelength.

    ``N`` are the values of the detector-dependent factor of the denominator. The
    monitor-dependent factor is applied after binning in Q, as for the denominator.
    """
    q = detector_term.coords['Q']
    return QDetector[RunType, ResolutionFirstMoment](
        detector_term.assign(sc.values(detector_term.data) * q)
    )


def resolution_second_moment(
    detector_term: QDetector[RunType, Denominator],
    variance: DetectorQVariance[RunType],
) -> QDetector[RunType, ResolutionSecondMoment]:
    """
    Compute ``N*(sigma**2 + Q**2)`` for each pixel and wavelength.

    ``N`` are the values of the detector-dependent factor of the denominator. The
    monitor-dependent factor is applied after binning in Q, as for the denominator.
    """
    q = detector_term.coords['Q']
    return QDetector[RunType, ResolutionSecondMoment](
        detector_term.assign(sc.values(detector_term.data) * (variance + q**2))
    )


def mask_and_scale_resolution_moment(
    da: BinnedQ[RunType, ResolutionMoment],
    mask: WavelengthMask,
    wavelength_term: MonitorTerm[RunType],
) -> NormalizedQ[RunType, ResolutionMoment]:
    """
    Apply the monitor-dependent factor of the denominator and the wavelength mask.

    Counterpart of :py:func:`ess.sans.conversions.mask_and_scale_wavelength_q`.
    """
    da = da * sc.values(wavelength_term)
    if mask is not None:
        da = mask_range(da, mask=mask)
    return NormalizedQ[RunType, ResolutionMoment](da)


def q_resolution(
    denominator: ReducedQ[RunType, Denominator],
    first_moment: ReducedQ[RunType, ResolutionFirstMoment],
    second_moment: ReducedQ[RunType, ResolutionSecondMoment],
) -> QResolution[RunType]:
    """
    Standard deviation of the resolution function of each bin of I(Q).

    Must be computed from sums that include all runs, see the module docstring.
    """
    s0 = sc.values(denominator.data)
    q_mean = first_moment.data / s0
    variance = second_moment.data / s0 - q_mean**2
    return QResolution[RunType](
        denominator.assign(sc.sqrt(variance)).assign_coords(Q_mean=q_mean)
    )


providers = (
    source_wavelength_spread_from_lookup_table,
    detector_q_variance,
    resolution_first_moment,
    resolution_second_moment,
    mask_and_scale_resolution_moment,
    q_resolution,
)
