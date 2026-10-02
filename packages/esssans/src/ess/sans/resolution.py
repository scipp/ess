# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Q resolution of I(Q).

The resolution of a single element (pixel and wavelength bin) ``i`` is approximated as
a Gaussian with variance (Mildner and Carpenter, keeping ``cos(theta)`` and
``cos(2 theta)``)

.. code-block:: text

    sigma_i**2 = (2 pi cos(theta) / lambda)**2
                 * [R1**2 / (4 L1**2) + R2**2 (1/L1 + cos(2 theta)/L2)**2 / 4
                    + sigma_pixel**2]
               + Q_i**2 * sigma_source**2 / lambda**2

with circular source and sample apertures of radius ``R1`` and ``R2``, collimation
length ``L1``, and ``sigma_pixel`` the standard deviation of the scattering angle
``2 theta`` within a pixel, see :py:func:`pixel_scattering_angle_variance`.
``sigma_source`` is the standard deviation of the true wavelength of neutrons detected
at the given pixel and wavelength.

The wavelength binning does not contribute: events are histogrammed in Q using their
own wavelength, so they do not spread beyond the Q bin they are counted in. The spread
of Q within a Q bin is accounted for below.

Since I(Q) in a bin is ``sum(C_i) / sum(N_i)``, with ``N_i`` the denominator, the
resolution function of the bin is the mixture of the element Gaussians weighted by
``N_i``. Its variance is computed from sums, which can be merged across runs by
addition:

.. code-block:: text

    S0 = sum(N_i)
    S1 = sum(N_i Q_i)
    S2 = sum(N_i (sigma_i**2 + Q_i**2))
    sigma_bin**2 = S2/S0 - (S1/S0)**2

``S0``, ``S1`` and ``S2`` are the :py:class:`ResolutionZerothMoment`,
:py:class:`ResolutionFirstMoment` and :py:class:`ResolutionSecondMoment` parts of
I(Q), processed by the same generic providers as numerator and denominator.
Elements whose events have no wavelength, because the lookup table is masked there,
contribute to the denominator but not to the counts, so they are excluded from the
sums. They are marked by a NaN :py:class:`SourceWavelengthSpread`. The variance must be computed only
after all merging. See https://github.com/scipp/ess/issues/763 for details.
"""

import numpy as np
import scipp as sc
import scippnexus as snx

from ess.reduce.unwrap import wavelength_spread

from .common import mask_range
from .conversions import ElasticCoordTransformGraph
from .normalization import pixel_cylinder
from .types import (
    BinnedQ,
    CollimationLength,
    CorrectedDetector,
    Denominator,
    DetectorLtotal,
    DetectorPixelShape,
    DetectorQVariance,
    EmptyDetector,
    LookupTable,
    LookupTableRelativeErrorThreshold,
    MonitorTerm,
    NeXusDetectorName,
    NeXusTransformation,
    NormalizedQ,
    Numerator,
    PixelScatteringAngleVariance,
    QDetector,
    QResolution,
    ReducedQ,
    ResolutionFirstMoment,
    ResolutionMoment,
    ResolutionSecondMoment,
    ResolutionZerothMoment,
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
    error_threshold: LookupTableRelativeErrorThreshold,
    detector_name: NeXusDetectorName,
) -> SourceWavelengthSpread[RunType]:
    """
    Wavelength spread from the variance stored in the wavelength lookup table.

    Where the relative spread exceeds the error threshold of the lookup table, events
    get no wavelength. The spread is NaN there, which excludes these pixels and
    wavelengths from the resolution. Since events get their wavelength by
    interpolating the table, this reproduces the masking of the table approximately.

    The table must store the true variance of the wavelength, as tables built from a
    simulation do. Tables built in ``analytical`` mode store half the range of
    possible wavelengths instead. With the default source time range of 5 ms and
    without pulse-shaping choppers, this is about 2.8 times the standard deviation for
    the ESS pulse.
    """
    wavelength = sc.midpoints(wavelength_bins)
    spread = wavelength_spread(table, ltotal=ltotal, wavelength=wavelength)
    masked = spread / wavelength > sc.scalar(error_threshold[detector_name])
    return SourceWavelengthSpread[RunType](
        sc.where(masked, sc.scalar(np.nan, unit=spread.unit), spread)
    )


def pixel_scattering_angle_variance(
    detector: EmptyDetector[RunType],
    pixel_shape: DetectorPixelShape[RunType],
    transform: NeXusTransformation[snx.NXdetector, RunType],
    graph: ElasticCoordTransformGraph[RunType],
) -> PixelScatteringAngleVariance[RunType]:
    """
    Variance of the scattering angle ``2 theta`` within each cylindrical pixel.

    Neutrons are assumed to be detected uniformly within the pixel volume. A
    displacement ``d`` from the pixel center changes the scattering angle by
    ``d . e / L2``, with ``e`` the unit vector perpendicular to the scattered beam, in
    the direction of increasing ``2 theta``:

    .. code-block:: text

        e = cos(2 theta) rho - sin(2 theta) k

    ``k`` is the incident beam direction and ``rho`` the unit vector from the beam axis
    towards the pixel, perpendicular to ``k``. For a cylinder of length ``l`` and radius
    ``r``, with ``c`` the cosine of the angle between its axis and ``e``, the variance
    of ``d . e`` is ``l**2 c**2 / 12 + r**2 (1 - c**2) / 4``.
    """
    coords = detector.transform_coords(
        ('incident_beam', 'scattered_beam'),
        graph=graph,
        keep_intermediate=False,
        rename_dims=False,
    ).coords
    k = coords['incident_beam'] / sc.norm(coords['incident_beam'])
    scattered = coords['scattered_beam']
    l2 = sc.norm(scattered)
    rho = scattered - sc.dot(scattered, k) * k
    rho_norm = sc.norm(rho)
    cos_2theta = sc.dot(scattered, k) / l2
    sin_2theta = rho_norm / l2
    e = cos_2theta * (rho / rho_norm) - sin_2theta * k

    axis, radius = pixel_cylinder(pixel_shape, transform)
    length = sc.norm(axis)
    c2 = sc.dot(axis / length, e) ** 2
    variance = length**2 * c2 / 12 + radius**2 * (1 - c2) / 4
    return PixelScatteringAngleVariance[RunType]((variance / l2**2).to(unit=''))


def detector_q_variance(
    detector_term: QDetector[RunType, Denominator],
    graph: ElasticCoordTransformGraph[RunType],
    source_spread: SourceWavelengthSpread[RunType],
    pixel_variance: PixelScatteringAngleVariance[RunType],
    source_aperture: SourceApertureRadius,
    sample_aperture: SampleApertureRadius,
    collimation_length: CollimationLength,
    numerator: CorrectedDetector[RunType, Numerator],
) -> DetectorQVariance[RunType]:
    """
    Variance ``sigma**2`` of the Q resolution of each pixel and wavelength.

    See the module docstring for the definition. Requires event data: if the counts
    were histogrammed in wavelength before conversion to Q, the width of the
    wavelength bins would have to be included.
    """
    if numerator.bins is None:
        raise ValueError(
            'The Q resolution requires event data. For data histogrammed in '
            'wavelength, the width of the wavelength bins would contribute.'
        )
    coords = detector_term.transform_coords(
        ('two_theta', 'L2'), graph=graph, keep_intermediate=False, rename_dims=False
    ).coords
    q = coords['Q']
    wavelength = coords['wavelength']
    two_theta = coords['two_theta']
    # Scattering point within the sample aperture, seen from the source and the pixel
    sample_term = sample_aperture * (
        sc.reciprocal(collimation_length) + sc.cos(two_theta) / coords['L2']
    )
    angular_variance = (
        (source_aperture / collimation_length).to(unit='') ** 2 / 4
        + sample_term.to(unit='') ** 2 / 4
        + pixel_variance
    )
    variance = (2 * np.pi * sc.cos(two_theta / 2) / wavelength) ** 2 * angular_variance
    variance += q**2 * (source_spread / wavelength).to(unit='') ** 2
    return DetectorQVariance[RunType](
        variance.to(unit=q.unit**2).transpose(detector_term.dims)
    )


def _resolution_weights(
    detector_term: sc.DataArray, variance: sc.Variable
) -> sc.Variable:
    """Values of ``N``, zero where the variance is NaN (see the module docstring)."""
    n = sc.values(detector_term.data)
    return sc.where(sc.isnan(variance), sc.zeros_like(n), n)


def resolution_zeroth_moment(
    detector_term: QDetector[RunType, Denominator],
    variance: DetectorQVariance[RunType],
) -> QDetector[RunType, ResolutionZerothMoment]:
    """
    Compute ``N`` for each pixel and wavelength.

    ``N`` are the values of the detector-dependent factor of the denominator. The
    monitor-dependent factor is applied after binning in Q, as for the denominator.
    """
    return QDetector[RunType, ResolutionZerothMoment](
        detector_term.assign(_resolution_weights(detector_term, variance))
    )


def resolution_first_moment(
    detector_term: QDetector[RunType, Denominator],
    variance: DetectorQVariance[RunType],
) -> QDetector[RunType, ResolutionFirstMoment]:
    """
    Compute ``N*Q`` for each pixel and wavelength.

    See :py:func:`resolution_zeroth_moment` for ``N``.
    """
    q = detector_term.coords['Q']
    return QDetector[RunType, ResolutionFirstMoment](
        detector_term.assign(_resolution_weights(detector_term, variance) * q)
    )


def resolution_second_moment(
    detector_term: QDetector[RunType, Denominator],
    variance: DetectorQVariance[RunType],
) -> QDetector[RunType, ResolutionSecondMoment]:
    """
    Compute ``N*(sigma**2 + Q**2)`` for each pixel and wavelength.

    See :py:func:`resolution_zeroth_moment` for ``N``.
    """
    q = detector_term.coords['Q']
    weights = _resolution_weights(detector_term, variance)
    # Avoid 0 * NaN for excluded elements
    variance = sc.where(sc.isnan(variance), sc.zeros_like(variance), variance)
    return QDetector[RunType, ResolutionSecondMoment](
        detector_term.assign(weights * (variance + q**2))
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
    zeroth_moment: ReducedQ[RunType, ResolutionZerothMoment],
    first_moment: ReducedQ[RunType, ResolutionFirstMoment],
    second_moment: ReducedQ[RunType, ResolutionSecondMoment],
) -> QResolution[RunType]:
    """
    Standard deviation of the resolution function of each bin of I(Q).

    Must be computed from sums that include all runs, see the module docstring.
    """
    s0 = zeroth_moment.data
    q_mean = first_moment.data / s0
    variance = second_moment.data / s0 - q_mean**2
    return QResolution[RunType](
        zeroth_moment.assign(sc.sqrt(variance)).assign_coords(Q_mean=q_mean)
    )


providers = (
    source_wavelength_spread_from_lookup_table,
    pixel_scattering_angle_variance,
    detector_q_variance,
    resolution_zeroth_moment,
    resolution_first_moment,
    resolution_second_moment,
    mask_and_scale_resolution_moment,
    q_resolution,
)
