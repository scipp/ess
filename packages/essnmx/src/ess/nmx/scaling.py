# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NewType, TypeVar

import scipp as sc

from .mtz_io import NMXMtzDataArray

# User defined or configurable types
WavelengthBins = NewType("WavelengthBins", sc.Variable | int)
"""User configurable wavelength binning"""
ReferenceWavelength = NewType("ReferenceWavelength", sc.Variable | None)
"""The wavelength to select reference intensities."""

# Computed types
"""Filtered mtz dataframe by the quad root of the sample standard deviation."""
WavelengthBinned = NewType("WavelengthBinned", sc.DataArray)
"""Binned mtz dataframe by wavelength(LAMBDA) with derived columns."""
SelectedReferenceWavelength = NewType("SelectedReferenceWavelength", sc.Variable)
"""The wavelength to select reference intensities."""
ReferenceIntensities = NewType("ReferenceIntensities", sc.DataArray)
"""Reference intensities selected by the wavelength."""
EstimatedScaleFactor = NewType("EstimatedScaleFactor", sc.DataArray)
"""The estimated scale factor from the reference intensities per ``hkl_asu``."""
EstimatedScaledIntensities = NewType("EstimatedScaledIntensities", sc.DataArray)
"""Scaled intensities by the estimated scale factor."""
FilteredEstimatedScaledIntensities = NewType(
    "FilteredEstimatedScaledIntensities", sc.DataArray
)

T = TypeVar("T")


def get_wavelength_binned(
    mtz_da: NMXMtzDataArray,
    wavelength_bins: WavelengthBins,
) -> WavelengthBinned:
    """Bin the whole dataset by wavelength(LAMBDA).

    Parameters
    ----------
    mtz_da:
        The merged dataset.

    wavelength_bins:
        The wavelength(LAMBDA) bins.

    Notes
    -----
        Wavelength(LAMBDA) binning should always be done on the merged dataset.

    """
    return WavelengthBinned(mtz_da.bin({"wavelength": wavelength_bins}))


def _is_bin_empty(binned: sc.DataArray, idx: int) -> bool:
    """Check if the bin is empty."""
    return binned[idx].values.size == 0


def _get_middle_bin_idx(binned: sc.DataArray) -> int:
    """Find the middle bin index.

    If the middle one is empty, the function will search for the nearest.
    """
    middle_number, offset = len(binned) // 2, 0

    while 0 < (cur_idx := middle_number + offset) < len(binned) and _is_bin_empty(
        binned, cur_idx
    ):
        offset = -offset + 1 if offset <= 0 else -offset

    if _is_bin_empty(binned, cur_idx):
        raise ValueError("No reference group found.")

    return cur_idx


def get_reference_wavelength(
    binned: WavelengthBinned,
    reference_wavelength: ReferenceWavelength,
) -> SelectedReferenceWavelength:
    """Select the reference wavelength.

    Parameters
    ----------
    binned:
        The wavelength binned data.

    reference_wavelength:
        The reference wavelength to select the intensities.
        If ``None``, the middle group is selected.
        It should be a scalar variable as it is selecting one of bins.

    """
    if reference_wavelength is None:
        ref_idx = _get_middle_bin_idx(binned)
        return SelectedReferenceWavelength(binned.coords["wavelength"][ref_idx])
    else:
        return SelectedReferenceWavelength(reference_wavelength)


def get_reference_intensities(
    binned: WavelengthBinned,
    reference_wavelength: SelectedReferenceWavelength,
) -> ReferenceIntensities:
    """Find the reference intensities by the wavelength.

    Parameters
    ----------
    binned:
        The wavelength binned data.

    reference_wavelength:
        The reference wavelength to select the intensities.

    Raises
    ------
    ValueError:
        If no reference group is found.

    """
    if reference_wavelength is None:
        ref_idx = _get_middle_bin_idx(binned)
        return binned[ref_idx].values.copy(deep=False)
    else:
        if reference_wavelength.dims:
            raise ValueError("Reference wavelength should be a scalar.")
        try:
            return binned["wavelength", reference_wavelength].values.copy(deep=False)
        except IndexError as err:
            raise IndexError(f"{reference_wavelength} out of range.") from err


def estimate_scale_factor_per_hkl_asu_from_reference(
    reference_intensities: ReferenceIntensities,
) -> EstimatedScaleFactor:
    """Calculate the estimated scale factor per ``hkl_asu``.

    The estimated scale factor is calculatd as the average
    of the inverse of the non-empty reference intensities.

    It is part of the calculation of estimated scaled intensities
    for fitting the scaling model.

    .. math::

        EstimatedScaleFactor_{(hkl)} = \\dfrac{
            \\sum_{i=1}^{N_{(hkl)}} \\dfrac{1}{I_{i}}
        }{
            N_{(hkl)}
        }
        = average( \\dfrac{1}{I_{(hkl)}} )

    Estimated scale factor is calculated per ``hkl_asu``.
    This is part of the calculation of roughly-scaled-intensities
    for fitting the scaling model.
    The whole procedure is described in :func:`average_roughly_scaled_intensities`.

    Parameters
    ----------
    reference_intensities:
        The reference intensities selected by wavelength.

    Returns
    -------
    :
        The estimated scale factor per ``hkl_asu``.
        The result should have a dimension of ``hkl_asu``.

        It does not have a dimension of ``wavelength`` since
        it is calculated from the reference intensities,
        which is selected by one ``wavelength``.

    """
    # Workaround for https://github.com/scipp/scipp/issues/3046
    # and https://github.com/scipp/scipp/issues/3425
    import numpy as np

    unique_hkl = np.unique(reference_intensities.coords["hkl_asu"].values)
    group_var = sc.array(dims=["hkl_asu"], values=unique_hkl)
    grouped = reference_intensities.group(group_var)

    return EstimatedScaleFactor((1 / grouped).bins.mean())


def average_roughly_scaled_intensities(
    binned: WavelengthBinned,
    scale_factor: EstimatedScaleFactor,
) -> EstimatedScaledIntensities:
    """Scale the intensities by the estimated scale factor.

    Parameters
    ----------
    binned:
        Intensities binned in the wavelength dimension.
        It will be grouped by reflection (hkl) in the process.

    scale_factor:
        The estimated scale factor per reflection(hkl) of the reference wavelength bin.
        See :func:`estimate_scale_factor_per_hkl_asu_from_reference`
        for the calculation of the estimated scale factor.

        .. math::

            EstimatedScaleFactor_{(hkl)} =
            average( \\dfrac{1}{I_{\\lambda=reference, (hkl)}} )

    Returns
    -------
    :
        Average scaled intensities on ``hkl(asu)`` indices per wavelength.

    Notes
    -----
    The average of roughly scaled intensities are calculated by the following formula:

    .. math::

        EstimatedScaledI_{\\lambda}
        = \\dfrac{
            \\sum_{i=1}^{N_{\\lambda, (hkl)}}
            EstimatedScaledI_{\\lambda, (hkl)}
        }{
            N_{\\lambda, (hkl)}
        }

    And scaled intensities on each ``hkl(asu)`` indices per wavelength
    are calculated by the following formula:

    .. math::
        :nowrap:

        \\begin{eqnarray}
        EstimatedScaledI_{\\lambda, (hkl)} \\\\
        = \\dfrac{
            \\sum_{i=1}^{N_{\\lambda=reference, (hkl)}}
            \\sum_{j=1}^{N_{\\lambda, (hkl)}}
            \\dfrac{I_{j}}{I_{i}}
        }{
            N_{\\lambda=reference, (hkl)}*N_{\\lambda, (hkl)}
        } \\\\
        = \\dfrac{
            \\sum_{i=1}^{N_{\\lambda=reference, (hkl)}} \\dfrac{1}{I_{i}}
        }{
            N_{\\lambda=reference, (hkl)}
        } * \\dfrac{
            \\sum_{j=1}^{N_{\\lambda, (hkl)}} I_{j}
        }{
            N_{\\lambda, (hkl)}
        } \\\\
        = average( \\dfrac{1}{I_{\\lambda=reference, (hkl)}} )
        * average( I_{\\lambda, (hkl)} )
        \\end{eqnarray}

    Therefore the ``binned(wavelength dimension)`` should be
    grouped along the ``hkl(asu)`` coordinate in the calculation.

    """
    # Group by HKL_EQ of the estimated scale factor from reference intensities
    grouped = binned.group(scale_factor.coords["hkl_asu"])

    # Drop variances of the scale factor
    # Scale each group each bin by the scale factor
    intensities = sc.nanmean(
        grouped.bins.nanmean() * sc.values(scale_factor), dim="hkl_asu"
    )
    # Take the midpoints of the wavelength bin coordinates
    # to represent the average wavelength of the bin
    # It is because the bin-edges are dropped while flattening the data
    # and the data is expected to be filtered after this step.
    intensities.coords["wavelength"] = sc.midpoints(
        intensities.coords["wavelength"],
    )
    return EstimatedScaledIntensities(intensities)


ScaledIntensityLeftTailThreshold = NewType(
    "ScaledIntensityLeftTailThreshold", sc.Variable
)
"""The threshold to cut the left tail of the estimated scaled intensities."""
DEFAULT_LEFT_TAIL_THRESHOLD = ScaledIntensityLeftTailThreshold(sc.scalar(0.1))
ScaledIntensityRightTailThreshold = NewType(
    "ScaledIntensityRightTailThreshold", sc.Variable
)
"""The threshold to cut the right tail of the estimated scaled intensities."""
DEFAULT_RIGHT_TAIL_THRESHOLD = ScaledIntensityRightTailThreshold(sc.scalar(2.0))


def cut_tails(
    scaled_intensities: EstimatedScaledIntensities,
    left_threashold: ScaledIntensityLeftTailThreshold = DEFAULT_LEFT_TAIL_THRESHOLD,
    right_threshold: ScaledIntensityRightTailThreshold = DEFAULT_RIGHT_TAIL_THRESHOLD,
) -> FilteredEstimatedScaledIntensities:
    """Cut the right tail of the estimated scaled intensities by the threshold.

    Parameters
    ----------
    scaled_intensities:
        The scaled intensities to be filtered.

    left_threashold:
        The threshold to be cut from the left tail.

    right_threshold:
        The threshold to be cut from the right tail.

    Returns
    -------
    :
        The filtered scaled intensities with the tails cut.

    """
    return FilteredEstimatedScaledIntensities(
        scaled_intensities[
            (scaled_intensities.data > left_threashold)
            & (scaled_intensities.data < right_threshold)
        ].copy(deep=False)
    )


@dataclass
class FittingResult:
    """Result of the fitting process."""

    fitting_func: Callable[..., sc.DataArray]
    """The fitting function to be used for fitting."""
    params: Mapping
    """Parameters of the fitting function."""
    covariance: Mapping
    """Covariance of the :attr:`~FittingParams`."""
    fit_output: sc.DataArray
    """The final output of the fitting function."""


def polyval_wavelength(
    wavelength: sc.Variable, *, out_unit: str, **kwargs
) -> sc.DataArray:
    """Polynomial helper for fitting.

    The coefficients are adjusted to make the fitting result
    have ``out_unit`` as unit.

    Parameters
    ----------
    wavelength:
        The wavelength coordinate.
    out_unit:
        The unit of the output.
    **kwargs:
        The polynomial coefficients.

    Returns
    -------
    :
        The polynomial calculated at the wavelength.


    """
    out = sc.zeros_like(wavelength)
    out.unit = out_unit
    xk = sc.ones_like(wavelength)
    for _, arg_value in enumerate(kwargs.values()):
        out += sc.values(arg_value) * xk * sc.scalar(1.0, unit=out.unit / xk.unit)
        xk *= wavelength
    return out


WavelengthFittingPolynomialDegree = NewType("WavelengthFittingPolynomialDegree", int)
DEFAULT_WAVELENGTH_FITTING_POLYNOMIAL_DEGREE = WavelengthFittingPolynomialDegree(7)


def fit_wavelength_scale_factor_polynomial(
    estimated_intensities: FilteredEstimatedScaledIntensities,
    *,
    n_degree: WavelengthFittingPolynomialDegree,
) -> FittingResult:
    """Fit the wavelength scale factor polynomial.

    It uses :func:`polyval_wavelength` as the fitting function
    and :func:`scipp.optimize.curve_fit` for the fitting process.
    The initial guess for the polynomial coefficients is set to 1
    for all degrees.
    The unit of the coefficients is adjusted to make the fitting result
    dimensionless.

    Parameters
    ----------
    estimated_intensities:
        The estimated scaled intensities to be fitted.
    n_degree:
        The degree of the polynomial to be fitted.

    Returns
    -------
    :
        The fitting result.

    """

    from functools import partial

    fitting_func = partial(polyval_wavelength, out_unit="dimensionless")
    p_result, cov_result = sc.curve_fit(
        coords=["wavelength"],
        f=fitting_func,
        da=estimated_intensities,
        p0={f"arg{i}": sc.scalar(1) for i in range(n_degree)},
    )
    data = fitting_func(estimated_intensities.coords["wavelength"], **p_result)
    return FittingResult(
        fitting_func=fitting_func,
        params=p_result,
        covariance=cov_result,
        fit_output=sc.DataArray(
            data=data.data,
            coords={"wavelength": estimated_intensities.coords["wavelength"]},
        ),
    )


WavelengthScaleFactors = NewType("WavelengthScaleFactors", sc.DataArray)
"""The scale factors of `"wavelength"`."""


def calculate_wavelength_scale_factor(
    fitting_result: FittingResult,
    reference_wavelength: SelectedReferenceWavelength,
) -> WavelengthScaleFactors:
    """Calculate the scale factors along the `"wavelength"`."""

    scaled_reference = fitting_result.fitting_func(
        reference_wavelength, **fitting_result.params
    )
    scale_factor = fitting_result.fit_output / scaled_reference
    return WavelengthScaleFactors(scale_factor)


providers = (
    cut_tails,
    get_wavelength_binned,
    get_reference_wavelength,
    get_reference_intensities,
    estimate_scale_factor_per_hkl_asu_from_reference,
    average_roughly_scaled_intensities,
    fit_wavelength_scale_factor_polynomial,
    calculate_wavelength_scale_factor,
)
"""Providers for scaling data."""

default_parameters = {
    WavelengthBins: sc.linspace("wavelength", 2.6, 3.6, 250, unit="angstrom"),
    ScaledIntensityLeftTailThreshold: DEFAULT_LEFT_TAIL_THRESHOLD,
    ScaledIntensityRightTailThreshold: DEFAULT_RIGHT_TAIL_THRESHOLD,
    WavelengthFittingPolynomialDegree: WavelengthFittingPolynomialDegree(7),
}
"""Default parameters for scaling data."""
