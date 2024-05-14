# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional, TypeVar

import scipp as sc

from .mtz_io import DEFAULT_WAVELENGTH_COORD_NAME, NMXMtzDataArray

# User defined or configurable types
WavelengthBinSize = NewType("WavelengthBinSize", int)
"""The size of the wavelength(LAMBDA) bins."""
MinWavelengthBinEdge = NewType("MinWavelengthBinEdge", sc.Variable)
"""The minimum edge of the wavelength(LAMBDA) bins."""
DEFAULT_MIN_WAVELENGTH_BIN_EDGE = MinWavelengthBinEdge(sc.scalar(2.6, unit="angstrom"))
"""Default minimum edge of the wavelength(LAMBDA) bins."""
MaxWavelengthBinEdge = NewType("MaxWavelengthBinEdge", sc.Variable)
"""The maximum edge of the wavelength(LAMBDA) bins."""
DEFAULT_MAX_WAVELENGTH_BIN_EDGE = MaxWavelengthBinEdge(sc.scalar(3.6, unit="angstrom"))
"""Default maximum edge of the wavelength(LAMBDA) bins."""
ReferenceWavelength = NewType("ReferenceWavelength", sc.Variable)
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


def _if_not_none_else(x: T | None, default: T) -> T:
    """Ternary operation helper for optional arguments."""
    return x if x is not None else default


def get_wavelength_binned(
    mtz_da: NMXMtzDataArray,
    wavelength_bin_size: WavelengthBinSize,
    min_wavelength_bin_edge: Optional[MinWavelengthBinEdge] = None,
    max_wavelength_bin_edge: Optional[MaxWavelengthBinEdge] = None,
) -> WavelengthBinned:
    """Bin the whole dataset by wavelength(LAMBDA).

    Parameters
    ----------
    mtz_da:
        The merged dataset.

    wavelength_bin_size:
        The size of the wavelength(LAMBDA) bins.

    min_wavelength_bin_edge:
        The minimum edge of the wavelength(LAMBDA) bins.
        Minimum value of the wavelength(LAMBDA) coordinate will be used if ``None``.

    max_wavelength_bin_edge:
        The maximum edge of the wavelength(LAMBDA) bins.
        Maximum value of the wavelength(LAMBDA) coordinate will be used if ``None``.

    Notes
    -----
        Wavelength(LAMBDA) binning should always be done on the merged dataset.

    """
    wavelength_coord = mtz_da.coords[DEFAULT_WAVELENGTH_COORD_NAME]
    start = _if_not_none_else(min_wavelength_bin_edge, wavelength_coord.min())
    stop = _if_not_none_else(max_wavelength_bin_edge, wavelength_coord.max())
    binning_var = sc.linspace(
        dim=DEFAULT_WAVELENGTH_COORD_NAME,
        start=start,
        stop=stop,
        num=wavelength_bin_size,
        unit=wavelength_coord.unit,
    )

    return WavelengthBinned(mtz_da.bin({DEFAULT_WAVELENGTH_COORD_NAME: binning_var}))


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
    reference_wavelength: Optional[ReferenceWavelength] = None,
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
        return SelectedReferenceWavelength(
            binned.coords[DEFAULT_WAVELENGTH_COORD_NAME][ref_idx]
        )
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
        except IndexError:
            raise IndexError(f"{reference_wavelength} out of range.")


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
        The wavelength binned data.

    scale_factor:
        The estimated scale factor.

    Returns
    -------
    :
        Average scaled intensties on ``hkl(asu)`` indices per wavelength.

    Notes
    -----
    The average of roughly scaled intensities are calculated by the following formula:

    .. math::

        EstimatedScaledI_{\\lambda}
        = \\dfrac{
            \\sum_{k=1}^{N_{\\lambda, (hkl)}} EstimatedScaledI_{\\lambda, (hkl)}
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
            \\sum_{i=1}^{N_{reference, (hkl)}}
            \\sum_{j=1}^{N_{\\lambda, (hkl)}}
            \\dfrac{I_{j}}{I_{i}}
        }{
            N_{reference, (hkl)}*N_{\\lambda, (hkl)}
        } \\\\
        = \\dfrac{
            \\sum_{i=1}^{N_{reference, (hkl)}} \\dfrac{1}{I_{i}}
        }{
            N_{reference, (hkl)}
        } * \\dfrac{
            \\sum_{j=1}^{N_{\\lambda, (hkl)}} I_{j}
        }{
            N_{\\lambda, (hkl)}
        } \\\\
        = average( \\dfrac{1}{I_{ref, (hkl)}} ) * average( I_{\\lambda, (hkl)} )
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
    intensities.coords[DEFAULT_WAVELENGTH_COORD_NAME] = sc.midpoints(
        intensities.coords[DEFAULT_WAVELENGTH_COORD_NAME],
    )
    return EstimatedScaledIntensities(intensities)


ScaledIntensityLeftTailThreshold = NewType(
    "ScaledIntensityLeftTailThreshold", sc.Variable
)
DEFAULT_LEFT_TAIL_THRESHOLD = ScaledIntensityLeftTailThreshold(sc.scalar(0.1))
ScaledIntensityRightTailThreshold = NewType(
    "ScaledIntensityRightTailThreshold", sc.Variable
)
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


# Providers and default parameters
scaling_providers = (
    cut_tails,
    get_wavelength_binned,
    get_reference_wavelength,
    get_reference_intensities,
    estimate_scale_factor_per_hkl_asu_from_reference,
    average_roughly_scaled_intensities,
)
"""Providers for scaling data."""

scaling_params = {
    MinWavelengthBinEdge: DEFAULT_MIN_WAVELENGTH_BIN_EDGE,
    MaxWavelengthBinEdge: DEFAULT_MAX_WAVELENGTH_BIN_EDGE,
    ScaledIntensityLeftTailThreshold: DEFAULT_LEFT_TAIL_THRESHOLD,
    ScaledIntensityRightTailThreshold: DEFAULT_RIGHT_TAIL_THRESHOLD,
}
"""Default parameters for scaling data."""
