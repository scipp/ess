# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import scipp as sc

from .mtz_io import DEFAULT_WAVELENGTH_COORD_NAME, NMXMtzDataArray

# User defined or configurable types
WavelengthBinSize = NewType("WavelengthBinSize", int)
"""The size of the wavelength(LAMBDA) bins."""
WavelengthRange = NewType("WavelengthRange", tuple[float, float])
"""The range of the wavelength(LAMBDA) bins."""
WavelengthBinCutProportion = NewType("WavelengthBinCutProportion", float)
"""The proportion of the wavelength(LAMBDA) bins to be cut off on both sides."""
DEFAULT_WAVELENGTH_CUT_PROPORTION = WavelengthBinCutProportion(0.25)
"""Default proportion of the wavelength(LAMBDA) bins to be cut off from both sides."""
ReferenceWavelength = NewType("ReferenceWavelength", sc.Variable)
"""The wavelength to select reference intensities."""
NRoot = NewType("NRoot", int)
"""The n-th root to be taken for the standard deviation."""
NRootStdDevCut = NewType("NRootStdDevCut", float)
"""The number of standard deviations to be cut from the n-th root data."""

# Computed types
"""Filtered mtz dataframe by the quad root of the sample standard deviation."""
WavelengthBinned = NewType("WavelengthBinned", sc.DataArray)
"""Binned mtz dataframe by wavelength(LAMBDA) with derived columns."""
FilteredWavelengthBinned = NewType("FilteredWavelengthBinned", sc.DataArray)
"""Filtered binned data."""
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


def get_wavelength_binned(
    mtz_da: NMXMtzDataArray,
    wavelength_bin_size: WavelengthBinSize,
    wavelength_range: Optional[WavelengthRange] = None,
) -> WavelengthBinned:
    """Bin the whole dataset by wavelength(LAMBDA).

    Notes
    -----
        Wavelength(LAMBDA) binning should always be done on the merged dataset.

    """
    if wavelength_range is None:
        binning_var = wavelength_bin_size
    else:
        binning_var = sc.linspace(
            dim=DEFAULT_WAVELENGTH_COORD_NAME,
            start=wavelength_range[0],
            stop=wavelength_range[1],
            num=wavelength_bin_size,
            unit=mtz_da.coords[DEFAULT_WAVELENGTH_COORD_NAME].unit,
        )

    binned = mtz_da.bin({DEFAULT_WAVELENGTH_COORD_NAME: binning_var})

    return WavelengthBinned(binned)


def filter_wavelegnth_binned(
    binned: WavelengthBinned,
    cut_proportion: WavelengthBinCutProportion = DEFAULT_WAVELENGTH_CUT_PROPORTION,
) -> FilteredWavelengthBinned:
    """Filter the binned data by cutting off the edges.

    Parameters
    ----------
    binned:
        The binned data by wavelength(LAMBDA).

    cut_proportion:
        The proportion of the wavelength(LAMBDA) bins to be cut off on both sides.
        The default value is :attr:`~DEFAULT_WAVELENGTH_CUT_PROPORTION`.

    Returns
    -------
    :
        The filtered binned data.

    """

    if cut_proportion < 0 or cut_proportion >= 0.5:
        raise ValueError(
            "The cut proportion should be in the range of 0 < proportion < 0.5."
        )

    cut_size = int(binned.sizes[DEFAULT_WAVELENGTH_COORD_NAME] * cut_proportion)
    return FilteredWavelengthBinned(
        binned[DEFAULT_WAVELENGTH_COORD_NAME, cut_size:-cut_size]
    )


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
    binned: FilteredWavelengthBinned,
    reference_wavelength: Optional[ReferenceWavelength] = None,
) -> SelectedReferenceWavelength:
    """Select the reference wavelength.

    Parameters
    ----------
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
    binned: FilteredWavelengthBinned,
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
    binned: FilteredWavelengthBinned,
    scale_factor: EstimatedScaleFactor,
) -> EstimatedScaledIntensities:
    """Scale the intensities by the estimated scale factor.

    Parameters
    ----------
    binned:
        Binned data by wavelength(LAMBDA) to be grouped and scaled.

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
    return EstimatedScaledIntensities(
        sc.nanmean(grouped.bins.nanmean() * sc.values(scale_factor), dim="hkl_asu")
    )


def _calculate_sample_standard_deviation(var: sc.Variable) -> sc.Variable:
    """Calculate the sample variation of the data.

    This helper function is a temporary solution before
    we release new scipp version with the statistics helper.
    """
    import numpy as np

    return sc.scalar(np.nanstd(var.values))


def cut_estimated_scaled_intensities_by_n_root_std_dev(
    scaled_intensities: EstimatedScaledIntensities,
    n_root: NRoot,
    n_root_std_dev_cut: NRootStdDevCut,
) -> FilteredEstimatedScaledIntensities:
    """Filter the mtz data array by the quad root of the sample standard deviation.

    Parameters
    ----------
    scaled_intensities:
        The scaled intensities to be filtered.

    n_root:
        The n-th root to be taken for the standard deviation.
        Higher n-th root means cutting is more effective on the right tail.
        More explanation can be found in the notes.

    n_root_std_dev_cut:
        The number of standard deviations to be cut from the n-th root data.

    Returns
    -------
    :
        The filtered scaled intensities.

    """
    # Check the range of the n-th root
    if n_root < 1:
        raise ValueError("The n-th root should be equal to or greater than 1.")

    copied = scaled_intensities.copy(deep=False)
    # Take the midpoints of the wavelength bin coordinates
    # to represent the average wavelength of the bin
    # It is because the bin-edges are dropped while flattening the data
    copied.coords[DEFAULT_WAVELENGTH_COORD_NAME] = sc.midpoints(
        copied.coords[DEFAULT_WAVELENGTH_COORD_NAME],
    )
    nth_root = copied.data ** (1 / n_root)
    # Calculate the mean
    nth_root_mean = nth_root.mean()
    # Calculate the sample standard deviation
    nth_root_std_dev = _calculate_sample_standard_deviation(nth_root)
    # Calculate the cut value
    half_window = n_root_std_dev_cut * nth_root_std_dev
    keep_range = (nth_root_mean - half_window, nth_root_mean + half_window)

    # Filter the data
    return FilteredEstimatedScaledIntensities(
        copied[(nth_root > keep_range[0]) & (nth_root < keep_range[1])]
    )


# Providers and default parameters
scaling_providers = (
    cut_estimated_scaled_intensities_by_n_root_std_dev,
    get_wavelength_binned,
    filter_wavelegnth_binned,
    get_reference_wavelength,
    get_reference_intensities,
    estimate_scale_factor_per_hkl_asu_from_reference,
    average_roughly_scaled_intensities,
)
"""Providers for scaling data."""

scaling_params = {
    WavelengthBinCutProportion: DEFAULT_WAVELENGTH_CUT_PROPORTION,
}
