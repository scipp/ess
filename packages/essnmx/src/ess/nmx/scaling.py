# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME, NMXMtzDataArray
from .reduction import _group

# User defined or configurable types
WavelengthBinSize = NewType("WavelengthBinSize", int)
"""The size of the wavelength(LAMBDA) bins."""


# Computed types
WavelengthBinned = NewType("WavelengthBinned", sc.DataArray)
"""Binned mtz dataframe by wavelength(LAMBDA) with derived columns."""
ReferenceWavelengthBin = NewType("ReferenceWavelengthBin", sc.DataArray)
"""The reference bin in the binned dataset."""
ReferenceScaleFactor = NewType("ReferenceScaleFactor", sc.DataArray)
"""The reference scale factor, grouped by HKL_EQ."""
ScaleFactorIntensity = NewType("ScaleFactorIntensity", float)
"""The scale factor for intensity."""
ScaleFactorSigmaIntensity = NewType("ScaleFactorSigmaIntensity", float)
"""The scale factor for the standard uncertainty of intensity."""
WavelengthScaled = NewType("WavelengthScaled", sc.DataArray)
"""Scaled wavelength by the reference bin."""


def _is_bin_empty(binned: sc.DataArray, idx: int) -> bool:
    return binned[idx].values.size == 0


def get_lambda_binned(
    mtz_da: NMXMtzDataArray,
    wavelength_bin_size: WavelengthBinSize,
) -> WavelengthBinned:
    """Bin the whole dataset by wavelength(LAMBDA).

    Notes
    -----
        Wavelength(LAMBDA) binning should always be done on the merged dataset.

    """

    return WavelengthBinned(
        mtz_da.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: wavelength_bin_size})
    )


def get_reference_bin(binned: WavelengthBinned) -> ReferenceWavelengthBin:
    """Find the reference group in the binned dataset.

    The reference group is the group in the middle of the binned dataset.
    If the middle group is empty, the function will search for the nearest.

    Parameters
    ----------
    binned:
        The wavelength binned data.

    Raises
    ------
    ValueError:
        If no reference group is found.

    """
    middle_number, offset = len(binned) // 2, 0

    while 0 < (cur_idx := middle_number + offset) < len(binned) and _is_bin_empty(
        binned, cur_idx
    ):
        offset = -offset + 1 if offset <= 0 else -offset

    if _is_bin_empty(binned, cur_idx):
        raise ValueError("No reference group found.")

    return binned[cur_idx].values.copy(deep=False)


def calculate_scale_factor_per_hkl_eq(
    ref_bin: ReferenceWavelengthBin,
) -> ReferenceScaleFactor:
    grouped = _group(ref_bin, "hkl_eq", hkl_eq=str)

    scale_factor_coords = ("I", "SIGI")
    for coord_name in scale_factor_coords:
        grouped.coords[f"scale_factor_{coord_name}"] = sc.concat(
            [sc.mean(1 / gr.values.coords[coord_name]) for gr in grouped],
            dim=grouped.dim,
        )

    return ReferenceScaleFactor(grouped)


# Providers and default parameters
scaling_providers = (
    get_lambda_binned,
    get_reference_bin,
    calculate_scale_factor_per_hkl_eq,
)
"""Providers for scaling data."""
