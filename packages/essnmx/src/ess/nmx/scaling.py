# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .mtz_io import DEFUAULT_WAVELENGTH_COLUMN_NAME, NMXMtzDataArray

# User defined or configurable types
WavelengthBinSize = NewType("WavelengthBinSize", int)
"""The size of the wavelength(LAMBDA) bins."""


# Computed types
WavelengthBinned = NewType("WavelengthBinned", sc.DataArray)
"""Binned mtz dataframe by wavelength(LAMBDA) with derived columns."""
ReferenceWavelengthBin = NewType("ReferenceWavelengthBin", sc.DataArray)
"""The reference bin in the binned dataset."""
ScaleFactorIntensity = NewType("ScaleFactorIntensity", float)
"""The scale factor for intensity."""
ScaleFactorSigmaIntensity = NewType("ScaleFactorSigmaIntensity", float)
"""The scale factor for the standard uncertainty of intensity."""
WavelengthScaled = NewType("WavelengthScaled", sc.DataArray)
"""Scaled wavelength by the reference bin."""


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
        mtz_da.bin({DEFUAULT_WAVELENGTH_COLUMN_NAME: wavelength_bin_size})
    )


def _is_bin_empty(binned: sc.DataArray, idx: int) -> bool:
    return binned[idx].values.size == 0


def get_reference_bin(
    binned: WavelengthBinned,
) -> ReferenceWavelengthBin:
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

    ref: sc.DataArray = binned[cur_idx].values.copy(deep=False)
    grouped: sc.DataArray = ref.group("hkl_eq_hash")
    scale_factor_coords = ("I", "SIGI")
    for coord_name in scale_factor_coords:
        grouped.coords[f"scale_factor_{coord_name}"] = sc.concat(
            [sc.mean(1 / gr.values.coords[coord_name]) for gr in grouped],
            dim=grouped.dim,
        )

    return ReferenceWavelengthBin(grouped)


# Providers and default parameters
scaling_providers = (get_lambda_binned,)
"""Providers for scaling data."""
