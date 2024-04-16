# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any, Callable, NewType, Sequence

import numpy as np
import scipp as sc

from .mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME, NMXMtzDataArray

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


def _is_bin_empty(binned: sc.DataArray, idx: int) -> bool:
    return binned[idx].values.size == 0


def _apply_elem_wise(
    func: Callable, var: sc.Variable, *, result_dtype: Any = None
) -> sc.Variable:
    """Apply a function element-wise to the variable values.

    This helper is only for vector-dtype variables.
    Use ``numpy.vectorize`` for other types.

    Parameters
    ----------
    func:
        The function to apply.
    var:
        The variable to apply the function to.
    result_dtype:
        The dtype of the resulting variable.
        It is needed especially when the function returns a vector.

    """

    def apply_func(val: Sequence, _cur_depth: int = 0) -> list:
        if _cur_depth == len(var.dims):
            return func(val)
        return [apply_func(v, _cur_depth + 1) for v in val]

    if result_dtype is None:
        return sc.Variable(
            dims=var.dims,
            values=apply_func(var.values),
        )
    return sc.Variable(
        dims=var.dims,
        values=apply_func(var.values),
        dtype=result_dtype,
    )


def _hash_repr(val: Any) -> int:
    """Hash the string representation of the value."""

    return hash(str(val))


def hash_variable(var: sc.Variable, hash_func: Callable = hash) -> sc.Variable:
    """Hash the coordinate values."""

    return _apply_elem_wise(hash_func, var)


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


def _detour_group(
    da: sc.DataArray, group_name: str, detour_func: Callable
) -> sc.DataArray:
    """Group the data array by a hash of a coordinate.

    It uses index of each unique hash value
    for grouping instead of hash value itself
    to avoid overflow issues.

    """
    from uuid import uuid4

    copied = da.copy(deep=False)

    # Temporary coords for grouping
    detour_idx_coord_name = uuid4().hex + "hash_idx"

    # Create a temporary detoured coordinate
    detour_var = _apply_elem_wise(detour_func, da.coords[group_name])
    # Create a temporary hash-index of each unique value
    unique_hashes = np.unique(detour_var.values)
    hash_to_idx = {hash_val: idx for idx, hash_val in enumerate(unique_hashes)}
    copied.coords[detour_idx_coord_name] = _apply_elem_wise(
        lambda idx: hash_to_idx[idx], detour_var
    )

    # Group by the hash-index
    grouped = copied.group(detour_idx_coord_name)

    # Restore the original values
    idx_to_detour = {idx: hash_val for hash_val, idx in hash_to_idx.items()}
    detour_to_var = {
        hash_val: var
        for var, hash_val in zip(da.coords[group_name].values, detour_var.values)
    }
    idx_to_var = {
        idx: detour_to_var[hash_val] for idx, hash_val in idx_to_detour.items()
    }
    grouped.coords[group_name] = _apply_elem_wise(
        lambda idx: idx_to_var[idx],
        grouped.coords[detour_idx_coord_name],
        result_dtype=da.coords[group_name].dtype,
    )
    # Rename dims back to group_name and drop the temporary hash-index coordinate
    return grouped.rename_dims({detour_idx_coord_name: group_name}).drop_coords(
        [detour_idx_coord_name]
    )


def group(da: sc.DataArray, /, *args: str, **group_detour_func_map) -> sc.DataArray:
    """Group the data array by the given coordinates.

    Parameters
    ----------
    da:
        The data array to group.
    args:
        The coordinates to group by.
    group_hash_func_map:
        The hash functions for each coordinate.

    Returns
    -------
    sc.DataArray
        The grouped data array.

    """
    grouped = da
    for group_name in args:
        if group_name in group_detour_func_map:
            grouped = _detour_group(
                grouped, group_name, group_detour_func_map[group_name]
            )
        else:
            try:
                grouped = sc.group(grouped, group_name)
            except Exception:
                grouped = _detour_group(
                    grouped, group_name, group_detour_func_map.get(group_name, hash)
                )

    return grouped


def calculate_scale_factor_per_hkl_eq(
    ref_bin: ReferenceWavelengthBin,
) -> ReferenceScaleFactor:
    grouped = group(ref_bin, "hkl_eq", hkl_eq=_hash_repr)

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
