# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.nmx.mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME
from ess.nmx.scaling import (
    ReferenceIntensities,
    estimate_scale_factor_per_hkl_asu_from_reference,
    get_reference_intensities,
)


@pytest.fixture
def nmx_data_array() -> sc.DataArray:
    da = sc.DataArray(
        data=sc.array(dims=["row"], values=[1, 2, 3, 4, 5, 3.1, 3.2]),
        coords={
            DEFAULT_WAVELENGTH_COLUMN_NAME: sc.Variable(
                dims=["row"], values=[1, 2, 3, 4, 5, 3, 3]
            ),
            "H_ASU": sc.array(dims=["row"], values=[1, 4, 7, 10, 13, 7, 9]),
            "K_ASU": sc.array(dims=["row"], values=[2, 5, 8, 11, 14, 8, 8]),
            "L_ASU": sc.array(dims=["row"], values=[3, 6, 9, 12, 15, 9, 7]),
        },
    )
    da.variances = (
        sc.array(dims=["row"], values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.31, 0.32]) ** 2
    )
    return da


def test_get_reference_bin_middle(nmx_data_array: sc.DataArray) -> None:
    """Test the middle bin."""

    ref_bin = get_reference_intensities(
        nmx_data_array.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: 6})
    )
    selected_idx = (2, 5, 6)
    assert all(
        ref_bin.data.values == [nmx_data_array.data.values[idx] for idx in selected_idx]
    )


@pytest.fixture
def reference_bin(nmx_data_array: sc.DataArray) -> ReferenceIntensities:
    return get_reference_intensities(
        nmx_data_array.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: 6})
    )


def test_reference_bin_scale_factor(reference_bin: ReferenceIntensities) -> None:
    """Test the scale factor for I."""
    scale_factor = estimate_scale_factor_per_hkl_asu_from_reference(reference_bin)
    expected_groups = [(7, 8, 9), (9, 8, 7)]

    assert len(scale_factor) == len(expected_groups)
    assert scale_factor.dim == "hkl_asu"
    for idx, group in enumerate(expected_groups):
        hkl = tuple(
            scale_factor.coords[coord][idx].value
            for coord in (f"{idx}_ASU" for idx in "HKL")
        )
        assert hkl == group
