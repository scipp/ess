# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from ess.nmx.scaling import (
    ReferenceIntensities,
    estimate_scale_factor_per_hkl_asu_from_reference,
    get_reference_intensities,
    get_reference_wavelength,
)


@pytest.fixture()
def nmx_data_array() -> sc.DataArray:
    da = sc.DataArray(
        data=sc.array(dims=["row"], values=[1, 2, 3, 4, 5, 3.1, 3.2]),
        coords={
            "wavelength": sc.Variable(dims=["row"], values=[1, 2, 3, 4, 5, 3, 3]),
            "hkl_asu": sc.array(
                dims=["row"],
                values=[
                    "[1, 2, 3]",
                    "[4, 5, 6]",
                    "[7, 8, 9]",
                    "[10, 11, 12]",
                    "[13, 14, 15]",
                    "[7, 8, 9]",
                    "[9, 8, 7]",
                ],
            ),
        },
    )
    da.variances = (
        sc.array(dims=["row"], values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.31, 0.32]) ** 2
    )
    return da


def test_get_reference_bin_middle(nmx_data_array: sc.DataArray) -> None:
    """Test the middle bin."""

    binned = nmx_data_array.bin({"wavelength": 6})
    reference_wavelength = get_reference_wavelength(binned, reference_wavelength=None)

    ref_bin = get_reference_intensities(
        nmx_data_array.bin({"wavelength": 6}),
        reference_wavelength,
    )
    selected_idx = (2, 5, 6)
    assert all(
        ref_bin.data.values == [nmx_data_array.data.values[idx] for idx in selected_idx]
    )


@pytest.fixture()
def reference_bin(nmx_data_array: sc.DataArray) -> ReferenceIntensities:
    binned = nmx_data_array.bin({"wavelength": 6})
    reference_wavelength = get_reference_wavelength(binned, reference_wavelength=None)

    return get_reference_intensities(
        binned,
        reference_wavelength,
    )


def test_reference_bin_scale_factor(reference_bin: ReferenceIntensities) -> None:
    """Test the scale factor for I."""
    scale_factor = estimate_scale_factor_per_hkl_asu_from_reference(reference_bin)
    expected_groups = [[7, 8, 9], [9, 8, 7]]

    assert len(scale_factor) == len(expected_groups)
    assert scale_factor.dim == "hkl_asu"
    for idx, group in enumerate(expected_groups):
        assert scale_factor.coords['hkl_asu'][idx].value == str(group)
