# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.nmx.mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME
from ess.nmx.scaling import ReferenceWavelengthBin, _apply_elem_wise, get_reference_bin


def test_apply_elem_wise_add() -> None:
    var = sc.Variable(dims=["x"], values=[1, 2, 3])

    assert sc.identical(
        _apply_elem_wise(lambda x: x + 1, var),
        sc.Variable(dims=["x"], values=var.values + 1),
    )


def test_apply_elem_wise_str() -> None:
    var = sc.Variable(dims=["x"], values=[1, 2, 3])

    assert sc.identical(
        _apply_elem_wise(str, var),
        sc.Variable(dims=["x"], values=["1", "2", "3"]),
    )


def test_apply_elem_wise_vectors() -> None:
    var = sc.vectors(dims=["x"], values=[(1, 2, 3), (4, 5, 6), (7, 8, 9)])

    assert sc.identical(
        _apply_elem_wise(sum, var),
        sc.array(dims=["x"], values=[6, 15, 24], dtype=float),
    )


def test_detour_group_str() -> None:
    from ess.nmx.scaling import _group

    da = sc.DataArray(
        data=sc.ones(dims=["x"], shape=[3]),
        coords={"x": sc.Variable(dims=["x"], values=["a", "b", "a"])},
    )

    grouped = _group(da, "x", x=lambda x: x)
    assert sc.identical(
        grouped.coords["x"],
        sc.Variable(dims=["x"], values=["a", "b"]),
    )


def test_detour_group_vector() -> None:
    from ess.nmx.scaling import _group

    da = sc.DataArray(
        data=sc.ones(dims=["x"], shape=[10]),
        coords={"x": sc.vectors(dims=["x"], values=[(1, 2, 3), (4, 5, 6)] * 5)},
    )

    grouped = _group(da, "x", x=str)
    assert sc.identical(
        grouped.coords["x"],
        sc.vectors(dims=["x"], values=[(1, 2, 3), (4, 5, 6)]),
    )


@pytest.fixture
def nmx_data_array() -> sc.DataArray:
    return sc.DataArray(
        data=sc.ones(dims=["row"], shape=[7]),
        coords={
            DEFAULT_WAVELENGTH_COLUMN_NAME: sc.Variable(
                dims=["row"], values=[1, 2, 3, 4, 5, 3, 3]
            ),
            "hkl_eq": sc.vectors(
                dims=["row"],
                values=[
                    (1, 2, 3),
                    (4, 5, 6),
                    (7, 8, 9),
                    (10, 11, 12),
                    (13, 14, 15),
                    (7, 8, 9),
                    (9, 8, 7),
                ],
            ),
            "I": sc.Variable(dims=["row"], values=[1, 2, 3, 4, 5, 3.1, 3.2]),
            "SIGI": sc.Variable(
                dims=["row"], values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.31, 0.32]
            ),
        },
    )


def test_get_reference_bin_middle(nmx_data_array: sc.DataArray) -> None:
    """Test the middle bin."""

    ref_bin = get_reference_bin(nmx_data_array.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: 6}))
    selected_idx = (2, 5, 6)
    for coord in ("I", "SIGI"):
        assert all(
            ref_bin.coords[coord].values
            == [nmx_data_array.coords[coord].values[idx] for idx in selected_idx]
        )


@pytest.fixture
def reference_bin(nmx_data_array: sc.DataArray) -> ReferenceWavelengthBin:
    return get_reference_bin(nmx_data_array.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: 6}))


def test_reference_bin_scale_factor(reference_bin: ReferenceWavelengthBin) -> None:
    """Test the scale factor for I."""
    from ess.nmx.scaling import _group, calculate_scale_factor_per_hkl_eq

    scale_factor_groups = calculate_scale_factor_per_hkl_eq(reference_bin)
    grouped = _group(reference_bin, "hkl_eq", hkl_eq=str)

    for hkl_eq in grouped.coords["hkl_eq"].values:
        calculated_gr = scale_factor_groups["hkl_eq", sc.vector(hkl_eq)]
        reference_gr = grouped["hkl_eq", sc.vector(hkl_eq)]
        for coord in ("I", "SIGI"):
            assert sc.identical(
                calculated_gr.coords[f"scale_factor_{coord}"],
                sc.mean(1 / reference_gr.values.coords[coord]),
            )
