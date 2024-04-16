# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.nmx.mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME
from ess.nmx.scaling import (
    _apply_elem_wise,
    _hash_repr,
    get_reference_bin,
    hash_variable,
)


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


def test_hash_variable_unique() -> None:
    """Different vector values should have different hashes."""
    from itertools import product

    import numpy as np

    var = sc.vectors(dims=["x"], values=list(product(range(20), repeat=3)))
    hash_var = hash_variable(var, hash_func=_hash_repr)
    assert len(hash_var.values) == len(np.unique(hash_var.values))


def test_hash_variable_same() -> None:
    """Same values should have the same hash."""
    var = sc.vectors(dims=["x"], values=[(1, 2, 3), (1, 2, 3)])
    hash_var = hash_variable(var, hash_func=_hash_repr)
    assert hash_var.values[0] == hash_var.values[1]


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
                    (8, 7, 9),
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
def reference_bin(nmx_data_array: sc.DataArray) -> sc.DataArray:
    return get_reference_bin(nmx_data_array.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: 6}))
