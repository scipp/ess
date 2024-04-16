# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.nmx.scaling import _apply_elem_wise, hash_variable


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
    hash_var = hash_variable(var)
    assert len(hash_var.values) == len(np.unique(hash_var.values))


def test_hash_variable_same() -> None:
    """Same values should have the same hash."""
    var = sc.vectors(dims=["x"], values=[(1, 2, 3), (1, 2, 3)])
    hash_var = hash_variable(var)
    assert hash_var.values[0] == hash_var.values[1]
