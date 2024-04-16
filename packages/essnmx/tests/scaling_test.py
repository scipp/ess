# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.nmx.scaling import _apply_elem_wise


def test_apply_elem_wise_add() -> None:
    var = sc.Variable(dims=["x"], values=[1, 2, 3])

    assert sc.identical(
        _apply_elem_wise(lambda x: x + 1, var),
        sc.Variable(dims=["x"], values=var.values + 1),
    )


def test_apply_elem_wise_str() -> None:
    from ess.nmx.scaling import _apply_elem_wise

    var = sc.Variable(dims=["x"], values=[1, 2, 3])

    assert sc.identical(
        _apply_elem_wise(str, var),
        sc.Variable(dims=["x"], values=["1", "2", "3"]),
    )


def test_apply_elem_wise_vectors() -> None:
    from ess.nmx.scaling import _apply_elem_wise

    var = sc.vectors(dims=["x"], values=[(1, 2, 3), (4, 5, 6), (7, 8, 9)])

    assert sc.identical(
        _apply_elem_wise(sum, var),
        sc.array(dims=["x"], values=[6, 15, 24], dtype=float),
    )
