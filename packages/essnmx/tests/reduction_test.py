# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.nmx.reduction import _apply_elem_wise, _group


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
    da = sc.DataArray(
        data=sc.ones(dims=["x"], shape=[10]),
        coords={"x": sc.vectors(dims=["x"], values=[(1, 2, 3), (4, 5, 6)] * 5)},
    )

    grouped = _group(da, "x", x=str)
    assert sc.identical(
        grouped.coords["x"],
        sc.vectors(dims=["x"], values=[(1, 2, 3), (4, 5, 6)]),
    )
