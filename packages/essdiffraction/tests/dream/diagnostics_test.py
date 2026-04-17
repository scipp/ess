# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from ess.dream.diagnostics import unwrap_flat_indices_2d


def test_unwrap_flat_indices_2d_single_dim() -> None:
    unwrap = unwrap_flat_indices_2d({'a': 3}, {'b': 5})
    assert unwrap(0, 0) == {'a': 0, 'b': 0}
    assert unwrap(1, 0) == {'a': 1, 'b': 0}
    assert unwrap(2, 0) == {'a': 2, 'b': 0}
    assert unwrap(0, 4) == {'a': 0, 'b': 4}
    assert unwrap(2, 4) == {'a': 2, 'b': 4}


def test_unwrap_flat_indices_2d_two_x_dims() -> None:
    unwrap = unwrap_flat_indices_2d({'a': 3, '2nd': 5}, {'b': 4})
    assert unwrap(0, 0) == {'a': 0, '2nd': 0, 'b': 0}
    assert unwrap(0, 1) == {'a': 0, '2nd': 0, 'b': 1}
    assert unwrap(0, 3) == {'a': 0, '2nd': 0, 'b': 3}

    assert unwrap(1, 0) == {'a': 0, '2nd': 1, 'b': 0}
    assert unwrap(1, 1) == {'a': 0, '2nd': 1, 'b': 1}
    assert unwrap(1, 3) == {'a': 0, '2nd': 1, 'b': 3}

    assert unwrap(4, 0) == {'a': 0, '2nd': 4, 'b': 0}
    assert unwrap(4, 1) == {'a': 0, '2nd': 4, 'b': 1}
    assert unwrap(4, 3) == {'a': 0, '2nd': 4, 'b': 3}

    assert unwrap(5, 0) == {'a': 1, '2nd': 0, 'b': 0}
    assert unwrap(5, 2) == {'a': 1, '2nd': 0, 'b': 2}
    assert unwrap(5, 3) == {'a': 1, '2nd': 0, 'b': 3}

    assert unwrap(11, 0) == {'a': 2, '2nd': 1, 'b': 0}
    assert unwrap(11, 2) == {'a': 2, '2nd': 1, 'b': 2}
    assert unwrap(11, 3) == {'a': 2, '2nd': 1, 'b': 3}

    assert unwrap(14, 0) == {'a': 2, '2nd': 4, 'b': 0}
    assert unwrap(14, 1) == {'a': 2, '2nd': 4, 'b': 1}
    assert unwrap(14, 2) == {'a': 2, '2nd': 4, 'b': 2}


def test_unwrap_flat_indices_2d_three_y_dims() -> None:
    unwrap = unwrap_flat_indices_2d({'a': 2}, {'b': 4, 'c': 1, 'd': 3})
    assert unwrap(0, 0) == {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    assert unwrap(1, 0) == {'a': 1, 'b': 0, 'c': 0, 'd': 0}

    assert unwrap(0, 3) == {'a': 0, 'b': 1, 'c': 0, 'd': 0}
    assert unwrap(1, 3) == {'a': 1, 'b': 1, 'c': 0, 'd': 0}

    assert unwrap(0, 7) == {'a': 0, 'b': 2, 'c': 0, 'd': 1}
    assert unwrap(1, 7) == {'a': 1, 'b': 2, 'c': 0, 'd': 1}


def test_unwrap_flat_indices_2d_dim_order() -> None:
    unwrap = unwrap_flat_indices_2d({'x': 3, 'a': 5, 'g': 1}, {'b': 4, 'y': 4})
    assert list(unwrap(0, 0)) == ['x', 'a', 'g', 'b', 'y']
