# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import sciline as sl

from ess.reflectometry.tools import BatchProcessor


def int_to_float(x: int) -> float:
    return 0.5 * x


def int_float_to_str(x: int, y: float) -> str:
    return f"{x};{y}"


def test_compute() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = BatchProcessor({'a': wfa, 'b': wfb})

    assert coll.compute(float) == {'a': 1.5, 'b': 2.0}
    assert coll.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}


def test_compute_multiple() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = BatchProcessor({'a': wfa, 'b': wfb})

    result = coll.compute([float, str])

    assert result[float] == {'a': 1.5, 'b': 2.0}
    assert result[str] == {'a': '3;1.5', 'b': '4;2.0'}


def test_setitem() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = BatchProcessor({'a': wfa, 'b': wfb})

    coll[int] = {'a': 7, 'b': 8}

    assert coll.compute(float) == {'a': 3.5, 'b': 4.0}
    assert coll.compute(str) == {'a': '7;3.5', 'b': '8;4.0'}


def test_copy() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = BatchProcessor({'a': wfa, 'b': wfb})

    coll_copy = coll.copy()

    assert coll_copy.compute(float) == {'a': 1.5, 'b': 2.0}
    assert coll_copy.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}

    coll_copy[int] = {'a': 7, 'b': 8}
    assert coll.compute(float) == {'a': 1.5, 'b': 2.0}
    assert coll.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}
    assert coll_copy.compute(float) == {'a': 3.5, 'b': 4.0}
    assert coll_copy.compute(str) == {'a': '7;3.5', 'b': '8;4.0'}
