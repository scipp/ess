# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pandas as pd
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
    batch = BatchProcessor({'a': wfa, 'b': wfb})

    assert batch.compute(float) == {'a': 1.5, 'b': 2.0}
    assert batch.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}


def test_compute_multiple() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    batch = BatchProcessor({'a': wfa, 'b': wfb})

    result = batch.compute([float, str])

    assert result[float] == {'a': 1.5, 'b': 2.0}
    assert result[str] == {'a': '3;1.5', 'b': '4;2.0'}


def test_compute_mapped() -> None:
    ints = [1, 2, 3]
    df = pd.DataFrame({int: ints})
    wf = sl.Pipeline([int_to_float, int_float_to_str]).map(df)
    batch = BatchProcessor({"": wf})
    res_float = batch.compute(float)
    assert res_float[""] == [0.5, 1.0, 1.5]
    res_str = batch.compute(str)
    assert res_str[""] == ["1;0.5", "2;1.0", "3;1.5"]


def test_compute_mixed_mapped_unmapped() -> None:
    ints = [1, 2, 3]
    df = pd.DataFrame({int: ints})
    unmapped = sl.Pipeline([int_to_float, int_float_to_str])
    mapped = unmapped.map(df)
    unmapped[int] = 5
    batch = BatchProcessor({"unmapped": unmapped, "mapped": mapped})
    res_float = batch.compute(float)
    assert res_float['unmapped'] == 2.5
    assert res_float['mapped'] == [0.5, 1.0, 1.5]
    res_str = batch.compute(str)
    assert res_str['unmapped'] == '5;2.5'
    assert res_str['mapped'] == ['1;0.5', '2;1.0', '3;1.5']


def test_setitem() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    batch = BatchProcessor({'a': wfa, 'b': wfb})

    batch[int] = {'a': 7, 'b': 8}

    assert batch.compute(float) == {'a': 3.5, 'b': 4.0}
    assert batch.compute(str) == {'a': '7;3.5', 'b': '8;4.0'}


def test_copy() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    batch = BatchProcessor({'a': wfa, 'b': wfb})

    batch_copy = batch.copy()

    assert batch_copy.compute(float) == {'a': 1.5, 'b': 2.0}
    assert batch_copy.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}

    batch_copy[int] = {'a': 7, 'b': 8}
    assert batch.compute(float) == {'a': 1.5, 'b': 2.0}
    assert batch.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}
    assert batch_copy.compute(float) == {'a': 3.5, 'b': 4.0}
    assert batch_copy.compute(str) == {'a': '7;3.5', 'b': '8;4.0'}
