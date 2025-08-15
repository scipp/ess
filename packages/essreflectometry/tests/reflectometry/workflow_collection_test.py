# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pandas as pd
import sciline as sl

from ess.reflectometry.tools import WorkflowCollection


def int_to_float(x: int) -> float:
    return 0.5 * x


def int_float_to_str(x: int, y: float) -> str:
    return f"{x};{y}"


def make_param_table(params: dict) -> pd.DataFrame:
    all_types = {t for v in params.values() for t in v.keys()}
    data = {t: [] for t in all_types}
    for param in params.values():
        for t in all_types:
            data[t].append(param[t])
    return pd.DataFrame(data, index=params.keys()).rename_axis(index='run_id')


def test_compute() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])

    coll = WorkflowCollection(wf, make_param_table({'a': {int: 3}, 'b': {int: 4}}))

    assert dict(coll.compute(float)) == {'a': 1.5, 'b': 2.0}
    assert dict(coll.compute(str)) == {'a': '3;1.5', 'b': '4;2.0'}


def test_compute_multiple() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])

    coll = WorkflowCollection(wf, make_param_table({'a': {int: 3}, 'b': {int: 4}}))

    # wfa = wf.copy()
    # wfa[int] = 3
    # wfb = wf.copy()
    # wfb[int] = 4
    # coll = WorkflowCollection({'a': wfa, 'b': wfb})

    result = coll.compute([float, str])

    assert result['a'] == {float: 1.5, str: '3;1.5'}
    assert result['b'] == {float: 2.0, str: '4;2.0'}


def test_setitem_mapping() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = WorkflowCollection({'a': wfa, 'b': wfb})

    coll[int] = {'a': 7, 'b': 8}

    assert coll.compute(float) == {'a': 3.5, 'b': 4.0}
    assert coll.compute(str) == {'a': '7;3.5', 'b': '8;4.0'}


def test_setitem_single_value() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = WorkflowCollection({'a': wfa, 'b': wfb})

    coll[int] = 5

    assert coll.compute(float) == {'a': 2.5, 'b': 2.5}
    assert coll.compute(str) == {'a': '5;2.5', 'b': '5;2.5'}


def test_copy() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = WorkflowCollection({'a': wfa, 'b': wfb})

    coll_copy = coll.copy()

    assert coll_copy.compute(float) == {'a': 1.5, 'b': 2.0}
    assert coll_copy.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}

    coll_copy[int] = {'a': 7, 'b': 8}
    assert coll.compute(float) == {'a': 1.5, 'b': 2.0}
    assert coll.compute(str) == {'a': '3;1.5', 'b': '4;2.0'}
    assert coll_copy.compute(float) == {'a': 3.5, 'b': 4.0}
    assert coll_copy.compute(str) == {'a': '7;3.5', 'b': '8;4.0'}


def test_add_workflow() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = WorkflowCollection({'a': wfa, 'b': wfb})

    wfc = wf.copy()
    wfc[int] = 5
    coll.add('c', wfc)

    assert coll.compute(float) == {'a': 1.5, 'b': 2.0, 'c': 2.5}
    assert coll.compute(str) == {'a': '3;1.5', 'b': '4;2.0', 'c': '5;2.5'}


def test_add_workflow_with_existing_key() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = WorkflowCollection({'a': wfa, 'b': wfb})

    wfc = wf.copy()
    wfc[int] = 5
    coll.add('a', wfc)

    assert coll.compute(float) == {'a': 2.5, 'b': 2.0}
    assert coll.compute(str) == {'a': '5;2.5', 'b': '4;2.0'}
    assert 'c' not in coll.keys()  # 'c' should not exist


def test_remove_workflow() -> None:
    wf = sl.Pipeline([int_to_float, int_float_to_str])
    wfa = wf.copy()
    wfa[int] = 3
    wfb = wf.copy()
    wfb[int] = 4
    coll = WorkflowCollection({'a': wfa, 'b': wfb})

    coll.remove('b')

    assert 'b' not in coll.keys()
    assert coll.compute(float) == {'a': 1.5}
    assert coll.compute(str) == {'a': '3;1.5'}
