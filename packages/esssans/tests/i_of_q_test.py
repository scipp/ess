# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess import sans

merge_func = sans.workflow.merge_contributions


def test_merge_banks_or_runs_returns_first_value_if_length_1() -> None:
    data = sc.DataArray(sc.scalar(1.0))
    assert merge_func(data) is data


def test_merge_banks_or_runs_sums_dense_data() -> None:
    data = [
        sc.DataArray(data=sc.array(dims=['Q'], values=[1.0, 2.0])),
        sc.DataArray(data=sc.array(dims=['Q'], values=[3.0, 4.0])),
    ]
    assert sc.identical(
        merge_func(*data),
        sc.DataArray(
            data=sc.array(dims=['Q'], values=[4.0, 6.0]),
        ),
    )


def test_merge_banks_or_runs_concats_bins() -> None:
    events1 = sc.DataArray(data=sc.array(dims=['event'], values=[1.0, 2.0, 3.0]))
    events2 = sc.DataArray(data=sc.array(dims=['event'], values=[4.0, 5.0]))
    data = [
        sc.bins(
            begin=sc.array(dims=['Q'], values=[0, 1], unit=None),
            dim='event',
            data=events1,
        ),
        sc.bins(
            begin=sc.array(dims=['Q'], values=[0, 2], unit=None),
            dim='event',
            data=events2,
        ),
    ]
    expected_events = sc.DataArray(
        data=sc.array(dims=['event'], values=[1.0, 4.0, 5.0, 2.0, 3.0]),
    )
    expected = sc.bins(
        begin=sc.array(dims=['Q'], values=[0, 3], unit=None),
        dim='event',
        data=expected_events,
    )
    assert sc.identical(merge_func(*data), expected)
