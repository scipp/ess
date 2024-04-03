# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess import sans


def test_no_bank_merge_returns_input() -> None:
    data = sc.DataArray(sc.scalar(1.0))
    assert sans.no_bank_merge(data) is data


def test_no_run_merge_returns_input() -> None:
    data = sc.DataArray(sc.scalar(1.0))
    assert sans.no_run_merge(data) is data


@pytest.mark.parametrize('merge_func', [sans.merge_banks, sans.merge_runs])
def test_merge_banks_or_runs_returns_first_value_if_length_1(merge_func) -> None:
    data = sc.DataArray(sc.scalar(1.0))
    assert merge_func({'bank1': data}) is data


@pytest.mark.parametrize('merge_func', [sans.merge_banks, sans.merge_runs])
def test_merge_banks_or_runs_sums_dense_data(merge_func) -> None:
    data = {
        'bank_or_run1': sc.DataArray(data=sc.array(dims=['Q'], values=[1.0, 2.0])),
        'bank_or_run2': sc.DataArray(data=sc.array(dims=['Q'], values=[3.0, 4.0])),
    }
    assert sc.identical(
        merge_func(data),
        sc.DataArray(
            data=sc.array(dims=['Q'], values=[4.0, 6.0]),
        ),
    )


@pytest.mark.parametrize('merge_func', [sans.merge_banks, sans.merge_runs])
def test_merge_banks_or_runs_concats_bins(merge_func) -> None:
    events1 = sc.DataArray(data=sc.array(dims=['event'], values=[1.0, 2.0, 3.0]))
    events2 = sc.DataArray(data=sc.array(dims=['event'], values=[4.0, 5.0]))
    data = {
        'bank_or_run1': sc.bins(
            begin=sc.array(dims=['Q'], values=[0, 1], unit=None),
            dim='event',
            data=events1,
        ),
        'bank_or_run2': sc.bins(
            begin=sc.array(dims=['Q'], values=[0, 2], unit=None),
            dim='event',
            data=events2,
        ),
    }
    expected_events = sc.DataArray(
        data=sc.array(dims=['event'], values=[1.0, 4.0, 5.0, 2.0, 3.0]),
    )
    expected = sc.bins(
        begin=sc.array(dims=['Q'], values=[0, 3], unit=None),
        dim='event',
        data=expected_events,
    )
    assert sc.identical(merge_func(data), expected)
