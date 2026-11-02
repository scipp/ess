# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

from ess.reduce.nexus import time_weighted_median


def make_log(times, values, *, unit='deg'):
    return sc.DataArray(
        sc.array(dims=['time'], values=values, unit=unit),
        coords={'time': sc.array(dims=['time'], values=times, unit='s')},
    )


def test_median_weights_values_by_duration():
    log = make_log([0, 1, 9], [100.0, 10.0, 20.0])
    result = time_weighted_median(
        log, start=sc.scalar(0, unit='s'), stop=sc.scalar(10, unit='s')
    )
    assert_allclose(result, sc.scalar(10.0, unit='deg'))


def test_median_sorts_values_instead_of_selecting_the_middle_time():
    log = make_log([0, 4, 7], [-20.0, 100.0, -30.0])
    result = time_weighted_median(
        log, start=sc.scalar(0, unit='s'), stop=sc.scalar(10, unit='s')
    )
    assert_allclose(result, sc.scalar(-20.0, unit='deg'))


@pytest.mark.parametrize(
    ('times', 'start', 'stop'),
    [([0, 5, 7, 15], 0, 10), ([0, 7, 9, 15], 2, 12)],
)
def test_median_exact_half_uses_time_weighted_mean_of_central_values(
    times, start, stop
):
    log = make_log(times, [10.0, 1.0, 4.0, float('nan')])
    result = time_weighted_median(
        log, start=sc.scalar(start, unit='s'), stop=sc.scalar(stop, unit='s')
    )
    # The central values 4 and 10 are held for 3 and 5 seconds in the interval.
    assert_allclose(result, sc.scalar((4.0 * 3 + 10.0 * 5) / 8, unit='deg'))


@pytest.mark.parametrize(
    ('times', 'values'),
    [
        ([0, 5], [0.0, 10.0]),
        ([0, 4, 5], [0.0, 0.0, 10.0]),
        ([0, 1, 5], [0.0, 0.0, 10.0]),
        ([0, 5, 6], [0.0, 10.0, 10.0]),
        ([0, 5, 9], [0.0, 10.0, 10.0]),
    ],
)
def test_median_is_unchanged_by_redundant_readings(times, values):
    result = time_weighted_median(
        make_log(times, values),
        start=sc.scalar(0, unit='s'),
        stop=sc.scalar(10, unit='s'),
    )
    assert_allclose(result, sc.scalar(5.0, unit='deg'))


def test_median_combines_equal_values_from_separate_intervals():
    log = make_log([0, 1, 3, 5, 7], [4.0, 10.0, 1.0, 4.0, 10.0])
    result = time_weighted_median(
        log, start=sc.scalar(0, unit='s'), stop=sc.scalar(10, unit='s')
    )
    # Values 1, 4, and 10 have total durations of 2, 3, and 5 seconds.
    assert_allclose(result, sc.scalar((4.0 * 3 + 10.0 * 5) / 8, unit='deg'))


@pytest.mark.parametrize('all_nan', [False, True])
def test_median_scalar_does_not_retain_log_storage(all_nan):
    log = make_log(np.arange(101), np.arange(101, dtype=float))
    if all_nan:
        log.values[:] = float('nan')
    result = time_weighted_median(
        log, start=sc.scalar(0, unit='s'), stop=sc.scalar(101, unit='s')
    )
    assert result.underlying_size() == result.copy().underlying_size()


@pytest.mark.parametrize('nan_index', [0, 2])
def test_median_propagates_nan_with_positive_duration(nan_index):
    values = [10.0, 20.0, 100.0]
    values[nan_index] = float('nan')
    log = make_log([0, 1, 9], values)
    result = time_weighted_median(
        log, start=sc.scalar(0, unit='s'), stop=sc.scalar(10, unit='s')
    )
    assert sc.isnan(result).value


def test_median_clips_interval_and_keeps_value_valid_at_start():
    log = make_log([0, 2, 10, 20], [1.0, 4.0, 8.0, float('nan')])
    result = time_weighted_median(
        log, start=sc.scalar(1, unit='s'), stop=sc.scalar(12, unit='s')
    )
    assert_allclose(result, sc.scalar(4.0, unit='deg'))


def test_median_accepts_datetime_timestamps():
    origin = sc.datetime('2026-10-28T12:35:10', unit='s')
    log = make_log([0, 1, 9], [10.0, 20.0, 100.0])
    log.coords['time'] = origin + log.coords['time']
    result = time_weighted_median(
        log, start=origin, stop=origin + sc.scalar(10, unit='s')
    )
    assert_allclose(result, sc.scalar(20.0, unit='deg'))


@pytest.mark.parametrize(
    ('start', 'stop', 'unit'), [(0.1, 2.1, 's'), (100, 2100, 'ms')]
)
def test_median_accepts_fractional_bounds_for_integer_timestamps(start, stop, unit):
    log = make_log([0, 1, 2], [1.0, 2.0, 3.0])
    result = time_weighted_median(
        log,
        start=sc.scalar(start, unit=unit),
        stop=sc.scalar(stop, unit=unit),
    )
    # Values 1, 2, and 3 are held for 0.9, 1.0, and 0.1 seconds.
    assert_allclose(result, sc.scalar(2.0, unit='deg'))


def test_single_value_is_held_for_entire_interval():
    log = make_log([0], [2.0], unit='mm')
    result = time_weighted_median(
        log, start=sc.scalar(5, unit='s'), stop=sc.scalar(10, unit='s')
    )
    assert_allclose(result, sc.scalar(2.0, unit='mm'))


def test_duplicate_timestamps_use_last_value():
    log = make_log([0, 0, 2], [float('nan'), 4.0, 8.0])
    result = time_weighted_median(
        log, start=sc.scalar(0, unit='s'), stop=sc.scalar(4, unit='s')
    )
    assert_allclose(result, sc.scalar(6.0, unit='deg'))


@pytest.mark.parametrize(
    ('times', 'start', 'stop', 'message'),
    [
        ([0, 2], -1, 10, 'No logged value'),
        ([0, 2], 1, 1, 'positive duration'),
        ([0, 2], 2, 1, 'positive duration'),
        ([2, 0], 2, 10, 'sorted'),
    ],
)
def test_median_rejects_invalid_time_information(times, start, stop, message):
    with pytest.raises(ValueError, match=message):
        time_weighted_median(
            make_log(times, [1.0, 2.0]),
            start=sc.scalar(start, unit='s'),
            stop=sc.scalar(stop, unit='s'),
        )


@pytest.mark.parametrize(('stop', 'expected'), [(None, 100.0), (10, 20.0)])
def test_median_uses_current_time_only_when_stop_is_missing(stop, expected):
    origin = sc.datetime('2026-10-21T00:00:00', unit='s')
    log = make_log([0, 1, 9], [10.0, 20.0, 100.0])
    log.coords['time'] = origin + log.coords['time']
    stop = None if stop is None else origin + sc.scalar(stop, unit='s')
    with patch('ess.reduce.nexus.filtering.datetime') as clock:
        clock.now.return_value = datetime(2026, 10, 21, 0, 0, 20, tzinfo=UTC)
        result = time_weighted_median(log, start=origin, stop=stop)
    assert_allclose(result, sc.scalar(expected, unit='deg'))


def test_median_requires_start():
    with pytest.raises(ValueError, match='start time is required'):
        time_weighted_median(make_log([0, 1], [1.0, 2.0]), start=None)


def test_relative_timestamps_require_explicit_stop():
    with pytest.raises(
        ValueError, match='end time is required for relative timestamps'
    ):
        time_weighted_median(make_log([0, 1], [1.0, 2.0]), start=sc.scalar(0, unit='s'))
