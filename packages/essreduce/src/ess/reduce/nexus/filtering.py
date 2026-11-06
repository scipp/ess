# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Filters for time-dependent NeXus transformation logs."""

from datetime import UTC, datetime

import numpy as np
import scipp as sc


def time_weighted_median(
    log: sc.DataArray, *, start: sc.Variable, stop: sc.Variable | None = None
) -> sc.Variable:
    """Return a log's median, weighting each value by how long it was held.

    Use this filter to obtain one representative log value for a reduction while
    limiting the effect of brief excursions.

    Each reading applies until the next reading, and the last reading applies until
    ``stop``. Only time within ``[start, stop)`` counts. Durations for repeated values
    are added together.

    Order the values from smallest to largest and add their durations until reaching
    half the interval's duration. Return the value reached. If the accumulated time
    is exactly half, return the time-weighted mean of that value and the next value.

    If readings share a timestamp, the last reading takes precedence. A NaN held
    for any time inside the interval makes the result NaN.

    Parameters
    ----------
    log:
        Nonempty, one-dimensional log with dimension and point coordinate ``time``,
        ordered by time. Times can be datetimes or times relative to an origin.
    start, stop:
        Scalar bounds using the same time origin as the log. The end must be later
        than the start, and the log must have a reading at or before the start.
        If ``stop`` is omitted or ``None``, use the current UTC time. Relative
        timestamps require an explicit end time.

    Returns
    -------
    :
        Scalar median with the unit of the logged values.

    Examples
    --------
    A log value of 2 held for 90 of a measurement's 100 seconds gives a result of
    2, even if most log entries describe changes during the other 10 seconds.
    Apply this filter to sample transformation logs in an existing workflow:

      >>> workflow[TransformationTimeFilter[snx.NXsample, RunType]] = (
      ...     time_weighted_median
      ... )

    The workflow sets ``start`` to the maximum of the start times in
    :class:`~ess.reduce.nexus.types.TimeInterval` and
    :class:`~ess.reduce.nexus.types.Measurement`, and ``stop`` to the minimum of their
    end times. Missing times are ignored.
    """
    if log.dims != ('time',):
        raise ValueError("Expected a one-dimensional log with dimension 'time'.")
    time = log.coords['time']
    if time.sizes != log.sizes:
        raise ValueError("Expected one timestamp per log value.")
    if start is None:
        raise ValueError('The interval start time is required.')
    if stop is None:
        if time.dtype != sc.DType.datetime64:
            raise ValueError('An end time is required for relative timestamps.')
        stop = sc.scalar(np.datetime64(datetime.now(UTC).replace(tzinfo=None), 'ns'))
    if start.ndim != 0 or stop.ndim != 0:
        raise ValueError('Interval bounds must be scalars.')
    time, start, stop = (
        sc.array(dims=t.dims, values=t.values.astype('float64'), unit=t.unit)
        for t in (time, start, stop)
    )
    start = start.to(unit=time.unit)
    stop = stop.to(unit=time.unit)
    if not start < stop:
        raise ValueError('The interval must have positive duration.')
    if not sc.issorted(time, 'time'):
        raise ValueError('Log timestamps must be sorted in ascending order.')
    if start < time[0]:
        raise ValueError('No logged value is available at the interval start time.')

    ends = sc.concat([time[1:], stop], 'time')
    durations = sc.where(ends > stop, stop, ends) - sc.where(time < start, start, time)
    # Ignore entries outside the interval and superseded duplicate timestamps.
    active = durations > sc.scalar(0, unit=durations.unit)
    weighted = sc.DataArray(log.data, coords={'duration': durations})[active]
    nan = sc.isnan(weighted.data)
    if sc.any(nan):
        return weighted.data[int(np.argmax(nan.values))].copy()
    weighted = sc.sort(weighted, key=weighted.data)
    values = weighted.data.values
    # Add durations for equal values.
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    durations = weighted.coords['duration'].values
    weights = np.add.reduceat(durations, starts)
    cumulative = np.cumsum(weights)
    half = cumulative[-1] / 2
    index = int(np.searchsorted(cumulative, half))
    value = weighted.data[int(starts[index])]
    if cumulative[index] == half:
        next_value = weighted.data[int(starts[index + 1])]
        left, right = weights[index : index + 2]
        return (value * left + next_value * right) / (left + right)
    return value.copy()
