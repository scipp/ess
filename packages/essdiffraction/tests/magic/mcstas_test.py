# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc
from ess.magic import MagicMcStasWorkflow
from ess.magic.mcstas import load_mcstas_events
from ess.magic.types import DetectorBank, Filename, RawMcStasEvents, SampleRun
from scipp.testing import assert_identical

from ess.reduce.unwrap import PulsePeriod

_PULSE_PERIOD = sc.scalar(1 / 14, unit='s')

_COLUMNS = ('p', 'x', 'y', 'z', 'vx', 'vy', 'vz', 't')
_UNITS = ('counts', 'm', 'm', 'm', 'm/s', 'm/s', 'm/s', 's')
_EVENTS = np.array(
    [
        [2.0, -100.0, 2.0, 3.0, 4.0, -5.0, 6.0, 0.2],
        [0.0, 7.0, 8.0, 9.0, -10.0, 11.0, 12.0, 0.01],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.03],
    ]
)


def _write_mcstas(tmp_path, *, columns=_COLUMNS, events=_EVENTS):
    filename = tmp_path / 'mccode.h5'
    with h5py.File(filename, 'w') as file:
        # Conflicting data here must not replace the selected component's events.
        file.create_dataset('entry1/data/unrelated/events', data=np.ones((1, 8)))
        simulation = file.create_group('entry1/simulation')
        simulation.attrs['program'] = np.bytes_('McStas 3.5.32')
        simulation.create_group('Param')
        for index, bank in enumerate(('A', 'B'), start=1):
            output = file.create_group(
                f'entry1/instrument/components/{index:04d}_{bank}_abs_logger/'
                'output/events_list'
            )
            output.attrs['variables'] = np.bytes_(' '.join(columns))
            output.create_dataset('events', data=events * index)
    return filename


@pytest.mark.parametrize('bank', list(DetectorBank))
def test_loads_only_selected_bank_and_preserves_raw_events(tmp_path, bank):
    filename = _write_mcstas(tmp_path)
    other = '0002_B' if bank is DetectorBank.detector_a else '0001_A'
    with h5py.File(filename, 'r+') as file:
        del file[f'entry1/instrument/components/{other}_abs_logger/output']

    events = load_mcstas_events(filename, bank, _PULSE_PERIOD)

    scale = 1 if bank is DetectorBank.detector_a else 2
    expected = _EVENTS * scale
    assert events.sizes == {'event': len(_EVENTS)}
    assert events.bins is None
    assert not events.masks
    assert_identical(
        events.data,
        sc.array(
            dims=['event'],
            values=expected[:, 0],
            variances=expected[:, 0] ** 2,
            unit='counts',
        ),
    )
    assert set(events.coords) == {*_COLUMNS[1:], 'event_time_offset', 'event_time_zero'}
    for index, (name, unit) in enumerate(zip(_COLUMNS, _UNITS, strict=True)):
        if name != 'p':
            assert_identical(
                events.coords[name],
                sc.array(dims=['event'], values=expected[:, index], unit=unit),
            )


def test_uses_column_names_instead_of_column_order(tmp_path):
    order = [7, 5, 1, 0, 6, 3, 2, 4]
    filename = _write_mcstas(
        tmp_path, columns=tuple(_COLUMNS[i] for i in order), events=_EVENTS[:, order]
    )
    actual = load_mcstas_events(filename, DetectorBank.detector_a, _PULSE_PERIOD)
    filename = _write_mcstas(tmp_path)
    expected = load_mcstas_events(filename, DetectorBank.detector_a, _PULSE_PERIOD)
    assert_identical(actual, expected)


@pytest.mark.parametrize('columns', [_COLUMNS[:-1], (*_COLUMNS[:-1], 'p')])
def test_rejects_missing_or_duplicate_columns(tmp_path, columns):
    filename = _write_mcstas(
        tmp_path, columns=columns, events=_EVENTS[:, : len(columns)]
    )
    with pytest.raises(ValueError, match='columns'):
        load_mcstas_events(filename, DetectorBank.detector_a, _PULSE_PERIOD)


def test_loads_empty_logger(tmp_path):
    filename = _write_mcstas(tmp_path, events=np.empty((0, 8)))
    events = load_mcstas_events(filename, DetectorBank.detector_a, _PULSE_PERIOD)
    assert events.sizes == {'event': 0}
    assert set(events.coords) == {*_COLUMNS[1:], 'event_time_offset', 'event_time_zero'}
    assert events.variances is not None


def test_loads_float32_events(tmp_path):
    values = _EVENTS.astype('float32')
    values[0, 0] = 1e-30
    filename = _write_mcstas(tmp_path, events=values)
    events = load_mcstas_events(filename, DetectorBank.detector_a, _PULSE_PERIOD)
    assert events.dtype == sc.DType.float64
    np.testing.assert_array_equal(events.values, values[:, 0])
    np.testing.assert_array_equal(events.variances, values[:, 0].astype('float64') ** 2)
    for index, column in enumerate(_COLUMNS[1:], start=1):
        np.testing.assert_array_equal(events.coords[column].values, values[:, index])


@pytest.mark.parametrize('bank', [None, DetectorBank.detector_b])
def test_workflow_computes_raw_sample_events(tmp_path, bank):
    filename = _write_mcstas(tmp_path)
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    if bank is not None:
        workflow[DetectorBank] = bank
    actual = workflow.compute(RawMcStasEvents[SampleRun])
    expected = load_mcstas_events(
        filename, bank or DetectorBank.detector_a, _PULSE_PERIOD
    )
    assert_identical(actual, expected)


@pytest.mark.parametrize('bank', list(DetectorBank))
def test_timing_preserves_arrival_times_at_frame_boundaries(tmp_path, bank):
    period = 1 / 14
    times = np.array(
        [
            -0.01,
            0.0,
            np.nextafter(period, 0),
            period,
            np.nextafter(period, np.inf),
            0.15,
            0.22,
            np.nextafter(7 * period, 0),
            7 * period,
            np.nextafter(7 * period, np.inf),
            np.nextafter(14 * period, 0),
            14 * period,
            np.nextafter(14 * period, np.inf),
        ]
    )
    values = np.zeros((len(times), 8))
    # The fixture scales bank B's rows by two.
    values[:, 7] = times / (1 if bank is DetectorBank.detector_a else 2)
    filename = _write_mcstas(tmp_path, events=values)
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[DetectorBank] = bank
    events = workflow.compute(RawMcStasEvents[SampleRun])

    np.testing.assert_array_equal(events.coords['t'].values, times)
    offset = events.coords['event_time_offset']
    zero = events.coords['event_time_zero']
    np.testing.assert_allclose(
        offset.values, np.remainder(times, period), atol=1e-16, rtol=0
    )
    assert offset.unit == 's'
    assert np.all((offset.values >= 0) & (offset.values < period))
    assert zero.dtype == sc.DType.datetime64
    assert zero.unit == 'ns'
    # Round pulse timestamps to nearest ns, rather than truncating.
    np.testing.assert_array_equal(
        zero.values.astype('int64'),
        [
            -71428571,
            0,
            0,
            71428571,
            71428571,
            142857143,
            214285714,
            428571429,
            500000000,
            500000000,
            928571429,
            1000000000,
            1000000000,
        ],
    )
    np.testing.assert_allclose(
        zero.values.astype('int64') * 1e-9 + offset.values,
        times,
        atol=0.5e-9,
        rtol=0,
    )


def test_workflow_accepts_integer_millisecond_pulse_period(tmp_path):
    times = [0.025, 0.05, 0.1, 0.125]
    values = np.zeros((len(times), 8))
    values[:, 7] = times
    filename = _write_mcstas(tmp_path, events=values)
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[PulsePeriod] = sc.scalar(50, unit='ms')
    events = workflow.compute(RawMcStasEvents[SampleRun])
    np.testing.assert_allclose(
        events.coords['event_time_offset'].values,
        [0.025, 0.0, 0.0, 0.025],
        atol=1e-17,
        rtol=0,
    )
    np.testing.assert_array_equal(
        events.coords['event_time_zero'].values.astype('int64'),
        [0, 50000000, 100000000, 100000000],
    )
    np.testing.assert_array_equal(events.coords['t'].values, times)


@pytest.mark.parametrize('time', [np.nan, np.inf, -np.inf])
def test_nonfinite_arrival_time_raises(tmp_path, time):
    values = _EVENTS.copy()
    values[0, 7] = time
    filename = _write_mcstas(tmp_path, events=values)
    with pytest.raises(ValueError, match='finite'):
        load_mcstas_events(filename, DetectorBank.detector_a, _PULSE_PERIOD)


@pytest.mark.parametrize(
    'period',
    [sc.scalar(value, unit='s') for value in (0.0, -0.1, np.nan, np.inf)]
    + [sc.array(dims=['period'], values=[0.05, 0.1], unit='s')],
)
def test_invalid_pulse_period_raises(tmp_path, period):
    filename = _write_mcstas(tmp_path)
    with pytest.raises(ValueError, match=r'(?i)period'):
        load_mcstas_events(filename, DetectorBank.detector_a, period)
