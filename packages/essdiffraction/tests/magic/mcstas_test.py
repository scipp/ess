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

    events = load_mcstas_events(filename, bank)

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
    assert set(events.coords) == set(_COLUMNS[1:])
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
    actual = load_mcstas_events(filename, DetectorBank.detector_a)
    filename = _write_mcstas(tmp_path)
    expected = load_mcstas_events(filename, DetectorBank.detector_a)
    assert_identical(actual, expected)


@pytest.mark.parametrize('columns', [_COLUMNS[:-1], (*_COLUMNS[:-1], 'p')])
def test_rejects_missing_or_duplicate_columns(tmp_path, columns):
    filename = _write_mcstas(
        tmp_path, columns=columns, events=_EVENTS[:, : len(columns)]
    )
    with pytest.raises(ValueError, match='columns'):
        load_mcstas_events(filename, DetectorBank.detector_a)


def test_loads_empty_logger(tmp_path):
    filename = _write_mcstas(tmp_path, events=np.empty((0, 8)))
    events = load_mcstas_events(filename, DetectorBank.detector_a)
    assert events.sizes == {'event': 0}
    assert set(events.coords) == set(_COLUMNS[1:])
    assert events.variances is not None


def test_loads_float32_events(tmp_path):
    values = _EVENTS.astype('float32')
    values[0, 0] = 1e-30
    filename = _write_mcstas(tmp_path, events=values)
    events = load_mcstas_events(filename, DetectorBank.detector_a)
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
    expected = load_mcstas_events(filename, bank or DetectorBank.detector_a)
    assert_identical(actual, expected)
