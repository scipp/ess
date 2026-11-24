# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest
import scipp as sc
from ess.magic import MagicMcStasWorkflow, mcstas
from ess.magic.types import DetectorBank, Filename, RawDetector, SampleRun
from scipp.testing import assert_identical

from ess.reduce.unwrap import PulsePeriod

_PULSE_PERIOD = sc.scalar(1 / 14, unit='s')

# voxelization_of_mcstas_events_for_detector_a/b in magic-notebooks,
# mantid_worfkflow/dependencies/voxelization.py at
# da5844563df1a51442171c8178241879b52a06b0. A tilt -9.101 deg; B tilt +10 deg.
_REFERENCE = {
    DetectorBank.detector_a: (
        [
            [0, -0.1, 1.2],
            [0.01, 0.1, 1.1],
            [0.7, -0.4, 1],
            [1, -0.5, 0.8],
            [-0.005, 0, 1.2],
        ],
        [189724, 94684, 208991, 271423, 188404],
    ),
    DetectorBank.detector_b: (
        [
            [0, 0, 1.2],
            [-0.01, 0.02, 1.1],
            [-0.7, -0.04, 1],
            [-1, 0, 0.8],
            [0.005, 0, 1.2],
        ],
        [542725, 517381, 548176, 563315, 542725],
    ),
}


def _write_events(tmp_path, bank, xyz, *, weights=None, parameters=None, times=None):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    events = np.zeros((len(xyz), 8))
    events[:, 0] = 1.0 if weights is None else weights
    events[:, 1:4] = xyz
    events[:, 7] = np.arange(len(xyz))[::-1] * 0.03 if times is None else times
    values = {
        'A_r1': 1.0,
        'A_section_length': 0.53,
        'A_casette_gamma': 1.05,
        'A_casette_omega': -9.101,
        'B_casette_omega': 10.0,
    } | (parameters or {})
    filename = tmp_path / 'simulation.h5'
    with h5py.File(filename, 'w') as file:
        file.create_group('entry1/data')
        simulation = file.create_group('entry1/simulation')
        simulation.attrs['program'] = np.bytes_('McStas 3.5.32')
        params = simulation.create_group('Param')
        for name, value in values.items():
            params[name] = [np.bytes_(str(value))]
        for index, selected in enumerate(DetectorBank, start=1):
            label = 'A' if selected is DetectorBank.detector_a else 'B'
            component = file.create_group(
                f'entry1/instrument/components/{index:04d}_{label}_abs_logger'
            )
            component['Position'] = [0.25, -0.5, 154.0]
            component['Rotation'] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
            if selected is bank:
                output = component.create_group('output/events_list')
                output.attrs['variables'] = np.bytes_('p x y z vx vy vz t')
                output['events'] = events
    return filename, events


@pytest.mark.parametrize('bank', list(DetectorBank))
def test_raw_detector_matches_reference_and_preserves_events_and_geometry(
    tmp_path, bank
):
    xyz, ids = _REFERENCE[bank]
    filename, events = _write_events(
        tmp_path, bank, [*xyz, xyz[0], [0, 0, 0]], weights=[2, 0, 3, 4, 5, 7, 11]
    )
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    geometry.masks['disabled'] = geometry.coords['pixel_id'] == sc.scalar(
        ids[0], unit=None
    )
    original = geometry.copy(deep=True)

    actual = mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)

    assert actual.sizes == geometry.sizes
    assert_identical(geometry, original)
    for name, coord in geometry.coords.items():
        assert_identical(actual.coords[name], coord)
    assert_identical(actual.masks['disabled'], geometry.masks['disabled'])
    assert actual.bins.size().values.sum() == 6
    expected_ids = np.array([*ids, ids[0]])
    for pixel_id in np.unique(expected_ids):
        index = pixel_id - geometry.coords['pixel_id'].values[0]
        pixel = actual['pixel_id', int(index)].value
        rows = events[:-1][expected_ids == pixel_id]
        np.testing.assert_array_equal(pixel.values, rows[:, 0])
        np.testing.assert_array_equal(pixel.variances, rows[:, 0] ** 2)
        np.testing.assert_array_equal(pixel.coords['t'].values, rows[:, 7])
        assert pixel.unit == 'counts'
        assert pixel.coords['t'].unit == 's'
        assert set(pixel.coords) <= {
            'pixel_id',
            't',
            'event_time_offset',
            'event_time_zero',
        }


@pytest.mark.parametrize('bank', list(DetectorBank))
@pytest.mark.parametrize(
    'xyz', [[], [[0, 0, 0], [0, 100, 1.2], [np.nan, 0, 1.2], [0, np.inf, 1.2]]]
)
def test_empty_and_rejected_events_retain_empty_pixels(tmp_path, bank, xyz):
    filename, _ = _write_events(tmp_path, bank, xyz)
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    actual = mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)
    assert actual.sizes == geometry.sizes
    assert actual.bins.size().values.sum() == 0
    assert_identical(actual.coords['position'], geometry.coords['position'])


@pytest.mark.parametrize(
    ('bank', 'top', 'bottom', 'expected'),
    [
        (DetectorBank.detector_a, 0.252863, -0.722903, [3962, 2]),
        (DetectorBank.detector_b, 0.051016, -0.051016, [493570, 491522]),
    ],
)
def test_acceptance_includes_front_and_top_excludes_back_and_bottom(
    tmp_path, bank, top, bottom, expected
):
    filename, _ = _write_events(
        tmp_path,
        bank,
        [[0, 0, 1], [0, 0, 1.53], [0, top, 1], [0, bottom, 1]],
        parameters={'A_casette_omega': 0.0, 'B_casette_omega': 0.0},
    )
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    actual = mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)
    populated = actual.bins.size().values != 0
    np.testing.assert_array_equal(
        actual.coords['pixel_id'].values[populated], sorted(expected)
    )
    assert actual.bins.size().values.sum() == 2


def test_geometry_missing_accepted_pixel_ids_raises(tmp_path):
    bank = DetectorBank.detector_a
    filename, _ = _write_events(tmp_path, bank, _REFERENCE[bank][0])
    wrong = mcstas.mcstas_detector_geometry(filename, DetectorBank.detector_b)
    with pytest.raises(ValueError, match=r'pixel|geometry|detector'):
        mcstas.load_mcstas_events(filename, bank, wrong, _PULSE_PERIOD)


def test_assignment_chunk_size_does_not_change_output(tmp_path, monkeypatch):
    bank = DetectorBank.detector_a
    filename, _ = _write_events(tmp_path, bank, _REFERENCE[bank][0])
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    expected = mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)
    monkeypatch.setattr(mcstas, '_ASSIGNMENT_CHUNK_SIZE', 2)
    assert_identical(
        mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD), expected
    )


def test_assignment_uses_recorded_a_dimensions_and_spacing(tmp_path):
    # First/last cathode centres and first anode. 0.51 degrees selects counter 2
    # with the recorded spacing, but counter 1 with the nominal 1.05 degrees.
    depth = np.array([0.01, 0.63])
    fraction = depth / 0.64
    top = 0.252863 + (0.416545 - 0.252863) * fraction
    bottom = -0.722903 + (-1.077999 + 0.722903) * fraction
    angle = np.deg2rad(0.51)
    xyz = np.column_stack(
        (
            (2 + depth) * np.sin(angle),
            top + (bottom - top) * (0.5 / 128),
            (2 + depth) * np.cos(angle),
        )
    )
    bank = DetectorBank.detector_a
    filename, _ = _write_events(
        tmp_path,
        bank,
        xyz,
        parameters={
            'A_r1': 2.0,
            'A_section_length': 0.64,
            'A_casette_gamma': 1.0,
            'A_casette_omega': 0.0,
        },
    )
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    actual = mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)
    np.testing.assert_array_equal(
        actual.coords['pixel_id'].values[actual.bins.size().values != 0], [3, 476163]
    )


def test_assignment_rejects_angular_footprint_crossing_atan2_branch(tmp_path):
    bank = DetectorBank.detector_a
    filename, _ = _write_events(
        tmp_path, bank, [[0, 0, 1.2]], parameters={'A_casette_gamma': 4.0}
    )
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    with pytest.raises(ValueError, match='angular footprint'):
        mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)


@pytest.mark.parametrize('bank', [None, DetectorBank.detector_b])
def test_workflow_computes_shared_raw_detector(tmp_path, bank):
    selected = bank or DetectorBank.detector_a
    filename, _ = _write_events(tmp_path, selected, _REFERENCE[selected][0])
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    if bank is not None:
        workflow[DetectorBank] = bank
    actual = workflow.compute(RawDetector[SampleRun])
    geometry = mcstas.mcstas_detector_geometry(filename, selected)
    assert_identical(
        actual, mcstas.load_mcstas_events(filename, selected, geometry, _PULSE_PERIOD)
    )
    assert actual.bins.size().values.sum() == 5


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
    filename, _ = _write_events(
        tmp_path, bank, [_REFERENCE[bank][0][0]] * len(times), times=times
    )
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[DetectorBank] = bank
    events = workflow.compute(RawDetector[SampleRun]).bins.constituents['data']

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
    # MAGIC's converter rounds pulse timestamps to nearest ns, rather than truncating.
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
    bank = DetectorBank.detector_a
    times = [0.025, 0.05, 0.1, 0.125]
    filename, _ = _write_events(
        tmp_path, bank, [_REFERENCE[bank][0][0]] * len(times), times=times
    )
    workflow = MagicMcStasWorkflow()
    workflow[Filename[SampleRun]] = filename
    workflow[PulsePeriod] = sc.scalar(50, unit='ms')
    events = workflow.compute(RawDetector[SampleRun]).bins.constituents['data']
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
def test_nonfinite_accepted_time_raises(tmp_path, time):
    bank = DetectorBank.detector_a
    filename, _ = _write_events(tmp_path, bank, [_REFERENCE[bank][0][0]], times=[time])
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    with pytest.raises(ValueError, match='finite'):
        mcstas.load_mcstas_events(filename, bank, geometry, _PULSE_PERIOD)


def test_nonfinite_rejected_time_does_not_contaminate_timing(tmp_path):
    bank = DetectorBank.detector_a
    filename, _ = _write_events(
        tmp_path, bank, [_REFERENCE[bank][0][0], [0, 0, 0]], times=[0.08, np.nan]
    )
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    events = mcstas.load_mcstas_events(
        filename, bank, geometry, _PULSE_PERIOD
    ).bins.constituents['data']
    np.testing.assert_array_equal(events.coords['t'].values, [0.08])
    assert np.isfinite(events.coords['event_time_offset'].values).all()
    assert not np.isnat(events.coords['event_time_zero'].values).any()


@pytest.mark.parametrize(
    'period',
    [sc.scalar(value, unit='s') for value in (0.0, -0.1, np.nan, np.inf)]
    + [sc.array(dims=['period'], values=[0.05, 0.1], unit='s')],
)
def test_invalid_pulse_period_raises(tmp_path, period):
    bank = DetectorBank.detector_a
    filename, _ = _write_events(tmp_path, bank, [_REFERENCE[bank][0][0]])
    geometry = mcstas.mcstas_detector_geometry(filename, bank)
    with pytest.raises(ValueError, match=r'(?i)period'):
        mcstas.load_mcstas_events(filename, bank, geometry, period)
