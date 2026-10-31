# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.merge import MergeRules, Rule, plan_merge

DETECTOR = '/entry/instrument/detector'
EVENTS = f'{DETECTOR}/data'
TEMPERATURE = '/entry/sample/temperature'
TARGET = '/entry/instrument/motor/target_value'
SECOND = 10**9
TARGET_TIME = SECOND // 2
"""When the motor got its target, before all runs."""


def _group(parent: h5py.Group, name: str, nx_class: str) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs['NX_class'] = nx_class
    return group


def _stream(group: h5py.Group, name: str, data: list, dtype: str) -> h5py.Dataset:
    """
    A dataset like those the file writer appends to while a run is going on.

    Compressed, unlike in files of the file writer, to test that the merge keeps
    the storage layout and filters.
    """
    numeric = np.dtype(dtype).kind in 'iuf'
    return group.create_dataset(
        name,
        data=np.asarray(data, dtype=dtype),
        maxshape=(None,),
        chunks=(4,),
        compression='gzip',
        compression_opts=7,
        shuffle=True,
        fletcher32=True,
        fillvalue=-1 if numeric else None,
    )


def _log(
    parent: h5py.Group, name: str, times: list[int], values: list[float], unit: str
) -> h5py.Group:
    """An NXlog as the file writer writes it, with an alarm series of one row."""
    log = _group(parent, name, 'NXlog')
    _stream(log, 'time', times, 'uint64').attrs['units'] = 'ns'
    _stream(log, 'value', values, 'float64').attrs['units'] = unit
    _stream(log, 'alarm_time', times[:1], 'uint64').attrs['units'] = 'ns'
    _stream(log, 'alarm_severity', [0], 'int16')
    _stream(log, 'alarm_message', [b''], 'S200')
    _stream(log, 'cue_timestamp_zero', times[-1:], 'uint64')
    _stream(log, 'cue_index', [len(times) - 1], 'int64')
    log['minimum_value'] = np.min(values)
    log['maximum_value'] = np.max(values)
    log['average_value'] = np.mean(values)
    return log


def make_run(
    path: Path,
    *,
    index: int,
    pulses: list[list[int]],
    temperature: list[float],
    target: float = 1.5,
    pixels: int = 4,
    title: str = 'water',
    temperature_unit: str = 'K',
) -> Path:
    """
    A NeXus file of run ``index``, with one event per pixel id in ``pulses``.

    The run starts at second ``t0 = 10 * index + 1`` and ends 8 s later. Its
    pulses and temperature readings are 1 s apart from ``t0``. The motor log holds
    the target from before the run, as in every run.
    """
    t0 = 10 * index + 1
    with h5py.File(path, 'w') as f:
        f.attrs['file_name'] = str(path)
        f.attrs['file_time'] = f'2026-10-26T01:00:{t0:02}Z'
        entry = _group(f, 'entry', 'NXentry')
        entry['start_time'] = f'2026-10-26T01:00:{t0:02}.000Z'
        entry['end_time'] = f'2026-10-26T01:00:{t0 + 8:02}.000Z'
        entry['entry_identifier'] = str(100 + index)
        entry['title'] = title
        instrument = _group(entry, 'instrument', 'NXinstrument')
        detector = _group(instrument, 'detector', 'NXdetector')
        detector['detector_number'] = np.arange(pixels, dtype='int32')
        data = _group(detector, 'data', 'NXevent_data')
        ids = [i for pulse in pulses for i in pulse]
        _stream(data, 'event_id', ids, 'int32')
        _stream(data, 'event_time_offset', [1000] * len(ids), 'int32').attrs[
            'units'
        ] = 'ns'
        pulse_times = [(t0 + i) * SECOND for i in range(len(pulses))]
        _stream(data, 'event_time_zero', pulse_times, 'uint64').attrs['units'] = 'ns'
        _stream(data, 'event_index', np.cumsum([0, *map(len, pulses)])[:-1], 'int64')
        _stream(data, 'cue_index', [len(ids)], 'int64')
        motor = _group(instrument, 'motor', 'NXpositioner')
        _log(motor, 'target_value', [TARGET_TIME], [target], 'mm')
        transformations = _group(detector, 'transformations', 'NXtransformations')
        transformations['x'] = motor['target_value']['value']  # hard link
        transformations['y'] = h5py.SoftLink(f'{TARGET}/value')
        sample = _group(entry, 'sample', 'NXsample')
        times = [(t0 + i) * SECOND for i in range(len(temperature))]
        log = _log(sample, 'temperature', times, temperature, temperature_unit)
        sample['temperature_value'] = log['value']  # hard link
    return path


def modify(path: Path, change: Callable[[h5py.File], object]) -> Path:
    with h5py.File(path, 'r+') as f:
        change(f)
    return path


RULES = MergeRules(
    rules=(
        Rule(paths=('/entry/instrument/*/target_value',), error=sc.scalar(0.0)),
        Rule(
            paths=(TEMPERATURE,),
            warn=sc.scalar(0.1, unit='K'),
            error=sc.scalar(500.0, unit='mK'),
        ),
    )
)


@pytest.fixture
def runs(tmp_path: Path) -> list[Path]:
    # Out of order: the merge sorts the runs by their start time.
    return [
        make_run(tmp_path / 'b.nxs', index=1, pulses=[[0, 1, 1]], temperature=[10.05]),
        make_run(
            tmp_path / 'a.nxs',
            index=0,
            pulses=[[3], [], [2, 2]],
            temperature=[10.0, 10.02],
        ),
    ]


@pytest.fixture
def merged(runs: list[Path], tmp_path: Path) -> Path:
    return plan_merge(runs, RULES).write(tmp_path / 'merged.nxs')


def _error_paths(runs: list[Path], rules: MergeRules = RULES) -> set[str]:
    return {issue.path for issue in plan_merge(runs, rules).errors}


def test_plan_of_compatible_runs_has_no_issues(runs: list[Path]) -> None:
    plan = plan_merge(runs, RULES)
    assert plan.issues == ()
    assert [p.name for p in plan.files] == ['a.nxs', 'b.nxs']
    assert all(p.is_absolute() for p in plan.files)


def test_write_concatenates_events_in_order_of_start_time(merged: Path) -> None:
    with h5py.File(merged) as f:
        assert list(f[f'{EVENTS}/event_id']) == [3, 2, 2, 0, 1, 1]
        assert list(f[f'{EVENTS}/event_index']) == [0, 1, 1, 3]
        assert list(f[f'{EVENTS}/cue_index']) == [3, 6]
    with snx.File(merged) as f:
        events = f[EVENTS][()]
    assert sc.identical(
        events.bins.size().data,
        sc.array(dims=['event_time_zero'], values=[1, 0, 2, 3], unit=None),
    )


def test_write_concatenates_logs_and_updates_their_summaries(merged: Path) -> None:
    with h5py.File(merged) as f:
        log = f[TEMPERATURE]
        assert list(log['value']) == [10.0, 10.02, 10.05]
        assert list(log['time']) == [1 * SECOND, 2 * SECOND, 11 * SECOND]
        assert list(log['alarm_time']) == [1 * SECOND, 11 * SECOND]
        assert list(log['cue_index']) == [1, 2]
        assert log['minimum_value'][()] == 10.0
        assert log['maximum_value'][()] == 10.05
        assert log['average_value'][()] == pytest.approx(10.023333)


def test_write_drops_rows_that_repeat_the_previous_run(merged: Path) -> None:
    with h5py.File(merged) as f:
        log = f[TARGET]
        assert list(log['time']) == [TARGET_TIME]
        assert list(log['value']) == [1.5]
        assert list(log['alarm_time']) == [TARGET_TIME]
        assert list(log['alarm_message']) == [b'']
        # The cue of the second run points to the row identical to its dropped row
        assert list(log['cue_index']) == [0, 0]
    # A static transformation, as the workflows require of a depends_on chain
    with snx.File(merged) as f:
        assert f[TARGET][()].sizes == {'time': 1}


def test_write_drops_several_rows_that_repeat_the_previous_run(
    tmp_path: Path,
) -> None:
    runs = [
        make_run(
            tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0, 10.1, 10.2]
        ),
        # Starts with the readings at 2 s and 3 s, from before it started at 11 s
        make_run(tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[0, 0, 10.3]),
    ]

    def repeat_end_of_run_a(f: h5py.File) -> None:
        f[f'{TEMPERATURE}/time'][:2] = [2 * SECOND, 3 * SECOND]
        f[f'{TEMPERATURE}/value'][:2] = [10.1, 10.2]

    modify(runs[1], repeat_end_of_run_a)
    merged = plan_merge(runs, RULES).write(tmp_path / 'merged.nxs')
    with h5py.File(merged) as f:
        assert list(f[f'{TEMPERATURE}/value']) == [10.0, 10.1, 10.2, 10.3]
        # Last row of each run
        assert list(f[f'{TEMPERATURE}/cue_index']) == [2, 3]


def test_write_appends_hard_linked_dataset_once(merged: Path) -> None:
    with h5py.File(merged) as f:
        assert f['/entry/sample/temperature_value'] == f[f'{TEMPERATURE}/value']
        assert list(f[f'{TEMPERATURE}/value']) == [10.0, 10.02, 10.05]
        assert f[f'{DETECTOR}/transformations/x'] == f[f'{TARGET}/value']
        link = f[f'{DETECTOR}/transformations'].get('y', getlink=True)
        assert isinstance(link, h5py.SoftLink)


def test_write_keeps_attributes_and_storage_layout(
    runs: list[Path], merged: Path
) -> None:
    with h5py.File(runs[0]) as run, h5py.File(merged) as f:
        assert f[TEMPERATURE].attrs['NX_class'] == 'NXlog'
        assert f[f'{TEMPERATURE}/value'].attrs['units'] == 'K'
        for path in (f'{EVENTS}/event_id', f'{TEMPERATURE}/value'):
            assert f[path].compression == 'gzip'
            assert f[path].maxshape == (None,)
            # Chunks, filters, and fill value
            assert f[path].id.get_create_plist() == run[path].id.get_create_plist()


def test_write_records_the_runs_and_the_merged_file(merged: Path) -> None:
    with h5py.File(merged) as f:
        assert f.attrs['file_name'] == str(merged)
        assert f.attrs['file_time'] != '2026-10-26T01:00:01Z'
        entry = f['entry']
        assert entry['start_time'].asstr()[()] == '2026-10-26T01:00:01.000Z'
        assert entry['end_time'].asstr()[()] == '2026-10-26T01:00:19.000Z'
        runs = entry['merged_runs']
        assert runs.attrs['NX_class'] == 'NXcollection'
        assert runs['run_0/file_name'].asstr()[()] == 'a.nxs'
        assert runs['run_1/file_name'].asstr()[()] == 'b.nxs'
        assert runs['run_0/entry_identifier'].asstr()[()] == '100'
        assert runs['run_1/end_time'].asstr()[()] == '2026-10-26T01:00:19.000Z'


def test_write_merges_three_runs(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / f'{i}.nxs', index=i, pulses=[[0], [1, 2]], temperature=[1])
        for i in range(3)
    ]
    merged = plan_merge(runs, MergeRules()).write(tmp_path / 'merged.nxs')
    with h5py.File(merged) as f:
        assert list(f[f'{EVENTS}/event_id']) == [0, 1, 2] * 3
        assert list(f[f'{EVENTS}/event_index']) == [0, 1, 3, 4, 6, 7]
        assert list(f[f'{TEMPERATURE}/cue_index']) == [0, 1, 2]


def test_write_leaves_no_file_if_writing_fails(
    runs: list[Path], tmp_path: Path
) -> None:
    plan = plan_merge(runs, RULES)
    plan.files[1].chmod(0)  # keeps the size and modification time
    try:
        with pytest.raises(OSError, match='nable to'):
            plan.write(tmp_path / 'merged.nxs')
    finally:
        plan.files[1].chmod(0o644)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.nxs', 'b.nxs']


def test_write_refuses_plan_with_errors(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[11.0]),
    ]
    with pytest.raises(ValueError, match='errors'):
        plan_merge(runs, RULES).write(tmp_path / 'merged.nxs')
    assert not (tmp_path / 'merged.nxs').exists()


def test_write_refuses_if_a_file_changed(runs: list[Path], tmp_path: Path) -> None:
    plan = plan_merge(runs, RULES)
    make_run(runs[0], index=1, pulses=[[0, 1, 1, 2]], temperature=[10.05])
    with pytest.raises(RuntimeError, match='changed'):
        plan.write(tmp_path / 'merged.nxs')


def test_write_refuses_to_overwrite(runs: list[Path], tmp_path: Path) -> None:
    (tmp_path / 'merged.nxs').touch()
    with pytest.raises(FileExistsError):
        plan_merge(runs, RULES).write(tmp_path / 'merged.nxs')


def test_merged_file_cannot_be_merged_again(merged: Path, tmp_path: Path) -> None:
    later = make_run(tmp_path / 'c.nxs', index=2, pulses=[[0]], temperature=[10.0])
    (issue,) = plan_merge([merged, later], RULES).issues
    assert (issue.severity, issue.path) == ('error', 'merged.nxs')


def test_overlapping_runs_are_an_error(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(tmp_path / 'b.nxs', index=0, pulses=[[0]], temperature=[10.0]),
    ]
    errors = plan_merge(runs, RULES).errors
    assert any('starts before the end' in error.message for error in errors)


def _delete(path: str) -> Callable[[h5py.File], object]:
    return lambda f: f.__delitem__(path)


def _replace_dtype(f: h5py.File) -> None:
    detector = f[DETECTOR]
    del detector['detector_number']
    detector['detector_number'] = np.arange(4, dtype='int64')


def _replace_hard_link(f: h5py.File) -> None:
    transformations = f[f'{DETECTOR}/transformations']
    del transformations['x']
    transformations.create_dataset('x', data=f[f'{TARGET}/value'][()], maxshape=(None,))


def _retarget_soft_link(f: h5py.File) -> None:
    transformations = f[f'{DETECTOR}/transformations']
    del transformations['y']
    transformations['y'] = h5py.SoftLink(f'{TEMPERATURE}/value')


@pytest.mark.parametrize(
    ('change', 'path'),
    [
        (_delete(f'{DETECTOR}/detector_number'), f'{DETECTOR}/detector_number'),
        (_replace_dtype, f'{DETECTOR}/detector_number'),
        (lambda f: f[TEMPERATURE].attrs.create('x', 1), TEMPERATURE),
        (lambda f: f['/'].attrs.create('x', 1), '/'),
        (_replace_hard_link, f'{TARGET}/value'),
        (_retarget_soft_link, f'{DETECTOR}/transformations/y'),
    ],
    ids=['missing', 'dtype', 'attribute', 'root attribute', 'hard link', 'soft link'],
)
def test_layout_difference_is_an_error(
    runs: list[Path], change: Callable[[h5py.File], object], path: str
) -> None:
    modify(runs[0], change)
    assert path in _error_paths(runs)


def test_differing_fixed_size_dataset_is_an_error_unless_ignored(
    tmp_path: Path,
) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(
            tmp_path / 'b.nxs',
            index=1,
            pulses=[[0]],
            temperature=[10.0],
            title='heavy water',
        ),
    ]
    (error,) = plan_merge(runs, RULES).errors
    assert error.path == '/entry/title'
    rules = MergeRules(rules=RULES.rules, ignore=('/entry/title',))
    assert plan_merge(runs, rules).issues == ()


def test_differing_shape_of_fixed_size_dataset_is_an_error(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(
            tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[10.0], pixels=5
        ),
    ]
    (error,) = plan_merge(runs, RULES).errors
    assert error.path == f'{DETECTOR}/detector_number'


def test_ignore_covers_the_contents_of_a_group(runs: list[Path]) -> None:
    modify(runs[0], lambda f: _group(f['entry'], 'user_ada', 'NXuser'))
    modify(runs[1], lambda f: _group(f['entry'], 'user_bob', 'NXuser'))
    modify(runs[1], lambda f: f['entry/user_bob'].create_dataset('name', data='Bob'))
    assert _error_paths(runs) == {
        '/entry/user_ada',
        '/entry/user_bob',
        '/entry/user_bob/name',
    }
    rules = MergeRules(rules=RULES.rules, ignore=('/entry/user_*',))
    assert plan_merge(runs, rules).issues == ()


def test_per_run_field_may_differ_in_shape(runs: list[Path]) -> None:
    modify(runs[0], lambda f: f['entry'].create_dataset('scripts', data=['a']))
    modify(runs[1], lambda f: f['entry'].create_dataset('scripts', data=['a', 'b']))
    assert plan_merge(runs, RULES).issues == ()


def test_padding_rows_with_time_zero_are_an_error(runs: list[Path]) -> None:
    def pad(f: h5py.File) -> None:
        time = f[f'{TEMPERATURE}/time']
        time.resize(time.shape[0] + 1, axis=0)  # the fill value is not 0
        time[-1] = 0

    modify(runs[0], pad)
    (error,) = plan_merge(runs, RULES).errors
    assert error.path == TEMPERATURE
    assert 'time 0' in error.message


def test_log_row_not_later_than_previous_run_is_an_error(runs: list[Path]) -> None:
    # Run b (runs[0]) has one reading, at 11 s. Move it before the end of run a.
    modify(runs[0], lambda f: f[f'{TEMPERATURE}/time'].__setitem__(0, SECOND))
    (error,) = plan_merge(runs, RULES).errors
    assert error.path == TEMPERATURE
    assert 'overlaps' in error.message


def test_unsorted_log_is_an_error(runs: list[Path]) -> None:
    # Run a (runs[1]) has readings at 1 s and 2 s.
    modify(runs[1], lambda f: f[f'{TEMPERATURE}/time'].__setitem__(0, 3 * SECOND))
    (error,) = plan_merge(runs, RULES).errors
    assert error.path == TEMPERATURE
    assert 'not sorted' in error.message


def test_unsorted_pulses_within_a_run_are_fine(runs: list[Path]) -> None:
    # Run a (runs[1]) has pulses at 1 s, 2 s, and 3 s.
    modify(runs[1], lambda f: f[f'{EVENTS}/event_time_zero'].__setitem__(0, 3 * SECOND))
    assert plan_merge(runs, RULES).issues == ()


def test_overlapping_pulses_are_an_error(runs: list[Path]) -> None:
    # Run b (runs[0]) has a pulse at 11 s. Move it before the last pulse of run a.
    modify(runs[0], lambda f: f[f'{EVENTS}/event_time_zero'].__setitem__(0, 3 * SECOND))
    (error,) = plan_merge(runs, RULES).errors
    assert error.path == EVENTS
    assert 'overlap' in error.message


def test_unknown_field_of_log_is_an_error(runs: list[Path]) -> None:
    for run in runs:
        modify(run, lambda f: _stream(f[TEMPERATURE], 'flux', [1], 'int64'))
    assert _error_paths(runs) == {f'{TEMPERATURE}/flux'}


@pytest.mark.parametrize(('size', 'refused'), [(1, True), (0, False)])
def test_index_field_of_unknown_group_is_an_error_unless_empty(
    runs: list[Path], size: int, refused: bool
) -> None:
    path = '/entry/instrument/monitor_events/event_index'
    for run in runs:
        modify(
            run,
            lambda f: _stream(
                _group(f['entry/instrument'], 'monitor_events', 'NXdata'),
                'event_index',
                [0] * size,
                'int64',
            ),
        )
    assert _error_paths(runs) == ({path} if refused else set())


def test_spread_beyond_error_tolerance_is_an_error(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[10.0]),
        make_run(tmp_path / 'c.nxs', index=2, pulses=[[0]], temperature=[10.6]),
    ]
    (issue,) = plan_merge(runs, RULES).issues
    assert (issue.severity, issue.path) == ('error', TEMPERATURE)


def test_spread_within_a_single_run_counts(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0, 10.2]),
        make_run(tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[10.1]),
    ]
    (issue,) = plan_merge(runs, RULES).issues
    assert (issue.severity, issue.path) == ('warning', TEMPERATURE)


def test_zero_tolerance_requires_identical_values(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(
            tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[10.0], target=1.6
        ),
    ]
    (issue,) = plan_merge(runs, RULES).issues
    assert (issue.severity, issue.path) == ('error', TARGET)


def test_nan_in_checked_log_is_an_error(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(tmp_path / 'b.nxs', index=1, pulses=[[0]], temperature=[np.nan]),
    ]
    (issue,) = plan_merge(runs, RULES).issues
    assert (issue.severity, issue.path) == ('error', TEMPERATURE)
    assert 'NaN' in issue.message


@pytest.mark.parametrize(
    'tolerance',
    [sc.scalar(1.0, unit='m'), sc.scalar(1.0)],
    ids=['other dimension', 'dimensionless'],
)
def test_tolerance_in_incompatible_unit_is_an_error(
    runs: list[Path], tolerance: sc.Variable
) -> None:
    rules = MergeRules(rules=(Rule(paths=(TEMPERATURE,), warn=tolerance),))
    (issue,) = plan_merge(runs, rules).issues
    assert (issue.severity, issue.path) == ('error', TEMPERATURE)
    assert 'cannot convert' in issue.message


def test_tolerance_is_converted_to_unit_of_log(tmp_path: Path) -> None:
    runs = [
        make_run(
            tmp_path / f'{i}.nxs',
            index=i,
            pulses=[[0]],
            temperature=[10_000.0 + 200 * i],
            temperature_unit='mK',
        )
        for i in range(2)
    ]
    (issue,) = plan_merge(runs, RULES).issues  # 200 mK, warn at 0.1 K
    assert (issue.severity, issue.path) == ('warning', TEMPERATURE)


def test_unit_change_between_runs_is_an_error(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(
            tmp_path / 'b.nxs',
            index=1,
            pulses=[[0]],
            temperature=[10_000.0],
            temperature_unit='mK',
        ),
    ]
    assert TEMPERATURE in _error_paths(runs)


def test_changing_log_without_rule_is_reported(runs: list[Path]) -> None:
    (issue,) = plan_merge(runs, MergeRules()).issues
    assert (issue.severity, issue.path) == ('info', TEMPERATURE)


def test_rule_that_matches_no_log_is_a_warning(runs: list[Path]) -> None:
    rules = MergeRules(rules=(*RULES.rules, Rule(paths=('/x',), error=sc.scalar(0.0))))
    (issue,) = plan_merge(runs, rules).issues
    assert (issue.severity, issue.path) == ('warning', '/x')


def test_rules_from_toml(tmp_path: Path) -> None:
    path = tmp_path / 'rules.toml'
    path.write_text(
        """
ignore = ["/entry/title"]

[[rule]]
paths = ["/entry/instrument/*/target_value"]
error = 0

[[rule]]
paths = ["/entry/sample/temperature"]
warn = "0.1K"
error = "5e-1 K"
"""
    )
    rules = MergeRules.from_toml(path)
    assert rules.ignore == ('/entry/title',)
    target, temperature = rules.rules
    assert rules.rule_for(TARGET) is target
    assert rules.rule_for(TEMPERATURE) is temperature
    assert sc.identical(target.error, sc.scalar(0.0))
    assert target.warn is None
    assert sc.identical(temperature.warn, sc.scalar(0.1, unit='K'))
    assert sc.identical(temperature.error, sc.scalar(0.5, unit='K'))


@pytest.mark.parametrize(
    ('content', 'match'),
    [
        ('[[rule]]\npaths = ["/a"]\neror = 0\n', 'eror'),
        ('[[rule]]\nerror = 0\n', 'rule 0 .* paths'),
        ('[[rule]]\npaths = "/a"\nerror = 0\n', 'rule 0 .* paths'),
        ('ignore = "/a"\n', 'ignore'),
        ('[[rule]]\npaths = ["/a"]\nerror = "K"\n', 'rule 0 .* tolerance'),
        ('[[rule]]\npaths = ["/a"]\nerror = "1 blorbs"\n', 'rule 0'),
        ('[[rule]]\npaths = ["/a"]\n', 'rule 0 .* neither'),
    ],
    ids=[
        'unknown key',
        'no paths',
        'paths not a list',
        'ignore not a list',
        'no number',
        'unknown unit',
        'no tolerance',
    ],
)
def test_rules_from_toml_refuses_invalid_rules(
    tmp_path: Path, content: str, match: str
) -> None:
    path = tmp_path / 'rules.toml'
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        MergeRules.from_toml(path)


def test_rule_refuses_tolerance_without_unit() -> None:
    with pytest.raises(ValueError, match='without unit'):
        Rule(paths=('/a',), error=sc.scalar(0.0, unit=None))
