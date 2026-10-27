# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
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
    return group.create_dataset(
        name,
        data=np.asarray(data, dtype=dtype),
        maxshape=(None,),
        chunks=(4,),
        compression='gzip',
        compression_opts=7,
        shuffle=True,
        fletcher32=True,
        fillvalue=-1,
    )


def _log(
    parent: h5py.Group, name: str, times: list[int], values: list[float], unit: str
) -> h5py.Group:
    log = _group(parent, name, 'NXlog')
    _stream(log, 'time', times, 'uint64').attrs['units'] = 'ns'
    _stream(log, 'value', values, 'float64').attrs['units'] = unit
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
) -> Path:
    """
    A NeXus file of run ``index``, with one event per pixel id in ``pulses``.

    The run starts at second ``10 * index``, its pulses are 1 s apart.
    """
    t0 = 10 * index
    with h5py.File(path, 'w') as f:
        f.attrs['file_name'] = str(path)
        f.attrs['file_time'] = f'2026-10-26T01:00:{t0:02}Z'
        entry = _group(f, 'entry', 'NXentry')
        entry['start_time'] = f'2026-10-26T01:00:{t0:02}.000Z'
        entry['end_time'] = f'2026-10-26T01:00:{t0 + 9:02}.000Z'
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
        pulse_times = [(t0 + i) * 10**9 for i in range(len(pulses))]
        _stream(data, 'event_time_zero', pulse_times, 'uint64').attrs['units'] = 'ns'
        _stream(data, 'event_index', np.cumsum([0, *map(len, pulses[:-1])]), 'int64')
        _stream(data, 'cue_index', [len(ids)], 'int64')
        motor = _group(instrument, 'motor', 'NXpositioner')
        _log(motor, 'target_value', [t0 * 10**9], [target], 'mm')
        transformations = _group(detector, 'transformations', 'NXtransformations')
        transformations['x'] = motor['target_value']['value']  # hard link
        transformations['y'] = h5py.SoftLink(f'{TARGET}/value')
        sample = _group(entry, 'sample', 'NXsample')
        times = [(t0 + i) * 10**9 for i in range(len(temperature))]
        _log(sample, 'temperature', times, temperature, 'K')
    return path


RULES = MergeRules(
    rules=(
        Rule(
            paths=('/entry/instrument/*/target_value',), error=sc.scalar(0.0, unit=None)
        ),
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


def test_plan_of_compatible_runs_has_no_issues(runs: list[Path]) -> None:
    plan = plan_merge(runs, RULES)
    assert plan.issues == ()
    assert [p.name for p in plan.files] == ['a.nxs', 'b.nxs']


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
        assert list(log['time']) == [0, 10**9, 10 * 10**9]
        assert list(log['cue_index']) == [1, 2]
        assert log['minimum_value'][()] == 10.0
        assert log['maximum_value'][()] == 10.05
        assert log['average_value'][()] == pytest.approx(10.023333)


def test_write_appends_hard_linked_dataset_once(merged: Path) -> None:
    with h5py.File(merged) as f:
        assert f[f'{DETECTOR}/transformations/x'] == f[f'{TARGET}/value']
        assert list(f[f'{TARGET}/value']) == [1.5, 1.5]
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


def test_write_takes_end_time_of_last_run_and_records_all_runs(
    merged: Path,
) -> None:
    with h5py.File(merged) as f:
        entry = f['entry']
        assert entry['start_time'].asstr()[()] == '2026-10-26T01:00:00.000Z'
        assert entry['end_time'].asstr()[()] == '2026-10-26T01:00:19.000Z'
        runs = entry['merged_runs']
        assert runs.attrs['NX_class'] == 'NXcollection'
        assert runs['run_0/file_name'].asstr()[()] == 'a.nxs'
        assert runs['run_1/file_name'].asstr()[()] == 'b.nxs'
        assert runs['run_0/entry_identifier'].asstr()[()] == '100'
        assert runs['run_1/end_time'].asstr()[()] == '2026-10-26T01:00:19.000Z'


def test_overlapping_runs_are_an_error(tmp_path: Path) -> None:
    runs = [
        make_run(tmp_path / 'a.nxs', index=0, pulses=[[0]], temperature=[10.0]),
        make_run(tmp_path / 'b.nxs', index=0, pulses=[[0]], temperature=[10.0]),
    ]
    (error,) = plan_merge(runs, RULES).errors
    assert 'starts before the end' in error.message


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


def test_tolerance_in_incompatible_unit_is_an_error(runs: list[Path]) -> None:
    rules = MergeRules(
        rules=(Rule(paths=(TEMPERATURE,), warn=sc.scalar(1.0, unit='m')),)
    )
    (issue,) = plan_merge(runs, rules).issues
    assert (issue.severity, issue.path) == ('error', TEMPERATURE)
    assert 'cannot convert' in issue.message


def test_changing_log_without_rule_is_reported(runs: list[Path]) -> None:
    (issue,) = plan_merge(runs, MergeRules()).issues
    assert (issue.severity, issue.path) == ('info', TEMPERATURE)


def test_rule_that_matches_no_log_is_a_warning(runs: list[Path]) -> None:
    rules = MergeRules(
        rules=(*RULES.rules, Rule(paths=('/x',), error=sc.scalar(0.0, unit=None)))
    )
    (issue,) = plan_merge(runs, rules).issues
    assert (issue.severity, issue.path) == ('warning', '/x')


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
warn = "0.1 K"
"""
    )
    rules = MergeRules.from_toml(path)
    assert rules.ignore == ('/entry/title',)
    target, temperature = rules.rules
    assert rules.rule_for(TARGET) is target
    assert rules.rule_for(TEMPERATURE) is temperature
    assert sc.identical(target.error, sc.scalar(0.0, unit=None))
    assert target.warn is None
    assert sc.identical(temperature.warn, sc.scalar(0.1, unit='K'))


def test_rules_from_toml_refuses_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / 'rules.toml'
    path.write_text('[[rule]]\npaths = ["/a"]\neror = 0\n')
    with pytest.raises(ValueError, match='eror'):
        MergeRules.from_toml(path)


def test_rule_needs_a_tolerance() -> None:
    with pytest.raises(ValueError, match='neither'):
        Rule(paths=('/a',))
