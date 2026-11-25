# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Merge the NeXus files of consecutive runs into a file that looks like one run.

A single file is accepted as well, which gives a copy with the padding stripped.
A merged file can be merged again, with more runs or with other merged files.

A dataset whose first dimension is unlimited holds data streamed during the run,
such as the fields of NXlog and NXevent_data groups, and of the NXdata groups of
monitors in histogram mode, with a histogram per row. The merge concatenates it
along that dimension, in the order of the runs' start times. Every other dataset
and attribute must be identical in all runs, except

- the per-run fields of NXentry (``start_time``, ``end_time``, ...), whose values
  for each run are kept in ``<entry>/merged_runs``, which lists the runs of a
  merged input in place of the input;
- ``minimum_value``, ``maximum_value``, and ``average_value`` of an NXlog, which
  are recomputed from the merged values;
- the paths that :attr:`MergeRules.ignore` matches.

The file writer allocates more rows than it fills and leaves the rest with
time 0. Those rows are an error, unless :attr:`MergeRules.strip_padding` is set,
which drops them from the start and the end of a series.

The file writer starts each series of an NXlog with the values from shortly
before the run. Those rows repeat the last rows of the previous run if it ended
shortly before. The merge drops them, so a log of a device that did not move
keeps a single value, as in a single run.

:class:`MergeRules` also says how much the values of an NXlog may vary over all
runs: the merge is meant for runs whose device settings are the same.
Make a plan, look at its issues, then write the merged file:

.. code-block:: python

    plan = plan_merge(files, MergeRules.from_toml(rules_file))
    for issue in plan.issues:
        print(issue)
    plan.write(output)

The merged file is the first file with the data of the other runs appended, so
it keeps the attributes, links, chunking, and compression of the first file.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from itertools import count, pairwise
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import scipp as sc
import tomllib

_PER_RUN_ENTRY_FIELDS = (
    'start_time',
    'end_time',
    'entry_identifier',
    'entry_identifier_uuid',
    'scripts',
)
_PER_RUN_ROOT_ATTRS = ('file_name', 'file_time')
_MERGED_RUNS = 'merged_runs'
_LOG_SUMMARIES = {
    'minimum_value': np.min,
    'maximum_value': np.max,
    'average_value': np.mean,
}
_LOG_SERIES = {
    'time': ('value',),
    'alarm_time': ('alarm_message', 'alarm_severity'),
    'connection_status_time': ('connection_status',),
}
"""
The time fields of the series in an NXlog, mapped to the fields with a row per time.
"""
_INDEX_FIELDS = {
    ('NXevent_data', 'event_index'): 'event_time_offset',
    ('NXevent_data', 'cue_index'): 'event_time_offset',
    ('NXlog', 'cue_index'): 'time',
}
"""Fields that index into a sibling field, mapped to the name of that sibling."""
_INDEX_NAMES = {name for _, name in _INDEX_FIELDS}
_LOG_STREAMS = {
    *_LOG_SERIES,
    *(name for fields in _LOG_SERIES.values() for name in fields),
    'cue_timestamp_zero',
    'cue_index',
}
"""The extendable fields of an NXlog that the merge knows how to append."""
_COUNT_TIMES = {'NXevent_data': 'event_time_zero', 'NXdata': 'time'}
"""
The time field of groups of neutron counts, with a row per pulse, or per message
of histograms such as those of monitors in histogram mode.
"""
_BLOCK_BYTES = 64 * 1024**2
"""Most bytes of a dataset held in memory at once when appending a run."""
_TOLERANCE = re.compile(r'\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*')

_Node = h5py.Group | h5py.Dataset | h5py.SoftLink | h5py.ExternalLink


@dataclass(frozen=True, kw_only=True)
class Rule:
    """
    Tolerances on the spread (maximum minus minimum) of the values of NXlogs.

    The spread is taken over the values of all runs, so a single run whose
    values vary too much fails the check as well. A tolerance is converted to
    the unit of the log. Scipp does not convert between units with an offset,
    such as K and degC, even though a spread is a difference: write the
    tolerance of such a log in the log's unit.
    """

    paths: tuple[str, ...]
    """
    Glob patterns (:func:`fnmatch.fnmatchcase`) on absolute paths of NXlog groups,
    such as ``'/entry/instrument/*/target_value'``. ``*`` also matches ``/``.
    """
    warn: sc.Variable | None = None
    """A larger spread is a warning."""
    error: sc.Variable | None = None
    """A larger spread is an error. Zero, in any unit, requires identical values."""

    def __post_init__(self) -> None:
        if isinstance(self.paths, str) or not self.paths:
            raise ValueError(
                f'paths of a rule must be a tuple of patterns: {self.paths}'
            )
        if self.warn is None and self.error is None:
            raise ValueError(f'rule for {self.paths} sets neither warn nor error')
        for tolerance in (self.warn, self.error):
            if tolerance is None:
                continue
            # A unit-less tolerance would silently take the log's unit, which
            # file producers change, e.g., from mm to m.
            if tolerance.unit is None:
                raise ValueError(f'rule for {self.paths} has a tolerance without unit')
            if tolerance.value < 0:
                raise ValueError(f'rule for {self.paths} has a negative tolerance')

    def matches(self, path: str) -> bool:
        return any(fnmatchcase(path, pattern) for pattern in self.paths)


@dataclass(frozen=True, kw_only=True)
class MergeRules:
    """Which differences between runs prevent merging them."""

    rules: tuple[Rule, ...] = ()
    """
    Checks of NXlogs; the first rule that matches a log applies. Logs that no
    rule matches are not checked.
    """
    ignore: tuple[str, ...] = ()
    """
    Glob patterns on absolute paths of groups and datasets whose values,
    attributes, and contents may differ between runs. The merged file has those
    of the first run. Extendable datasets in them are appended all the same.
    """

    strip_padding: bool = False
    """
    Drop the rows of the series of NXlogs that the file writer padded with time 0,
    which it does by allocating more rows than it fills. The padding must be a
    block at the start or the end of a series, and zero in all fields of the
    series. By default, such rows are an error.
    """

    def __post_init__(self) -> None:
        if isinstance(self.ignore, str):
            raise ValueError(f'ignore must be a tuple of patterns: {self.ignore}')

    @classmethod
    def from_toml(cls, path: Path | str) -> MergeRules:
        """
        Rules from a TOML file.

        A tolerance is a string with a unit, such as ``'0.5 K'``, converted to the
        unit of the log's values. A number is dimensionless, except that ``0``
        requires identical values in any unit:

        .. code-block:: toml

            ignore = ["/entry/experiment_description"]
            strip_padding = true

            [[rule]]
            paths = ["/entry/instrument/*/target_value"]
            error = 0

            [[rule]]
            paths = ["/entry/sample/temperature"]
            warn = "0.1 K"
            error = "0.5 K"

        Parameters
        ----------
        path:
            The TOML file.

        Returns
        -------
        :
            The rules in the file.
        """
        with Path(path).open('rb') as f:
            content = tomllib.load(f)
        _refuse_unknown_keys(content, {'ignore', 'rule', 'strip_padding'}, f'{path}')
        rules = []
        for i, rule in enumerate(content.get('rule', [])):
            where = f'rule {i} in {path}'
            _refuse_unknown_keys(rule, {'paths', 'warn', 'error'}, where)
            try:
                rules.append(
                    Rule(
                        paths=_patterns(rule.get('paths'), 'paths'),
                        warn=_parse_tolerance(rule.get('warn')),
                        error=_parse_tolerance(rule.get('error')),
                    )
                )
            except (ValueError, sc.UnitError) as err:
                raise ValueError(f'{where}: {err}') from err
        try:
            ignore = _patterns(content.get('ignore', []), 'ignore')
        except ValueError as err:
            raise ValueError(f'{path}: {err}') from err
        strip_padding = content.get('strip_padding', False)
        if not isinstance(strip_padding, bool):
            raise ValueError(
                f'{path}: strip_padding must be true or false, got {strip_padding!r}'
            )
        return cls(rules=tuple(rules), ignore=ignore, strip_padding=strip_padding)

    def rule_for(self, path: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.matches(path)), None)

    def ignores(self, path: str) -> bool:
        """Whether ``path`` or a group that contains it is ignored."""
        return any(
            fnmatchcase(group, pattern)
            for group in _self_and_ancestors(path)
            for pattern in self.ignore
        )


def _refuse_unknown_keys(table: dict[str, Any], known: set[str], where: str) -> None:
    if unknown := table.keys() - known:
        raise ValueError(f'unknown keys {sorted(unknown)} in {where}')


def _patterns(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        raise ValueError(f'{name} must be a list of strings, got {value!r}')
    return tuple(value)


def _parse_tolerance(value: Any) -> sc.Variable | None:
    if value is None:
        return None
    if isinstance(value, str):
        match = _TOLERANCE.fullmatch(value)
        if match is None:
            raise ValueError(
                f'cannot parse tolerance {value!r}, expected a number and a unit, '
                'such as "0.5 K"'
            )
        number, unit = match.groups()
        return sc.scalar(float(number), unit=unit)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return sc.scalar(float(value))
    raise ValueError(f'a tolerance is a number or a string, got {value!r}')


def _self_and_ancestors(path: str) -> list[str]:
    """``'/a/b'`` gives ``['/a', '/a/b']``, and ``'/'`` gives ``['/']``."""
    parts = path.strip('/').split('/')
    return ['/' + '/'.join(parts[: i + 1]) for i in range(len(parts))]


@dataclass(frozen=True)
class Issue:
    """A finding about the runs to merge."""

    severity: Literal['info', 'warning', 'error']
    path: str
    """Path in the files, a file name, or the patterns of a rule."""
    message: str

    def __str__(self) -> str:
        return f'{self.severity}: {self.path}: {self.message}'


@dataclass(frozen=True)
class MergePlan:
    """
    Runs to merge, and what the checks found. Holds no data of the runs.

    Make one with :func:`plan_merge`.
    """

    files: tuple[Path, ...]
    """The files of the runs, absolute, in the order of their start times."""
    issues: tuple[Issue, ...]
    stamps: tuple[tuple[int, int], ...] = field(repr=False)
    """Size and modification time of each file, to detect a changed file."""
    strip_padding: bool = field(repr=False)

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == 'error')

    def write(self, output: Path | str) -> Path:
        """
        Write the merged file.

        The file is written under a temporary name next to ``output`` and renamed
        when complete, so ``output`` never holds a partial file. A partial file
        would look valid: it starts as a copy of the first run's file, so
        reducing it would silently give wrong results.

        Parameters
        ----------
        output:
            Path of the merged file. Must not exist.

        Returns
        -------
        :
            ``output``.

        Raises
        ------
        ValueError
            If the plan has errors.
        RuntimeError
            If a file changed since the plan was made.
        FileExistsError
            If ``output`` exists.
        """
        output = Path(output).resolve()
        if self.errors:
            raise ValueError(
                'cannot merge, the plan has errors:\n'
                + '\n'.join(map(str, self.errors))
            )
        if _stamps(self.files) != self.stamps:
            raise RuntimeError('input files changed since the plan was made')
        if output.exists():
            raise FileExistsError(output)
        partial = output.with_name(f'{output.name}.partial')
        try:
            shutil.copyfile(self.files[0], partial)
            with ExitStack() as stack:
                sources = [stack.enter_context(h5py.File(p, 'r')) for p in self.files]
                merged = stack.enter_context(h5py.File(partial, 'r+'))
                _append_runs(merged, sources, self.strip_padding)
                _update_logs(merged)
                _update_entry(merged, sources, self.files)
                _update_root(merged, output)
            partial.replace(output)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return output


def plan_merge(files: Sequence[Path | str], rules: MergeRules) -> MergePlan:
    """
    Check whether the runs in ``files`` can be merged.

    Reads metadata, NXlogs, and pulse times only, not events.

    Parameters
    ----------
    files:
        The NeXus files of the runs, in any order.
    rules:
        Which differences between the runs prevent merging them.

    Returns
    -------
    :
        The runs in the order of their start times, and the issues found.
    """
    if not files:
        raise ValueError('merging needs at least one file')
    with ExitStack() as stack:
        runs = sorted(
            (
                (Path(p).resolve(), stack.enter_context(h5py.File(p, 'r')))
                for p in files
            ),
            key=lambda run: _entry_time(run[1], 'start_time'),
        )
        sources = [file for _, file in runs]
        issues = [
            *_check_order(runs),
            *_check_layout(runs, rules),
            *_check_streams(sources),
            *_check_pulses(sources),
            *_check_logs(sources, rules),
        ]
    paths = tuple(path for path, _ in runs)
    return MergePlan(
        files=paths,
        issues=tuple(issues),
        stamps=_stamps(paths),
        strip_padding=rules.strip_padding,
    )


def _stamps(files: Sequence[Path]) -> tuple[tuple[int, int], ...]:
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in files)


def _entry(file: h5py.File) -> h5py.Group:
    entries = [g for g in file.values() if _nx_class(g) == 'NXentry']
    if len(entries) != 1:
        raise ValueError(f'{file.filename} has {len(entries)} NXentry groups, not 1')
    return entries[0]


def _entry_time(file: h5py.File, name: str) -> datetime:
    return datetime.fromisoformat(_entry(file)[name].asstr()[()])


def _nx_class(node: _Node) -> str | None:
    if not isinstance(node, h5py.Group | h5py.Dataset):
        return None
    nx_class = node.attrs.get('NX_class')
    return nx_class.decode() if isinstance(nx_class, bytes) else nx_class


def _is_extendable(node: _Node) -> bool:
    return (
        isinstance(node, h5py.Dataset)
        and node.maxshape is not None
        and len(node.maxshape) > 0
        and node.maxshape[0] is None
    )


def _layout(file: h5py.File) -> dict[str, _Node]:
    """
    Every path in ``file``, mapped to its object, or to its link if the link is
    soft or external.

    A group reached by several hard links is listed once with its contents.
    """
    layout: dict[str, _Node] = {'/': file}
    seen = {file}

    def walk(group: h5py.Group, prefix: str) -> None:
        for name in group:
            path = f'{prefix}/{name}'
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                layout[path] = link
                continue
            node = group[name]
            layout[path] = node
            if isinstance(node, h5py.Group) and node not in seen:
                seen.add(node)
                walk(node, path)

    walk(file, '')
    return layout


def _canonical_paths(layout: dict[str, _Node]) -> dict[str, str]:
    """The first path of each object, by path; hard links share the object."""
    first: dict[_Node, str] = {}
    return {
        path: first.setdefault(node, path)
        if isinstance(node, h5py.Group | h5py.Dataset)
        else path
        for path, node in layout.items()
    }


def _streams(file: h5py.File) -> dict[h5py.Dataset, list[str]]:
    """The extendable datasets of ``file``, each with all its paths."""
    streams: dict[h5py.Dataset, list[str]] = {}
    for path, node in _layout(file).items():
        if _is_extendable(node):
            streams.setdefault(node, []).append(path)
    return streams


def _parent(file: h5py.File, path: str) -> h5py.Group:
    return file[path.rsplit('/', 1)[0] or '/']


def _check_order(runs: list[tuple[Path, h5py.File]]) -> Iterator[Issue]:
    for (path_a, a), (path_b, b) in pairwise(runs):
        if _entry_time(a, 'end_time') > _entry_time(b, 'start_time'):
            yield Issue('error', path_b.name, f'starts before the end of {path_a.name}')


def _check_layout(
    runs: list[tuple[Path, h5py.File]], rules: MergeRules
) -> Iterator[Issue]:
    (first_path, first), *others = runs
    reference = _layout(first)
    canonical = _canonical_paths(reference)
    entry = _entry(first).name
    per_run = {f'{entry}/{name}' for name in (*_PER_RUN_ENTRY_FIELDS, _MERGED_RUNS)} | {
        f'{path}/{summary}'
        for path, node in reference.items()
        if _nx_class(node) == 'NXlog'
        for summary in _LOG_SUMMARIES
    }

    def may_differ(path: str) -> bool:
        return any(
            group in per_run for group in _self_and_ancestors(path)
        ) or rules.ignores(path)

    for other_path, other in others:
        where = f'{first_path.name} and {other_path.name}'
        layout = _layout(other)
        other_canonical = _canonical_paths(layout)
        for path in sorted(reference.keys() - layout.keys()):
            if not may_differ(path) or _is_extendable(reference[path]):
                yield Issue('error', path, f'missing in {other_path.name}')
        for path in sorted(layout.keys() - reference.keys()):
            if not may_differ(path) or _is_extendable(layout[path]):
                yield Issue('error', path, f'missing in {first_path.name}')
        for path in sorted(reference.keys() & layout.keys()):
            a, b = reference[path], layout[path]
            if may_differ(path):
                # Only what appending needs: an extendable dataset in both.
                if _is_extendable(a) or _is_extendable(b):
                    yield from _compare(path, a, b, where=where, compare_values=False)
            elif canonical[path] != other_canonical[path]:
                yield Issue('error', path, f'hard links differ between {where}')
            elif canonical[path] == path:
                yield from _compare(path, a, b, where=where, compare_values=True)


def _compare(
    path: str, a: _Node, b: _Node, *, where: str, compare_values: bool
) -> Iterator[Issue]:
    if type(a) is not type(b):
        yield Issue('error', path, f'kind of object differs between {where}')
    elif isinstance(a, h5py.SoftLink):
        if a.path != b.path:
            yield Issue('error', path, f'soft link targets differ between {where}')
    elif isinstance(a, h5py.ExternalLink):
        if (a.filename, a.path) != (b.filename, b.path):
            yield Issue('error', path, f'external link targets differ between {where}')
    else:
        if isinstance(a, h5py.Dataset):
            yield from _compare_datasets(
                path, a, b, where=where, compare_values=compare_values
            )
        if compare_values:
            yield from _compare_attrs(path, a, b, where=where)


def _compare_datasets(
    path: str, a: h5py.Dataset, b: h5py.Dataset, *, where: str, compare_values: bool
) -> Iterator[Issue]:
    if a.id.get_type() != b.id.get_type():
        yield Issue('error', path, f'dtypes {a.dtype} and {b.dtype} of {where} differ')
    elif _is_extendable(a) != _is_extendable(b):
        yield Issue('error', path, f'only one of {where} has an unlimited dimension')
    elif _is_extendable(a):
        if a.shape[1:] != b.shape[1:]:
            yield Issue('error', path, f'shapes {a.shape} and {b.shape} of {where}')
    elif a.shape != b.shape:
        yield Issue('error', path, f'shapes {a.shape} and {b.shape} of {where}')
    elif compare_values and not _equal(a[()], b[()]):
        yield Issue('error', path, f'values differ between {where}')


def _compare_attrs(
    path: str, a: h5py.HLObject, b: h5py.HLObject, *, where: str
) -> Iterator[Issue]:
    per_run = _PER_RUN_ROOT_ATTRS if path == '/' else ()
    names_a = a.attrs.keys() - per_run
    names_b = b.attrs.keys() - per_run
    if names_a != names_b:
        yield Issue('error', path, f'attribute names differ between {where}')
    for name in sorted(names_a & names_b):
        if not _equal(a.attrs[name], b.attrs[name]):
            yield Issue('error', path, f'attribute {name!r} differs between {where}')


def _equal(a: Any, b: Any) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'))


def _check_streams(sources: list[h5py.File]) -> Iterator[Issue]:
    """Refuse extendable datasets that the merge does not know how to append."""
    for paths in _streams(sources[0]).values():
        for path in paths:
            name = path.rsplit('/', 1)[1]
            nx_class = _nx_class(_parent(sources[0], path))
            if nx_class == 'NXlog' and name not in _LOG_STREAMS:
                yield Issue('error', path, 'unknown field of an NXlog')
            elif (
                name in _INDEX_NAMES
                and (nx_class, name) not in _INDEX_FIELDS
                and any(source[path].shape[0] for source in sources)
            ):
                yield Issue('error', path, f'cannot offset {name} in a {nx_class}')


def _check_pulses(sources: list[h5py.File]) -> Iterator[Issue]:
    """
    Refuse runs whose pulses overlap, as the merged file could count neutrons twice.

    Messages from Kafka arrive out of order, so the pulses of a run are not
    sorted. The earliest pulse of a run must be later than the latest pulse of
    the runs before. The same holds for the rows of histograms (``NXdata``).
    """
    for path, node in _layout(sources[0]).items():
        time_name = _COUNT_TIMES.get(_nx_class(node))
        if time_name is None or not _is_extendable(node.get(time_name)):
            continue
        latest = None
        for source in sources:
            if (pulses := source.get(f'{path}/{time_name}')) is None:
                break  # reported by _check_layout
            times = pulses[()]
            if times.size == 0:
                continue
            if latest is not None and times.min() <= latest:
                name = Path(source.filename).name
                yield Issue(
                    'error', path, f'{time_name} in {name} overlaps previous runs'
                )
                break
            latest = times.max() if latest is None else max(latest, times.max())


def _check_logs(sources: list[h5py.File], rules: MergeRules) -> Iterator[Issue]:
    used: set[int] = set()  # ids, a Rule is not hashable
    seen: set[h5py.Dataset] = set()  # logs whose fields are hard-linked are alike
    for path, node in _layout(sources[0]).items():
        if _nx_class(node) != 'NXlog':
            continue
        rule = rules.rule_for(path)
        if rule is not None:
            used.add(id(rule))
        logs = [source.get(path) for source in sources]
        key = node.get('value', node.get('time'))
        if any(log is None for log in logs) or key is None or key in seen:
            continue  # missing logs are reported by _check_layout
        seen.add(key)
        issues, series = _check_log_series(
            path, logs, strip_padding=rules.strip_padding
        )
        yield from issues
        if any(issue.severity == 'error' for issue in issues) or 'value' not in node:
            continue
        valid = (
            [run.valid for run in series['time']]
            if 'time' in series
            else [range(log['value'].shape[0]) for log in logs]
        )
        yield from _check_log_values(path, [log['value'] for log in logs], valid, rule)
    for rule in rules.rules:
        if id(rule) not in used:
            yield Issue('warning', ', '.join(rule.paths), 'rule matches no NXlog')


def _check_log_series(
    path: str, logs: list[h5py.Group], *, strip_padding: bool
) -> tuple[list[Issue], dict[str, list[_Rows]]]:
    """The issues of the series of a log, and the rows of each valid series."""
    issues: list[Issue] = []
    series: dict[str, list[_Rows]] = {}
    for time_name in _LOG_SERIES:
        if time_name not in logs[0]:
            continue
        try:
            rows = _series_rows(logs, time_name, strip_padding=strip_padding)
        except ValueError as err:
            issues.append(Issue('error', path, str(err)))
            continue
        series[time_name] = rows
        for log, run in zip(logs, rows, strict=True):
            if (stripped := log[time_name].shape[0] - len(run.valid)) > 0:
                name = Path(log.file.filename).name
                issues.append(
                    Issue(
                        'info', path, f'{time_name} in {name}: stripped {stripped} rows'
                    )
                )
    return issues, series


@dataclass(frozen=True)
class _Rows:
    """The rows of a series of one run that go into the merged file."""

    valid: range
    """The rows that are not padding."""
    keep: range
    """The valid rows without the leading rows that repeat the previous run."""


def _valid_rows(
    log: h5py.Group, time_name: str, names: list[str], *, strip_padding: bool
) -> range:
    """The rows of a series that are not padding by the file writer."""
    times = log[time_name][()]
    run = Path(log.file.filename).name
    for name in names:
        if (rows := log[name].shape[0]) != len(times):
            raise ValueError(
                f'{name} in {run} has {rows} rows, but {time_name} has {len(times)}'
            )
    if not np.any(times == 0):
        return range(len(times))
    if not strip_padding:
        raise ValueError(
            f'{time_name} in {run} has rows with time 0, possibly padding by the '
            'file writer'
        )
    (written,) = np.nonzero(times)
    valid = range(written[0], written[-1] + 1) if written.size else range(0)
    if np.any(times[valid.start : valid.stop] == 0):
        raise ValueError(f'{time_name} in {run} has rows with time 0 between others')
    for name in names:
        field = log[name]
        zero = np.zeros((), field.dtype)
        if np.any(field[: valid.start] != zero) or np.any(field[valid.stop :] != zero):
            raise ValueError(f'{name} in {run} has data in rows with time 0')
    return valid


def _series_rows(
    logs: list[h5py.Group], time_name: str, *, strip_padding: bool
) -> list[_Rows]:
    """
    The rows of a series of each run that go into the merged file.

    The file writer does not guarantee sorted times, so the rows of a run may be
    unsorted. The leading rows not later than the latest time of the previous run
    are dropped if they are identical to the last rows of the previous run.
    Otherwise, they are an error, as are later rows that are not later than the
    previous run.
    """
    names = [name for name in (time_name, *_LOG_SERIES[time_name]) if name in logs[0]]
    rows: list[_Rows] = []
    previous: h5py.Group | None = None  # the last run with rows in the series
    previous_valid = range(0)
    for log in logs:
        valid = _valid_rows(log, time_name, names, strip_padding=strip_padding)
        times = log[time_name][valid.start : valid.stop]
        drop = 0
        if previous is not None:
            previous_times = previous[time_name][
                previous_valid.start : previous_valid.stop
            ]
            latest = previous_times.max()
            later = times > latest
            drop = int(np.argmax(later)) if later.any() else len(times)
            tail = previous_valid.stop - drop
            if (
                tail < previous_valid.start
                or np.any(times[drop:] <= latest)
                or not all(
                    _equal(
                        log[name][valid.start : valid.start + drop],
                        previous[name][tail : previous_valid.stop],
                    )
                    for name in names
                )
            ):
                raise ValueError(
                    f'{time_name} of {Path(log.file.filename).name} overlaps the '
                    'previous run'
                )
        rows.append(_Rows(valid=valid, keep=range(valid.start + drop, valid.stop)))
        if len(times) > drop:
            previous, previous_valid = log, valid
    return rows


def _check_log_values(
    path: str, values: list[h5py.Dataset], valid: list[range], rule: Rule | None
) -> Iterator[Issue]:
    if any(v.dtype.kind not in 'biuf' for v in values):
        if rule is not None:
            yield Issue(
                'error', path, f'cannot check values of dtype {values[0].dtype}'
            )
        return
    units = [_unit(v) for v in values]
    if len(set(units)) > 1:
        yield Issue('error', path, f'units differ between runs: {units}')
        return
    unit = units[0]
    merged = np.concatenate(
        [v[rows.start : rows.stop] for v, rows in zip(values, valid, strict=True)]
    )
    if merged.size == 0:
        return
    if merged.dtype.kind == 'f' and np.isnan(merged).any():
        if rule is not None:
            yield Issue('error', path, 'cannot check values that include NaN')
        return
    if merged.dtype.kind == 'b':
        merged = merged.astype('int8')
    # In the dtype of the values, as float64 cannot hold large integers exactly
    spread = np.max(np.max(merged, axis=0) - np.min(merged, axis=0)).item()
    if rule is None:
        if spread > 0:
            yield Issue('info', path, f'not checked, spread is {spread:g} {unit}')
        return
    for severity, tolerance in (('error', rule.error), ('warning', rule.warn)):
        if tolerance is None:
            continue
        try:
            limit = _in_unit(tolerance, unit)
        except sc.UnitError as err:
            yield Issue('error', path, f'cannot convert tolerance: {err}')
            return
        if spread > limit:
            yield Issue(
                severity,
                path,
                f'spread {spread:g} {unit} exceeds {severity} tolerance '
                f'{tolerance.value:g} {tolerance.unit}',
            )
            return


def _unit(dataset: h5py.Dataset) -> str | None:
    unit = dataset.attrs.get('units')
    return unit.decode() if isinstance(unit, bytes) else unit


def _in_unit(tolerance: sc.Variable, unit: str | None) -> float:
    if tolerance.value == 0:
        return 0.0  # the same in any unit, and needs none
    return tolerance.to(unit=unit, dtype='float64').value


def _append_runs(
    merged: h5py.File, sources: list[h5py.File], strip_padding: bool
) -> None:
    """Write the kept rows of the extendable datasets of ``sources`` to ``merged``."""
    for target, paths in _streams(merged).items():
        # Of the paths of a hard-linked dataset, one in an NXlog says which rows
        # to keep.
        path = next(
            (p for p in paths if _nx_class(_parent(merged, p)) == 'NXlog'), paths[0]
        )
        parent, name = path.rsplit('/', 1)
        nx_class = _nx_class(_parent(merged, path))
        parts = [source[path] for source in sources]
        rows = _rows(sources, parent, nx_class, name, strip_padding)
        offsets = [0] * len(sources)
        clips: list[range | None] = [None] * len(sources)
        if (indexed := _INDEX_FIELDS.get((nx_class, name))) is not None:
            indexed_rows = _rows(sources, parent, nx_class, indexed, strip_padding)
            kept = [len(run.keep) for run in indexed_rows]
            # An index of a dropped row points to the identical row before it.
            offsets = [
                sum(kept[:k]) - run.keep.start for k, run in enumerate(indexed_rows)
            ]
            # An index of a padding row points to the nearest row that is not.
            clips = [
                run.valid
                if run.valid != range(source[f'{parent}/{indexed}'].shape[0])
                else None
                for source, run in zip(sources, indexed_rows, strict=True)
            ]
        _write(
            target,
            parts,
            rows=[run.keep for run in rows],
            offsets=offsets,
            clips=clips,
        )


def _rows(
    sources: list[h5py.File],
    parent: str,
    nx_class: str | None,
    name: str,
    strip_padding: bool,
) -> list[_Rows]:
    """The rows of ``parent/name`` of each source that go into the merged file."""
    if nx_class == 'NXlog':
        for time_name, fields in _LOG_SERIES.items():
            if name == time_name or name in fields:
                return _series_rows(
                    [source[parent] for source in sources],
                    time_name,
                    strip_padding=strip_padding,
                )
    return [
        _Rows(valid=range(n), keep=range(n))
        for n in (source[f'{parent}/{name}'].shape[0] for source in sources)
    ]


def _write(
    target: h5py.Dataset,
    parts: list[h5py.Dataset],
    *,
    rows: list[range],
    offsets: list[int],
    clips: list[range | None],
) -> None:
    """
    Make ``target`` hold ``rows`` of ``parts``, one after the other.

    Indices in a part are moved by its offset after clipping them to its clip.
    The first part is the same as the content of ``target``, which is left alone
    if all of it is kept.
    """
    target.resize(sum(len(run) for run in rows), axis=0)
    row_bytes = target.dtype.itemsize * int(np.prod(target.shape[1:]))
    step = max(1, _BLOCK_BYTES // max(1, row_bytes))
    start = 0
    for k, (part, keep, offset, clip) in enumerate(
        zip(parts, rows, offsets, clips, strict=True)
    ):
        if k == 0 and keep == range(part.shape[0]) and not offset and clip is None:
            start = len(keep)  # already in place
            continue
        for begin in range(keep.start, keep.stop, step):
            block = part[begin : min(begin + step, keep.stop)]
            if clip is not None and len(clip):
                block = np.clip(block, clip.start, clip.stop - 1)
            if offset:
                block += block.dtype.type(offset)
            target[start : start + len(block)] = block
            start += len(block)


def _update_logs(merged: h5py.File) -> None:
    for node in _layout(merged).values():
        if _nx_class(node) != 'NXlog' or 'value' not in node:
            continue
        values = node['value'][()]
        if values.size == 0 or values.dtype.kind not in 'biuf':
            continue
        for name, summarize in _LOG_SUMMARIES.items():
            if name in node:
                node[name][()] = summarize(values)


def _update_entry(
    merged: h5py.File, sources: list[h5py.File], files: Sequence[Path]
) -> None:
    """
    Set the end time of the last run, and keep the per-run fields of all runs.

    A merged source contributes the runs it lists, not a run of its own.
    """
    entry = _entry(merged)
    if 'end_time' in entry:
        entry['end_time'][()] = _entry(sources[-1])['end_time'][()]
    if _MERGED_RUNS in entry:  # the first source is a merged file
        del entry[_MERGED_RUNS]
    merged_runs = entry.create_group(_MERGED_RUNS)
    merged_runs.attrs['NX_class'] = 'NXcollection'
    names = (f'run_{i}' for i in count())
    for source, path in zip(sources, files, strict=True):
        source_entry = _entry(source)
        if (runs := source_entry.get(_MERGED_RUNS)) is not None:
            for i in range(len(runs)):
                source.copy(runs[f'run_{i}'], merged_runs, name=next(names))
            continue
        group = merged_runs.create_group(next(names))
        group.attrs['NX_class'] = 'NXcollection'
        group.create_dataset('file_name', data=path.name, dtype=h5py.string_dtype())
        for name in _PER_RUN_ENTRY_FIELDS:
            if name in source_entry:
                source.copy(source_entry[name], group, name=name)


def _update_root(merged: h5py.File, output: Path) -> None:
    """Describe the merged file in the attributes the file writer sets."""
    if 'file_name' in merged.attrs:
        merged.attrs['file_name'] = str(output)
    if 'file_time' in merged.attrs:
        now = datetime.now(UTC).isoformat(timespec='milliseconds')
        merged.attrs['file_time'] = now.replace('+00:00', 'Z')
