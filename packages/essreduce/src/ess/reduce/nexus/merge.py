# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Merge the NeXus files of consecutive runs into a file that looks like one run.

A dataset whose first dimension is unlimited holds data streamed during the run,
such as the fields of NXlog and NXevent_data groups. The merge concatenates it
along that dimension, in the order of the runs' start times. Every other dataset
and attribute must be identical in all runs, except

- the per-run fields of NXentry (``start_time``, ``end_time``, ...), whose values
  for each run are kept in ``<entry>/merged_runs``;
- ``minimum_value``, ``maximum_value``, and ``average_value`` of an NXlog, which
  are recomputed from the merged values;
- the paths that :attr:`MergeRules.ignore` matches.

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

import shutil
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from itertools import pairwise
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
_LOG_SUMMARIES = {
    'minimum_value': np.min,
    'maximum_value': np.max,
    'average_value': np.mean,
}
_INDEX_FIELDS = {
    ('NXevent_data', 'event_index'): 'event_time_offset',
    ('NXevent_data', 'cue_index'): 'event_time_offset',
    ('NXlog', 'cue_index'): 'time',
}
"""Fields that index into a sibling field, mapped to the name of that sibling."""
_BLOCK_BYTES = 64 * 1024**2
"""Bytes of a dataset copied at once when appending a run."""

_Node = h5py.Group | h5py.Dataset | h5py.SoftLink | h5py.ExternalLink


@dataclass(frozen=True, kw_only=True)
class Rule:
    """
    Tolerances on the spread (maximum minus minimum) of the values of NXlogs.

    The spread is taken over the values of all runs, so a single run whose
    values vary too much fails the check as well.
    """

    paths: tuple[str, ...]
    """
    Glob patterns (:func:`fnmatch.fnmatchcase`) on absolute paths of NXlog groups,
    such as ``'/entry/instrument/*/target_value'``. ``*`` also matches ``/``.
    """
    warn: sc.Variable | None = None
    """A larger spread is a warning."""
    error: sc.Variable | None = None
    """A larger spread is an error. ``0`` requires identical values."""

    def __post_init__(self) -> None:
        if self.warn is None and self.error is None:
            raise ValueError(f'rule for {self.paths} sets neither warn nor error')
        for tolerance in (self.warn, self.error):
            if tolerance is not None and tolerance.value < 0:
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
    Glob patterns on absolute paths of groups and fixed-size datasets whose values
    and attributes may differ between runs. The merged file has those of the
    first run.
    """

    @classmethod
    def from_toml(cls, path: Path | str) -> MergeRules:
        """
        Rules from a TOML file.

        A tolerance is a number in the unit of the log's values, or a string with a
        unit, such as ``'0.5 K'``, converted to the unit of the log's values:

        .. code-block:: toml

            ignore = ["/entry/experiment_description"]

            [[rule]]
            paths = ["/entry/instrument/*/target_value"]
            error = 0

            [[rule]]
            paths = ["/entry/sample/temperature"]
            warn = "0.1 K"
            error = "0.5 K"
        """
        with Path(path).open('rb') as f:
            content = tomllib.load(f)
        _refuse_unknown_keys(content, {'ignore', 'rule'}, f'{path}')
        rules = []
        for rule in content.get('rule', []):
            _refuse_unknown_keys(rule, {'paths', 'warn', 'error'}, f'rule in {path}')
            rules.append(
                Rule(
                    paths=tuple(rule['paths']),
                    warn=_parse_tolerance(rule.get('warn')),
                    error=_parse_tolerance(rule.get('error')),
                )
            )
        return cls(rules=tuple(rules), ignore=tuple(content.get('ignore', ())))

    def rule_for(self, path: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.matches(path)), None)

    def ignores(self, path: str) -> bool:
        return any(fnmatchcase(path, pattern) for pattern in self.ignore)


def _refuse_unknown_keys(table: dict[str, Any], known: set[str], where: str) -> None:
    if unknown := table.keys() - known:
        raise ValueError(f'unknown keys {sorted(unknown)} in {where}')


def _parse_tolerance(value: float | str | None) -> sc.Variable | None:
    if value is None:
        return None
    if isinstance(value, str):
        number, *unit = value.split(maxsplit=1)
        return sc.scalar(float(number), unit=unit[0] if unit else None)
    return sc.scalar(float(value), unit=None)


@dataclass(frozen=True)
class Issue:
    """A finding about the runs to merge."""

    severity: Literal['info', 'warning', 'error']
    path: str
    """Path in the files, or the patterns of a rule."""
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
    """The files of the runs, in the order of their start times."""
    issues: tuple[Issue, ...]
    stamps: tuple[tuple[int, int], ...] = field(repr=False)
    """Size and modification time of each file, to detect a changed file."""

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == 'error')

    def write(self, output: Path | str) -> Path:
        """
        Write the merged file, and return its path.

        Raises if the plan has errors, if a file changed since the plan was
        made, or if ``output`` exists. Removes ``output`` if writing fails.
        """
        output = Path(output)
        if self.errors:
            raise ValueError(
                'cannot merge, the plan has errors:\n'
                + '\n'.join(map(str, self.errors))
            )
        if _stamps(self.files) != self.stamps:
            raise RuntimeError('input files changed since the plan was made')
        if output.exists():
            raise FileExistsError(output)
        shutil.copyfile(self.files[0], output)
        try:
            with ExitStack() as stack:
                runs = [stack.enter_context(h5py.File(p, 'r')) for p in self.files]
                merged = stack.enter_context(h5py.File(output, 'r+'))
                _append_runs(merged, runs)
                _update_logs(merged)
                _update_entry(merged, runs, self.files)
        except BaseException:
            output.unlink()
            raise
        return output


def plan_merge(files: Sequence[Path | str], rules: MergeRules) -> MergePlan:
    """
    Check whether the runs in ``files`` can be merged.

    Reads metadata and NXlogs only, not event data.
    """
    if len(files) < 2:
        raise ValueError('merging needs at least two files')
    with ExitStack() as stack:
        runs = sorted(
            ((Path(p), stack.enter_context(h5py.File(p, 'r'))) for p in files),
            key=lambda run: _entry_time(run[1], 'start_time'),
        )
        issues = [
            *_check_order(runs),
            *_check_layout(runs, rules),
            *_check_logs([file for _, file in runs], rules),
        ]
    paths = tuple(path for path, _ in runs)
    return MergePlan(files=paths, issues=tuple(issues), stamps=_stamps(paths))


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
    exempt = {f'{entry}/{field}' for field in _PER_RUN_ENTRY_FIELDS} | {
        f'{path}/{summary}'
        for path, node in reference.items()
        if _nx_class(node) == 'NXlog'
        for summary in _LOG_SUMMARIES
    }
    for other_path, other in others:
        where = f'{first_path.name} and {other_path.name}'
        layout = _layout(other)
        other_canonical = _canonical_paths(layout)
        for path in sorted(reference.keys() - layout.keys()):
            yield Issue('error', path, f'missing in {other_path.name}')
        for path in sorted(layout.keys() - reference.keys()):
            yield Issue('error', path, f'missing in {first_path.name}')
        for path in sorted(reference.keys() & layout.keys()):
            if canonical[path] != other_canonical[path]:
                yield Issue('error', path, f'hard links differ between {where}')
            elif canonical[path] == path:
                yield from _compare(
                    path,
                    reference[path],
                    layout[path],
                    where=where,
                    compare_values=not (path in exempt or rules.ignores(path)),
                )


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
    exempt = _PER_RUN_ROOT_ATTRS if path == '/' else ()
    names_a = a.attrs.keys() - exempt
    names_b = b.attrs.keys() - exempt
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


def _check_logs(files: list[h5py.File], rules: MergeRules) -> Iterator[Issue]:
    used: set[int] = set()  # ids, a Rule is not hashable
    for path, node in _layout(files[0]).items():
        if _nx_class(node) != 'NXlog' or 'value' not in node:
            continue
        rule = rules.rule_for(path)
        if rule is not None:
            used.add(id(rule))
        values = [file.get(f'{path}/value') for file in files]
        if any(v is None for v in values):
            continue  # reported by _check_layout
        yield from _check_log(path, values, rule)
    for rule in rules.rules:
        if id(rule) not in used:
            yield Issue('warning', ', '.join(rule.paths), 'rule matches no NXlog')


def _check_log(
    path: str, values: list[h5py.Dataset], rule: Rule | None
) -> Iterator[Issue]:
    if any(v.dtype.kind not in 'biuf' for v in values):
        if rule is not None:
            yield Issue(
                'error', path, f'cannot check values of dtype {values[0].dtype}'
            )
        return
    merged = np.concatenate([v[()].astype('float64', copy=False) for v in values])
    if merged.size == 0:
        return
    spread = float(np.max(np.nanmax(merged, axis=0) - np.nanmin(merged, axis=0)))
    unit = values[0].attrs.get('units')
    unit = unit.decode() if isinstance(unit, bytes) else unit
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
                f'spread {spread:g} {unit} exceeds {severity} tolerance {limit:g}',
            )
            return


def _in_unit(tolerance: sc.Variable, unit: str | None) -> float:
    if tolerance.unit is None:
        return tolerance.value
    return tolerance.to(unit=unit, dtype='float64').value


def _append_runs(merged: h5py.File, runs: list[h5py.File]) -> None:
    """Append the extendable datasets of ``runs[1:]`` to those of ``merged``."""
    targets: dict[h5py.Dataset, str] = {}
    for path, node in _layout(merged).items():
        if _is_extendable(node):
            targets.setdefault(node, path)  # once per object, not per hard link
    for target, path in targets.items():
        parent, name = path.rsplit('/', 1)
        indexed = _INDEX_FIELDS.get((_nx_class(merged[parent or '/']), name))
        offsets = (
            np.cumsum([run[f'{parent}/{indexed}'].shape[0] for run in runs[:-1]])
            if indexed is not None
            else np.zeros(len(runs) - 1, dtype=int)
        )
        _append(target, [run[path] for run in runs[1:]], offsets)


def _append(
    target: h5py.Dataset, parts: list[h5py.Dataset], offsets: np.ndarray
) -> None:
    start = target.shape[0]
    target.resize(start + sum(part.shape[0] for part in parts), axis=0)
    row_bytes = target.dtype.itemsize * int(np.prod(target.shape[1:]))
    step = max(1, _BLOCK_BYTES // max(1, row_bytes))
    for part, offset in zip(parts, offsets, strict=True):
        for begin in range(0, part.shape[0], step):
            block = part[begin : begin + step]
            if offset:
                block += block.dtype.type(offset)
            target[start + begin : start + begin + len(block)] = block
        start += part.shape[0]


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
    merged: h5py.File, runs: list[h5py.File], files: Sequence[Path]
) -> None:
    """Set the end time of the last run, and keep the per-run fields of all runs."""
    entry = _entry(merged)
    if 'end_time' in entry:
        entry['end_time'][()] = _entry(runs[-1])['end_time'][()]
    merged_runs = entry.create_group('merged_runs')
    merged_runs.attrs['NX_class'] = 'NXcollection'
    for i, (run, path) in enumerate(zip(runs, files, strict=True)):
        group = merged_runs.create_group(f'run_{i}')
        group.attrs['NX_class'] = 'NXcollection'
        group.create_dataset('file_name', data=path.name, dtype=h5py.string_dtype())
        run_entry = _entry(run)
        for name in _PER_RUN_ENTRY_FIELDS:
            if name in run_entry:
                run.copy(run_entry[name], group, name=name)
