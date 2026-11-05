# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Patches for fixing defects in NeXus files while loading them.

A patch is a list of operations, applied in order to a read-only view of a file.
The file on disk is not modified and no data is copied:
Datasets are read from the original file when accessed.

All paths in operations are absolute paths in the patched file,
i.e., they take preceding operations into account.
Patches apply to paths, not to objects:
If an object is reachable under multiple paths through hard links,
a patch for one path does not affect the others.
Soft links are resolved in the patched file.
As in HDF5, moving or removing the target of a soft link breaks the link.

Patches are typically stored in TOML files and loaded with :func:`load_patch`:

.. code-block:: toml

    # Explain the defect being fixed here.
    [[op]]
    kind = "move"
    source = "/entry/instrument/detector/value"
    target = "/entry/instrument/detector/transformations/motion"

    [[op]]
    kind = "set_attrs"
    path = "/entry/instrument/detector/transformations/motion"
    attrs = { transformation_type = "translation", vector = [0.0, 0.0, 1.0] }

A patch is a plain list, so it can be amended in memory before use:

.. code-block:: python

    from ess.reduce.nexus import patch

    p = patch.load_patch('fix.toml') + [patch.Remove(path='/entry/broken')]
    workflow[NeXusFilePatch[SampleRun]] = p
"""

from __future__ import annotations

import os
import posixpath
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any

import h5py
import scippnexus as snx
import tomllib


@dataclass(frozen=True, kw_only=True)
class SetAttrs:
    """Add attributes or replace existing ones."""

    path: str
    attrs: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class DeleteAttrs:
    """Delete attributes. All attributes must exist."""

    path: str
    names: Sequence[str]


@dataclass(frozen=True, kw_only=True)
class Remove:
    """Remove a group or dataset."""

    path: str


@dataclass(frozen=True, kw_only=True)
class Move:
    """Move or rename a group or dataset. The target must not exist."""

    source: str
    target: str


@dataclass(frozen=True, kw_only=True)
class SetDataset:
    """Add a dataset or replace an existing dataset.

    The parent group must exist.
    A replaced dataset does not keep its attributes.
    """

    path: str
    value: Any


PatchOp = SetAttrs | DeleteAttrs | Remove | Move | SetDataset

_op_kinds: dict[str, type[PatchOp]] = {
    'set_attrs': SetAttrs,
    'delete_attrs': DeleteAttrs,
    'remove': Remove,
    'move': Move,
    'set_dataset': SetDataset,
}


def load_patch(path: str | os.PathLike[str]) -> list[PatchOp]:
    """Load a patch from a TOML file.

    The file contains an array of tables named ``op``.
    Each table has a ``kind`` (one of ``set_attrs``, ``delete_attrs``, ``remove``,
    ``move``, ``set_dataset``) and the fields of the corresponding operation class.
    """
    with open(path, 'rb') as f:
        content = tomllib.load(f)
    if unknown := set(content) - {'op'}:
        raise ValueError(f"Unknown top-level keys in NeXus patch {path}: {unknown}")
    ops = []
    for entry in content.get('op', []):
        fields = dict(entry)
        kind = fields.pop('kind')
        if (op := _op_kinds.get(kind)) is None:
            raise ValueError(
                f"Unknown operation kind '{kind}' in NeXus patch {path}, "
                f"expected one of {list(_op_kinds)}"
            )
        ops.append(op(**fields))
    return ops


def open_patched_file(
    path: str | os.PathLike[str],
    patch: Iterable[PatchOp],
    *,
    definitions: Mapping[str, type] | None,
    **kwargs: Any,
) -> PatchedNeXusFile:
    """Open a NeXus file and apply a patch.

    Parameters
    ----------
    path:
        Path of the file to open.
    patch:
        Operations to apply.
    definitions:
        Application definitions to use for the file, see :class:`scippnexus.File`.
    kwargs:
        Forwarded to :class:`h5py.File`.

    Returns
    -------
    :
        The patched file.
        Can be used as a context manager to close the file.
    """
    file = h5py.File(path, 'r', **kwargs)
    try:
        return PatchedNeXusFile(PatchedH5File(file, patch), definitions=definitions)
    except BaseException:
        file.close()
        raise


class PatchedNeXusFile(snx.Group, AbstractContextManager['PatchedNeXusFile']):
    """ScippNexus group for a patched file that closes the file on exit."""

    def __init__(
        self, file: PatchedH5File, *, definitions: Mapping[str, type] | None
    ) -> None:
        super().__init__(file, definitions=definitions)
        self._patched_file = file

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._patched_file.close()


class _PatchedAttrs(Mapping[str, Any]):
    def __init__(
        self,
        base: h5py.AttributeManager,
        new: h5py.AttributeManager | None,
        deleted: set[str],
    ) -> None:
        self._base = base
        self._new = new
        self._deleted = deleted

    def __getitem__(self, name: str) -> Any:
        if self._new is not None and name in self._new:
            return self._new[name]
        if name in self._deleted:
            raise KeyError(name)
        return self._base[name]

    def __iter__(self) -> Iterator[str]:
        new = () if self._new is None else tuple(self._new)
        yield from new
        yield from (k for k in self._base if k not in self._deleted and k not in new)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def get_id(self, name: bytes) -> Any:
        """Used by ScippNexus to determine the encoding of string attributes."""
        if self._new is not None and name.decode() in self._new:
            return self._new.get_id(name)
        return self._base.get_id(name)


class _Node:
    def __init__(
        self, obj: h5py.Group | h5py.Dataset, path: str, file: PatchedH5File
    ) -> None:
        self._obj = obj
        self._path = path
        self._file = file

    @property
    def name(self) -> str:
        return self._path

    @property
    def file(self) -> PatchedH5File:
        return self._file

    @property
    def parent(self) -> _Group:
        return self._file[posixpath.dirname(self._path)]

    @property
    def attrs(self) -> _PatchedAttrs:
        return _PatchedAttrs(
            self._obj.attrs,
            self._file._new_attrs.get(self._path),
            self._file._deleted_attrs.get(self._path, set()),
        )

    def __repr__(self) -> str:
        return f'<{type(self).__name__} "{self._path}">'


class _Dataset(_Node):
    def __getattr__(self, name: str) -> Any:
        # shape, dtype, ndim, read_direct, asstr, ... of the underlying dataset.
        return getattr(self._obj, name)

    def __getitem__(self, index: Any) -> Any:
        return self._obj[index]

    def __len__(self) -> int:
        return len(self._obj)


class _Group(_Node):
    def _child_path(self, name: str) -> str:
        return posixpath.join(self._path, name)

    def keys(self) -> list[str]:
        names = [
            name
            for name in self._obj
            if not self._file._is_removed(self._child_path(name))
        ]
        for path, obj in self._file._mounts.items():
            name = posixpath.basename(path)
            if (
                obj is not None
                and posixpath.dirname(path) == self._path
                and name not in names
            ):
                names.append(name)
        return names

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def items(self) -> list[tuple[str, _Group | _Dataset | None]]:
        # As in h5py, dangling soft links yield None.
        return [(name, self.get(name)) for name in self.keys()]

    def __contains__(self, name: str) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True

    def __getitem__(self, name: str) -> _Group | _Dataset:
        path = posixpath.normpath(posixpath.join(self._path, name))
        if '/' in name:
            obj = self._file._lookup(path)
        else:
            obj = self._file._child(self._obj, path)
        return self._file._wrap(obj, path)

    def get(self, name: str) -> _Group | _Dataset | None:
        try:
            return self[name]
        except KeyError:
            return None


class PatchedH5File(_Group):
    """Read-only view of an HDF5 file with a patch applied.

    Implements the subset of the :mod:`h5py` interface used by ScippNexus.
    Closing this file also closes the underlying file.
    """

    def __init__(self, file: h5py.File, patch: Iterable[PatchOp]) -> None:
        super().__init__(file, '/', self)
        # Objects replacing or added to the original file, by path in the patched
        # file. ``None`` marks removed objects.
        self._mounts: dict[str, h5py.Group | h5py.Dataset | None] = {}
        self._new_attrs: dict[str, h5py.AttributeManager] = {}
        self._deleted_attrs: dict[str, set[str]] = {}
        # New datasets and attributes are stored in an in-memory HDF5 file so they
        # support the full h5py interface, e.g., string encodings and `asstr`.
        self._scratch = h5py.File(
            uuid.uuid4().hex, 'w', driver='core', backing_store=False
        )
        for op in patch:
            self._apply(op)

    @property
    def parent(self) -> PatchedH5File:
        return self

    def close(self) -> None:
        self._scratch.close()
        self._obj.close()

    def __getitem__(self, name: str) -> _Group | _Dataset:
        if posixpath.normpath(name) == '/':
            return self
        return super().__getitem__(name)

    def _wrap(self, obj: h5py.Group | h5py.Dataset, path: str) -> _Group | _Dataset:
        if isinstance(obj, h5py.Dataset):
            return _Dataset(obj, path, self)
        return _Group(obj, path, self)

    def _is_removed(self, path: str) -> bool:
        return path in self._mounts and self._mounts[path] is None

    def _scratch_path(self, kind: str) -> str:
        return f'{kind}/{len(self._scratch.require_group(kind))}'

    def _child(self, parent: h5py.Group, path: str) -> h5py.Group | h5py.Dataset:
        """Return the child of ``parent`` with the given path in the patched file."""
        if self._is_removed(path):
            raise KeyError(f"'{path}' was removed by a patch")
        if (obj := self._mounts.get(path)) is not None:
            return obj
        name = posixpath.basename(path)
        link = parent.get(name, getlink=True)
        if link is None:
            raise KeyError(f"'{path}' not found")
        if isinstance(link, h5py.SoftLink):
            return self._lookup(
                posixpath.normpath(posixpath.join(posixpath.dirname(path), link.path))
            )
        return parent[name]

    def _lookup(self, path: str) -> h5py.Group | h5py.Dataset:
        obj = self._obj
        current = '/'
        for name in path.strip('/').split('/'):
            if not isinstance(obj, h5py.Group):
                raise KeyError(f"'{path}' not found, '{current}' is not a group")
            current = posixpath.join(current, name)
            obj = self._child(obj, current)
        return obj

    def _require(self, path: str) -> _Group | _Dataset:
        try:
            return self[path]
        except KeyError:
            raise KeyError(
                f"Cannot apply patch, '{path}' does not exist in the patched file"
            ) from None

    def _require_parent_group(self, path: str) -> None:
        if not isinstance(self._require(posixpath.dirname(path)), _Group):
            raise ValueError(f"Cannot apply patch, parent of '{path}' is not a group")

    def _drop(self, path: str) -> None:
        """Forget patches of ``path`` and everything below it."""
        for patches in (self._mounts, self._new_attrs, self._deleted_attrs):
            for key in [k for k in patches if _is_within(k, path)]:
                del patches[key]

    def _apply(self, op: PatchOp) -> None:
        match _normalize_paths(op):
            case SetAttrs(path=path, attrs=attrs):
                self._require(path)
                if path not in self._new_attrs:
                    holder = self._scratch.create_group(self._scratch_path('attrs'))
                    self._new_attrs[path] = holder.attrs
                self._new_attrs[path].update(attrs)
                self._deleted_attrs.get(path, set()).difference_update(attrs)
            case DeleteAttrs(path=path, names=names):
                if missing := set(names) - set(self._require(path).attrs):
                    raise KeyError(
                        f"Cannot apply patch, attributes {missing} "
                        f"do not exist at '{path}'"
                    )
                new = self._new_attrs.get(path, {})
                for name in names:
                    if name in new:
                        del new[name]
                self._deleted_attrs.setdefault(path, set()).update(names)
            case Remove(path=path):
                if path == '/':
                    raise ValueError("Cannot remove the root group")
                self._require(path)
                self._drop(path)
                self._mounts[path] = None
            case Move(source=source, target=target):
                obj = self._require(source)._obj
                if _is_within(target, source):
                    raise ValueError(f"Cannot move '{source}' into itself")
                if target in self:
                    raise ValueError(f"Cannot move '{source}', '{target}' exists")
                self._require_parent_group(target)
                self._drop(target)
                for patches in (self._mounts, self._new_attrs, self._deleted_attrs):
                    for key in [k for k in patches if _is_within(k, source)]:
                        patches[target + key[len(source) :]] = patches.pop(key)
                self._mounts[target] = obj
                self._mounts[source] = None
            case SetDataset(path=path, value=value):
                self._require_parent_group(path)
                if isinstance(self.get(path), _Group):
                    raise ValueError(f"Cannot replace group '{path}' by a dataset")
                self._drop(path)
                self._mounts[path] = self._scratch.create_dataset(
                    self._scratch_path('datasets'), data=value
                )
            case _:
                raise TypeError(f"Unknown patch operation {op!r}")


def _normalize_paths(op: PatchOp) -> PatchOp:
    paths = {
        field: getattr(op, field)
        for field in ('path', 'source', 'target')
        if hasattr(op, field)
    }
    for path in paths.values():
        if not path.startswith('/'):
            raise ValueError(f"Patch paths must be absolute, got '{path}'")
    # Note that normpath preserves a leading '//'.
    return replace(
        op, **{k: posixpath.normpath('/' + v.lstrip('/')) for k, v in paths.items()}
    )


def _is_within(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip('/') + '/')
