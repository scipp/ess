# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterator
from pathlib import Path

import h5py
import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus import open_nexus_file

from .types import ReducedReference, ReferenceFilePath


def load_nx(
    group: snx.Group | str | Path, *paths: str
) -> Iterator[sc.DataGroup | sc.DataArray | None]:
    """Yield NeXus groups or arrays at the provided paths.

    Parameters
    ----------
    group:
        NeXus group or file path.
    *paths:
        One or more NeXus paths to load.

    Returns
    -------
    :
        Iterator of loaded NeXus objects (group or array), or ``None`` if missing.
    """
    with open_nexus_file(group) as g:
        for path in paths:
            for p in path.strip('/').split('/'):
                g = (
                    _unique_child_group(g, getattr(snx, p))
                    if p.startswith('NX')
                    else g.get(p)
                )
            yield g[...] if g is not None else None


# Remove when this function is exposed in essreduce
def _unique_child_group(
    group: snx.Group,
    nx_class: type[snx.NXobject],
) -> snx.Group:
    children = group[nx_class]
    if len(children) == 0:
        return None
    elif len(children) != 1:
        raise ValueError(f'Expected exactly one {nx_class} group, got {len(children)}')
    return next(iter(children.values()))  # type: ignore[return-value]


def load_h5(
    group: h5py.Group | str, *paths: str
) -> Iterator[h5py.Group | h5py.Dataset | None]:
    """Yield HDF5 groups or datasets at the provided paths.

    Parameters
    ----------
    group:
        HDF5 group or file path.
    *paths:
        One or more HDF5 paths to load.

    Returns
    -------
    :
        Iterator of loaded HDF5 objects (group or dataset), or ``None`` if missing.
    """
    if isinstance(group, str):
        with h5py.File(group) as group:
            yield from load_h5(group, *paths)
        return
    for path in paths:
        g = group
        for p in path.strip('/').split('/'):
            g = _unique_child_group_h5(g, p) if p.startswith('NX') else g.get(p)
        yield g


def _unique_child_group_h5(
    group: h5py.Group,
    nx_class: str,
) -> h5py.Group | None:
    out = None
    for v in group.values():
        if v.attrs.get("NX_class") == nx_class.encode():
            if out is None:
                out = v
            else:
                raise ValueError(
                    f'Expected exactly one {nx_class} group, but found more'
                )
    return out


def save_reference(pl: sciline.Pipeline, fname: str) -> str:
    """Compute and save a reduced reference to HDF5.

    Parameters
    ----------
    pl:
        Sciline pipeline that can compute ``ReducedReference``.
    fname:
        Output file path.

    Returns
    -------
    :
        The output file path.
    """
    pl.compute(ReducedReference).save_hdf5(fname)
    return fname


def load_reference(fname: ReferenceFilePath) -> ReducedReference:
    """Load a reduced reference from HDF5.

    Returns
    -------
    :
        The reduced reference data.
    """
    return sc.io.hdf5.load_hdf5(fname)
