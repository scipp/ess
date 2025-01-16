# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc
import scippnexus as snx

from .types import ReducedReference, ReferenceFilePath


def load_nx(group: snx.Group | str, *paths: str):
    if isinstance(group, str):
        with snx.File(group) as group:
            yield from load_nx(group, *paths)
        return
    for path in paths:
        g = group
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


def save_reference(pl: sciline.Pipeline, fname: str):
    pl.compute(ReducedReference).save_hdf5(fname)
    return fname


def load_reference(fname: ReferenceFilePath) -> ReducedReference:
    return sc.io.hdf5.load_hdf5(fname)
