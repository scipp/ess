from typing import Optional, Type, Union

import sciline
import scipp as sc
import scippnexus as snx

from .types import IdealReferenceIntensity, ReferenceFilePath


def load_nx(group: Union[snx.Group, str], *paths: str):
    if isinstance(group, str):
        with snx.File(group) as group:
            yield from load_nx(group, *paths)
        return
    for path in paths:
        g = group
        for p in path.strip('/').split('/'):
            g = (
                _unique_child_group(g, getattr(snx, p), None)
                if p.startswith('NX')
                else g[p]
            )
        yield g[...]


# Remove when this function is exposed in essreduce
def _unique_child_group(
    group: snx.Group, nx_class: Type[snx.NXobject], name: Optional[str]
) -> snx.Group:
    if name is not None:
        child = group[name]
        if isinstance(child, snx.Field):
            raise ValueError(
                f"Expected a NeXus group as item '{name}' but got a field."
            )
        if child.nx_class != nx_class:
            raise ValueError(
                f"The NeXus group '{name}' was expected to be a "
                f'{nx_class} but is a {child.nx_class}.'
            )
        return child

    children = group[nx_class]
    if len(children) != 1:
        raise ValueError(f'Expected exactly one {nx_class} group, got {len(children)}')
    return next(iter(children.values()))  # type: ignore[return-value]


def save_reference(pl: sciline.Pipeline, fname: str):
    pl.compute(IdealReferenceIntensity).save_hdf5(fname)
    return fname


def load_reference(fname: ReferenceFilePath) -> IdealReferenceIntensity:
    return sc.io.hdf5.load_hdf5(fname)
