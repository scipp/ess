# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from os import PathLike
from typing import Dict, Union, Tuple, Callable
import scipp as sc
import scippnexus as snx
from functools import partial


def _get_entry(group: snx.NXobject) -> snx.NXentry:
    return group if group.nx_class == snx.NXentry else group.entry


def _load_items(items: Dict[str, snx.NXobject],
                skip_errors: bool = False) -> Dict[str, sc.DataArray]:
    result = {}
    for k, v in items.items():
        try:
            result[k] = v[()]
        except IndexError as e:
            if not skip_errors:
                raise e from None
    return result


def _load(group: snx.NXobject,
          nxclass: Union[type, Tuple[type]],
          skip_errors: bool = False) -> Dict[str, sc.DataArray]:
    return _load_items(group[nxclass], skip_errors=skip_errors)


def _load_sections(group: snx.NXobject, nxclasses, skip_errors: bool = False):
    data = {}
    for key, nxclass in nxclasses.items():
        data[key] = _load(group, nxclass, skip_errors=skip_errors)
    return data


def load_detectors(filename: Union[str, PathLike]) -> dict:
    with snx.File(filename) as f:
        return _load(f.entry.instrument, snx.NXdetector, skip_errors=True)


def load_monitors(filename: Union[str, PathLike]) -> dict:
    with snx.File(filename) as f:
        return _load(f.entry, snx.NXmonitor, skip_errors=True)


def load_metadata(filename: Union[str, PathLike]) -> dict:
    """Load everything except detectors and monitors, which could be large."""
    with snx.File(filename) as f:
        entry = _load_entry(f.entry)


def make_field(name, from_nexus):
    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


def make_loader(nxclass: Union[type, Tuple[type]],
                skip_errors: bool = False) -> Callable:

    def func(cls, group: snx.NXobject) -> Dict[str, sc.DataArray]:
        return cls(_load_items(group[nxclass], skip_errors=skip_errors))

    return func


Fields = make_field("Fields", make_loader(nxclass=(snx.Field, snx.NXlog)))
Detectors = make_field("Detectors", make_loader(nxclass=snx.NXdetector,
                                                skip_errors=True))
Monitors = make_field("Monitors", make_loader(nxclass=snx.NXmonitor))
Sample = make_field("Sample", lambda cls, group: cls(group.sample[()]))
