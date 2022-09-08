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


def _load_single(group, index=(), skip_errors=False):
    try:
        return group[index]
    except IndexError as e:
        if not skip_errors:
            raise e from None


def _load_items(items: Dict[str, snx.NXobject],
                skip_errors: bool = False) -> Dict[str, sc.DataArray]:
    result = {}
    for k, v in items.items():
        if (loaded := _load_single(v, skip_errors=skip_errors)) is not None:
            result[k] = loaded
    return result


def select_nxclass(key) -> Callable:

    def func(group: snx.NXobject) -> dict:
        return group[key]

    return func


def default_load(targets, skip_errors=False):
    if isinstance(targets, dict):
        return _load_items(targets, skip_errors=skip_errors)
    else:
        return _load_single(targets, skip_errors=skip_errors)


def select_events_and_load(targets, pulse_max=None, **kwargs):
    if pulse_max is not None:
        targets = {k: v.select_events['pulse', :pulse_max] for k, v in targets.items()}
    return default_load(targets, **kwargs)


def make_field(name, select: Callable, load: Callable = default_load):

    def from_nexus(cls, group, **kwargs):
        return cls(load(select(group), **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


Fields = make_field("Fields", select=select_nxclass((snx.Field, snx.NXlog)))
Detectors = make_field("Detectors", select_nxclass(snx.NXdetector),
                       select_events_and_load)
Monitors = make_field("Monitors", select_nxclass(snx.NXmonitor))
Sample = make_field("Sample", select=lambda group: group.sample)
