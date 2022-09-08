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


def find_nxclass(key) -> Callable:

    def func(group: snx.NXobject) -> dict:
        return group[key]

    return func


def default_load(targets, skip_errors=False):
    if isinstance(targets, dict):
        return _load_items(targets, skip_errors=skip_errors)
    else:
        return _load_single(targets, skip_errors=skip_errors)


def select_events_and_load(targets, pulse_min=None, pulse_max=None, **kwargs):
    if pulse_min is not None or pulse_max is not None:
        targets = {
            k: v.select_events['pulse', pulse_min:pulse_max]
            for k, v in targets.items()
        }
    return default_load(targets, **kwargs)


def make_multi_field(name, key, load: Callable = _load_items):

    def from_nexus(cls, group, **kwargs):
        return cls(load(group[key], **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


def make_field(name, key, load: Callable = _load_single):

    def from_nexus(cls, group, **kwargs):
        return cls(load(group.__getattr__(key.__name__[2:]), **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


Fields = make_multi_field("Fields", [snx.Field, snx.NXlog])
Detectors = make_multi_field("Detectors", snx.NXdetector, select_events_and_load)
Monitors = make_multi_field("Monitors", snx.NXmonitor)
Sample = make_field("Sample", snx.NXsample)
