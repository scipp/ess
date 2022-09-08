# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from os import PathLike
from typing import Dict, Union, Tuple, Callable
import scipp as sc
import scippnexus as snx


# TODO make decorator factory, list exception types
def _load_single(group, index=(), skip_errors=False):
    try:
        return group[index]
    except IndexError as e:
        if not skip_errors:
            raise e from None


def _load_multi(load: Callable, items: Dict[str, snx.NXobject], /,
                **kwargs) -> Dict[str, sc.DataArray]:
    result = {}
    for k, v in items.items():
        if (loaded := load(v, **kwargs)) is not None:
            result[k] = loaded
    return result


def select_events_and_load(detector, pulse_min=None, pulse_max=None, **kwargs):
    detector = detector.select_events['pulse', pulse_min:pulse_max]
    return _load_single(detector, **kwargs)


def make_multi_field(name, key, load: Callable = _load_single):

    def from_nexus(cls, group, **kwargs):
        return cls(_load_multi(load, group[key], **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


def make_field(name, key, load: Callable = _load_single):

    def from_nexus(cls, group, **kwargs):
        return cls(load(group.__getattr__(key.__name__[2:]), **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


Fields = make_multi_field("Fields", [snx.Field, snx.NXlog])
Detectors = make_multi_field("Detectors", snx.NXdetector, select_events_and_load)
Monitors = make_multi_field("Monitors", snx.NXmonitor)
Sample = make_field("Sample", snx.NXsample)
