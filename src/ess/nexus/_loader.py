# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from os import PathLike
from typing import Dict, Union, Tuple, Callable, List, Any, Optional
from dataclasses import dataclass, fields
import scipp as sc
import scippnexus as snx


def _load_dataclass(group: snx.NXobject, schema: type, arg_dict: dict):
    loaded = {}
    for field in fields(schema):
        key = field.name
        cls = field.type
        field_kwargs = arg_dict.get(key, {})
        loaded[key] = cls.from_nexus(group, **field_kwargs)
    return schema(**loaded)


class InstrumentMixin:

    @classmethod
    def from_nexus(cls, group: snx.NXobject, /, **kwargs: Any):
        return _load_dataclass(group.instrument, cls, kwargs)


class EntryMixin:

    @classmethod
    def from_nexus(cls, group: Union[snx.NXobject, PathLike], /, **kwargs: Any):
        if isinstance(group, snx.NXentry):
            return _load_dataclass(group, cls, kwargs)
        if isinstance(group, snx.NXroot):
            return cls.from_nexus(group.entry, **kwargs)
        with snx.File(group) as f:
            return cls.from_nexus(f, **kwargs)


# TODO make decorator factory, list exception types
def _load_single(group,
                 index=(),
                 skip_errors=False) -> Optional[Union[sc.DataArray, dict]]:
    try:
        return group[index]
    except IndexError as e:
        if not skip_errors:
            raise e from None


def _load_multi(load: Callable, items: Dict[str, snx.NXobject], /,
                **kwargs) -> Dict[str, sc.DataArray]:
    return {
        key: loaded
        for key in items if (loaded := load(items[key], **kwargs)) is not None
    }


def _select_events_and_load(detector, pulse_min=None, pulse_max=None, **kwargs):
    detector = detector.select_events['pulse', pulse_min:pulse_max]
    return _load_single(detector, **kwargs)


def make_section(name: str,
                 key: Union[type, List[type]],
                 load: Callable = _load_single) -> type:

    def from_nexus(cls, group: snx.NXobject, **kwargs):
        return cls(_load_multi(load, group[key], **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


def make_field(name: str,
               key: Union[type, List[type]],
               load: Callable = _load_single) -> type:

    def from_nexus(cls, group: snx.NXobject, **kwargs):
        return cls(load(group.__getattr__(key.__name__[2:]), **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


Fields = make_section("Fields", [snx.Field, snx.NXlog])
Detectors = make_section("Detectors", snx.NXdetector, _select_events_and_load)
Monitors = make_section("Monitors", snx.NXmonitor)
Sample = make_field("Sample", snx.NXsample)
Source = make_field("Source", snx.NXsource)


@dataclass
class BasicInstrument(InstrumentMixin):
    fields: Fields
    detectors: Detectors


@dataclass
class BasicEntry(EntryMixin):
    fields: Fields
    instrument: BasicInstrument
    monitors: Monitors
    sample: Sample
