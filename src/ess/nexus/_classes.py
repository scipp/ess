# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from dataclasses import fields
import scippnexus as snx


def _load_dataclass(group: snx.NXobject, schema: type, arg_dict):
    loaded = {}
    for field in fields(schema):
        key = field.name
        cls = field.type
        field_kwargs = arg_dict.get(key, {})
        loaded[key] = cls.from_nexus(group, **field_kwargs)
    return schema(**loaded)


class InstrumentMixin:

    @classmethod
    def from_nexus(cls, group, /, **kwargs):
        return _load_dataclass(group.instrument, cls, kwargs)


class EntryMixin:

    @classmethod
    def from_nexus(cls, group, /, **kwargs):
        if isinstance(group, snx.NXentry):
            return _load_dataclass(group, cls, kwargs)
        if isinstance(group, snx.NXroot):
            return cls.from_nexus(group.entry, **kwargs)
        with snx.File(group) as f:
            return cls.from_nexus(f, **kwargs)
