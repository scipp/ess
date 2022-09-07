# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from dataclasses import fields
import scippnexus as snx

# Use dataclasses since dict keys such as `source` or `detectors` might clash with
# other field names

# Nesting is a bad idea for user facing
# Must write loaders in a way that upstream ading new groups/class does not break anything
# -> explicit list of everything we load?
# How can we ensure code<->file compatibility over many years of changes?


def load(group: snx.NXobject, schema: type):
    loaded = {}
    for field in fields(schema):
        key = field.name
        cls = field.type
        loaded[key] = cls.from_nexus(group)
    return schema(**loaded)


class InstrumentMixin:

    @classmethod
    def from_nexus(cls, group):
        return load(group.instrument, cls)


class EntryMixin:

    @classmethod
    def from_nexus(cls, group):
        if isinstance(group, snx.NXentry):
            return load(group, cls)
        if isinstance(group, snx.NXroot):
            return cls.from_nexus(group.entry)
        with snx.File(group) as f:
            return cls.from_nexus(f)
