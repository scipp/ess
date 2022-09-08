# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from os import PathLike
from typing import Dict, Union, Callable, List, Any, Optional, TypeVar, Type
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


T1 = TypeVar('T1', bound='InstrumentMixin')


class InstrumentMixin:
    """Mixin providing the ``from_nexus`` classmethod for dataclasses representing
    NXinstrument.
    """

    @classmethod
    def from_nexus(cls: Type[T1], group: snx.NXobject, /, **kwargs: Any) -> T1:
        """Load data from the NXinstrument group."""
        return _load_dataclass(group.instrument, cls, kwargs)


T2 = TypeVar('T2', bound='EntryMixin')


class EntryMixin:
    """Mixin providing the ``from_nexus`` classmethod for dataclasses representing
    NXentry.
    """

    @classmethod
    def from_nexus(cls: Type[T2], group: Union[snx.NXobject, PathLike], /,
                   **kwargs: Any) -> T2:
        """Load data from the NXentry group."""
        if isinstance(group, snx.NXentry):
            return _load_dataclass(group, cls, kwargs)
        if isinstance(group, snx.NXroot):
            return cls.from_nexus(group.entry, **kwargs)
        with snx.File(group) as f:
            return cls.from_nexus(f, **kwargs)


def _load_single(group,
                 index=(),
                 skip_errors=False) -> Optional[Union[sc.DataArray, dict]]:
    # The list of caugth exceptions is copied from scippneutron.load_nexus. We will
    # likely wanto to add better customization options.
    try:
        return group[index]
    except (snx.NexusStructureError, KeyError, sc.DTypeError, ValueError,
            IndexError) as e:
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
    """
    Make a loader component for a dict-like section containing all children with
    NX_class matching the given key.

    Parameters
    ----------
    key:
        Key as accepted by :py:meth:`scippnexus.NXobject.__getitem__` for selecting a
        subset of children based on the class. Examples: ``scippnexus.NXdetector``,
        ``[scippnexus.Field, scippnexus.NXlog]``.
    load:
        Callable to perform the loading of a single group. This will be called once for
        each child matching ``key``.
    """

    def from_nexus(cls, group: snx.NXobject, **kwargs):
        return cls(_load_multi(load, group[key], **kwargs))

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


def make_leaf(name: str, key: Union[str, type], load: Callable = _load_single) -> type:
    """
    Make a loader component for a "leaf" containing a single child with NX_class
    matching the given key.

    Parameters
    ----------
    key:
        Child name or child's NX_class (subclass of :py:class:`scippnexus.NXobject`).
    load:
        Callable to perform the loading the child group.
    """

    def from_nexus(cls, group: snx.NXobject, **kwargs):
        c = group[key] if isinstance(key, str) else group.__getattr__(key.__name__[2:])
        data = load(c, **kwargs)
        return cls(data) if isinstance(data, dict) else data

    return type(name, (dict, ), dict(from_nexus=classmethod(from_nexus)))


Fields = make_section("Fields", [snx.Field, snx.NXlog])
Choppers = make_section("Choppers", [snx.NXdisk_chopper, snx.NXfermi_chopper])
Detectors = make_section("Detectors", snx.NXdetector, _select_events_and_load)
Monitors = make_section("Monitors", snx.NXmonitor)
Sample = make_leaf("Sample", snx.NXsample)
Source = make_leaf("Source", snx.NXsource)


@dataclass
class BasicInstrument(InstrumentMixin):
    """
    Basic loader component for NXinstrument.

    Produces a basic subset of NXinstrument, loading NXdetector children and fields
    (including NXlog). Subclass this in a new dataclass to add sections for more NeXus
    classes.
    """
    fields: Fields
    detectors: Detectors


@dataclass
class BasicEntry(EntryMixin):
    """
    Basic loader component for NXentry.

    Produces a basic subset of NXentry, loading NXinstrument, NXmonitor, and NXsample
    children as well as fields (including NXlog). Subclass this in a new dataclass to
    add sections for more NeXus classes.
    """
    fields: Fields
    instrument: BasicInstrument
    monitors: Monitors
    sample: Sample
