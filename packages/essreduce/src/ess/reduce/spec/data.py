# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Data fields: parameters and outputs that hold data rather than literals.

A data field is a parameter or output field whose value is a file or an array.
Its type is a union of two forms, and a :class:`DataField` annotation says
which kind of data the field holds:

* At submission the field holds a :data:`Ref`, a reference to data that exists
  elsewhere: an output of an earlier run, or a dataset the framework did not
  compute. A remote consumer sees only this form in the JSON Schema.
* Inside the workflow the field holds the materialized value: a local path for
  files, a scipp object for arrays. Which form a framework produces for each
  :class:`Kind` is its business; the spec only declares the kind.

Arrays are constrained by :class:`ArraySpec`, on both sides, so an output field
of one spec can feed a parameter field of another when their structure matches.
Collections of data fields, ``list[...]`` and ``dict[str, ...]`` of one declared
type, are allowed and a reference may name one element of a collection output.
This module imports no scipp; the structural check of a scipp object against its
:class:`ArraySpec` lives in :mod:`ess.reduce.spec.conversions`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic_core import PydanticOmit, core_schema


class Kind(StrEnum):
    """What a data field holds, and thus how a framework materializes it."""

    NEXUS = 'nexus'
    """A raw NeXus file, materialized as a local path."""
    OPAQUE = 'opaque'
    """A file of a format the framework does not read, materialized as a path."""
    ARRAY = 'array'
    """A scipp object."""


class ArraySpec(BaseModel, frozen=True):
    """
    Structural description of an array: dimensions, units, and whether binned.

    Shape-independent, so a consumer can prepare for the data, e.g., select a
    plotter, before any has been computed. A scalar with a unit is the 0-d case,
    ``ArraySpec(dims=(), unit='counts')``.
    """

    dims: tuple[str, ...] = Field(description="Dimension names, outermost first.")
    unit: str | None = Field(
        default=None,
        description=(
            "Unit of the array values. None means no unit at all, as for "
            "strings or datetimes; a dimensionless quantity is 'dimensionless'."
        ),
    )
    coords: dict[str, str | None] = Field(
        default_factory=dict,
        description="Coordinate names mapped to their units, as for ``unit``.",
    )
    binned: bool = Field(
        default=False,
        description="Event data in bins; never plotted directly.",
    )


class OutputRef(BaseModel, frozen=True):
    """Output ``output`` of record ``record``, or one element ``key`` of it."""

    record: str = Field(min_length=1)
    output: str = Field(min_length=1)
    key: str | None = None

    def __str__(self) -> str:
        key = f'[{self.key}]' if self.key is not None else ''
        return f'{self.record}.{self.output}{key}'


class DatasetRef(BaseModel, frozen=True):
    """
    Data the framework did not compute, named by an identity the framework owns.

    What the identity means, a catalogue PID or a local file's identity, is not
    the spec's concern; a dataset satisfies a data field of any kind.
    """

    dataset: str = Field(min_length=1)

    def __str__(self) -> str:
        return self.dataset


Ref = OutputRef | DatasetRef
"""A reference: the value a data field holds at submission."""


@dataclass(frozen=True)
class DataField:
    """
    Field annotation marking a data field, with its kind and array structure.

    The union type on the field admits both the reference and the materialized
    form; this annotation says what the materialized form must be. Serialized
    into JSON Schema under the ``dataField`` key so that remote consumers can
    tell data fields from literals and know their structure.
    """

    kind: Kind
    array: ArraySpec | None = None

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Any) -> Any:
        schema = handler(core_schema)
        schema['dataField'] = {'kind': self.kind.value}
        if self.array is not None:
            schema['dataField']['array'] = self.array.model_dump(mode='json')
        return schema


class Materialized:
    """
    The in-process form of an array field: anything that is not plain data.

    Admits a scipp object without naming scipp, which the spec layer must not
    import. Omitted from JSON Schema, where only the reference form exists.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            if isinstance(value, dict | list | str | int | float | bool | type(None)):
                raise ValueError('expected a reference or an in-process data object')
            return value

        return core_schema.no_info_plain_validator_function(check)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        raise PydanticOmit


NexusFile = Annotated[Ref | Path, DataField(kind=Kind.NEXUS)]
"""A raw NeXus file; the workflow receives a local path."""
OpaqueFile = Annotated[Ref | Path | bytes, DataField(kind=Kind.OPAQUE)]
"""A file the framework does not read; a workflow returns one as bytes."""


def Array(spec: ArraySpec | None = None) -> Any:
    """Type of a field holding a scipp object, constrained by ``spec`` if given."""
    return Annotated[Ref | Materialized, DataField(kind=Kind.ARRAY, array=spec)]


def _members(annotation: Any) -> Iterator[Any]:
    """The annotation and, through unions, optionals, and collections, its parts."""
    yield annotation
    origin = get_origin(annotation)
    if origin is Annotated:
        yield from _members(get_args(annotation)[0])
    elif origin in (Union, UnionType):
        for arg in get_args(annotation):
            yield from _members(arg)
    elif origin is list:
        yield from _members(get_args(annotation)[0])
    elif origin is dict:
        yield from _members(get_args(annotation)[1])


def _data_field(annotation: Any) -> DataField | None:
    for member in _members(annotation):
        if get_origin(member) is Annotated:
            for metadata in get_args(member)[1:]:
                if isinstance(metadata, DataField):
                    return metadata
    return None


def data_fields(model: type[BaseModel]) -> dict[str, DataField]:
    """
    Data fields of a params or outputs model, by name.

    Optional fields and collections count; every element of a collection shares
    the annotation.
    """
    fields = {}
    for name, field in model.model_fields.items():
        found = next((m for m in field.metadata if isinstance(m, DataField)), None)
        if found is None:
            found = _data_field(field.annotation)
        if found is not None:
            fields[name] = found
    return fields


def ref_fields(model: type[BaseModel]) -> set[str]:
    """Fields that may hold a reference: data fields and literal-or-reference unions."""
    data = data_fields(model)
    return {
        name
        for name, field in model.model_fields.items()
        if name in data
        or any(m in (OutputRef, DatasetRef) for m in _members(field.annotation))
    }


_OUTPUT_REF_KEYS = frozenset(OutputRef.model_fields)


def as_ref(value: Any) -> Ref | None:
    """The reference a plain value denotes, if it is one."""
    if isinstance(value, OutputRef | DatasetRef):
        return value
    if isinstance(value, dict):
        keys = set(value)
        if {'record', 'output'} <= keys <= _OUTPUT_REF_KEYS:
            return OutputRef.model_validate(value)
        if keys == {'dataset'}:
            return DatasetRef.model_validate(value)
    return None


def walk_refs(value: Any, path: str = '') -> Iterator[tuple[str, Ref]]:
    """Yield every reference in a plain (JSON-shaped) value, with its path."""
    if (ref := as_ref(value)) is not None:
        yield path, ref
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from walk_refs(v, f'{path}.{k}' if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk_refs(v, f'{path}[{i}]')
