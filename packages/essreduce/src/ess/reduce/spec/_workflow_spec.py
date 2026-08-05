# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Workflow specifications: implementation-independent workflow metadata.

A :class:`WorkflowSpec` describes a workflow's user-facing interface — identity,
display metadata, parameters, and outputs — without reference to how or where
the workflow is computed. User interfaces (widgets, dashboards, command-line
tools) are generated from the spec alone; the binding from a spec to an
executor is a separate, parallel mechanism deliberately not defined here.

Two forms exist, related by a one-way projection:

* :class:`WorkflowSpec` is the in-process form. It holds the params *model
  class*, so consumers in the same process get full pydantic validation,
  including cross-field validators.
* :class:`SerializedWorkflowSpec` is the plain-data form produced by
  :meth:`WorkflowSpec.serialize`, with params as JSON Schema. It is what a
  service announces to remote consumers, which can render forms and validate
  optimistically against the schema. There is intentionally no inverse:
  validators do not round-trip through JSON Schema, and authoritative
  validation always happens in the process owning the model class.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NoParams(BaseModel):
    """
    Params model for workflows that take no configuration.

    Workflows always have a params model, so consumers never branch on its
    absence; "takes no parameters" is expressed as a model with no fields.
    Extra fields are rejected so that sending params to such a workflow is an
    error rather than silently ignored.
    """

    model_config = ConfigDict(extra='forbid')


class ArraySpec(BaseModel, frozen=True):
    """
    Structural description of an array-valued workflow output.

    Describes shape-independent structure — dimensions, unit, and coordinate
    units — sufficient for a consumer to prepare for the data (e.g., select a
    plotter) before any has been computed. A scalar value with a unit is the
    0-d case: ``ArraySpec(dims=(), unit='counts')``.
    """

    dims: tuple[str, ...] = Field(description="Dimension names, outermost first.")
    unit: str | None = Field(
        default=None, description="Unit of the array values, if any."
    )
    coords: dict[str, str | None] = Field(
        default_factory=dict,
        description="Coordinate names mapped to their units (None for unitless).",
    )


class OutputSpec(BaseModel, frozen=True):
    """Description of a single named workflow output."""

    title: str = Field(min_length=1, description="Display title of the output.")
    description: str = Field(default='', description="Description of the output.")
    array: ArraySpec | None = Field(
        default=None,
        description=(
            "Structural description of the output data, if array-valued and known."
        ),
    )


def _default_outputs() -> dict[str, OutputSpec]:
    return {'result': OutputSpec(title='Result', description='Workflow output.')}


class _SpecFields(BaseModel, frozen=True):
    """Metadata fields shared by both forms of the workflow spec."""

    name: str = Field(
        min_length=1,
        description=(
            "Machine-readable workflow identifier. Unique within the context "
            "that enumerates the spec; global uniqueness is the enumerator's "
            "responsibility, not the spec's."
        ),
    )
    version: int = Field(
        ge=1,
        description=(
            "Version of the workflow interface. Increment on any change a "
            "consumer could observe: params model, outputs, or semantics."
        ),
    )
    title: str = Field(min_length=1, description="Display title of the workflow.")
    description: str = Field(
        min_length=1, description="Description of what the workflow computes."
    )


class WorkflowSpec(_SpecFields, frozen=True):
    """
    Implementation-independent specification of a workflow's user interface.

    Holds identity and display metadata, the pydantic model class defining the
    workflow's parameters, and descriptions of its outputs. Contains no
    factory, no executor, and no reference to any workflow implementation;
    pairing a spec with something that computes it is a separate mechanism.
    """

    params: type[BaseModel] = Field(
        default=NoParams,
        description=(
            "Pydantic model class defining the workflow parameters. Defaults "
            "to :class:`NoParams` for workflows that take no configuration."
        ),
    )
    outputs: dict[str, OutputSpec] = Field(
        default_factory=_default_outputs,
        description=(
            "Named outputs the workflow produces. Order is meaningful: "
            "consumers present outputs in this order and may auto-select the "
            "first, so put the primary output first."
        ),
    )

    def serialize(self) -> SerializedWorkflowSpec:
        """
        Project to the plain-data form with params as JSON Schema.

        The projection is one-way: pydantic validators do not survive it, so
        a consumer of the serialized form can validate only optimistically.
        Authoritative validation happens where the model class lives.
        """
        return SerializedWorkflowSpec(
            name=self.name,
            version=self.version,
            title=self.title,
            description=self.description,
            params_schema=self.params.model_json_schema(),
            outputs=self.outputs,
        )


class SerializedWorkflowSpec(_SpecFields, frozen=True):
    """
    Plain-data form of a workflow spec, safe to send across process boundaries.

    Produced by :meth:`WorkflowSpec.serialize`; round-trips through JSON. Params
    are represented as JSON Schema, sufficient for form generation and
    optimistic validation but not for authoritative validation — that remains
    with the process owning the params model class.
    """

    params_schema: dict[str, Any] = Field(
        description="JSON Schema of the workflow's params model."
    )
    outputs: dict[str, OutputSpec] = Field(
        description="Named outputs the workflow produces, in display order."
    )
