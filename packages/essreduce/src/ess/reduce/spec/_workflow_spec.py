# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Workflow specifications: implementation-independent workflow metadata.

A :class:`WorkflowSpec` describes a workflow's user-facing interface — identity,
display metadata, parameters, and outputs — without reference to how or where
the workflow is computed. User interfaces (widgets, dashboards, command-line
tools) are generated from the spec alone; the binding from a spec to an
executor is a separate, parallel mechanism deliberately not defined here.

Parameters and outputs are both pydantic model classes over one vocabulary
(:mod:`ess.reduce.spec.parameters` for literals, :mod:`ess.reduce.spec.data`
for files and arrays), so an output field of one workflow can feed a parameter
field of another when their types match.

Two forms exist, related by a one-way projection:

* :class:`WorkflowSpec` is the in-process form. It holds the params and outputs
  model *classes*, so consumers in the same process get full pydantic
  validation, including cross-field validators.
* :class:`SerializedWorkflowSpec` is the plain-data form produced by
  :meth:`WorkflowSpec.serialize`, with both models as JSON Schema. It is what a
  service announces to remote consumers, which can render forms and validate
  optimistically against the schema. There is intentionally no inverse:
  validators do not round-trip through JSON Schema, and authoritative
  validation always happens in the process owning the model classes.
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
    code_revision: str | None = Field(
        default=None,
        description=(
            "Git commit or package version of the workflow code this spec "
            "describes, so that a record made from a development branch is "
            "honest about what ran. Provenance, not identity: the interface "
            "version is ``version``."
        ),
    )


class WorkflowSpec(_SpecFields, frozen=True):
    """
    Implementation-independent specification of a workflow's user interface.

    Holds identity and display metadata and the pydantic model classes defining
    the workflow's parameters and outputs. Contains no factory, no executor, and
    no reference to any workflow implementation; pairing a spec with something
    that computes it is a separate mechanism. The module defining a spec must
    therefore be importable without importing the workflow code, so that a
    service can load and validate every spec it knows without that code.
    """

    params: type[BaseModel] = Field(
        default=NoParams,
        description=(
            "Pydantic model class defining the workflow parameters. Defaults "
            "to :class:`NoParams` for workflows that take no configuration."
        ),
    )
    outputs: type[BaseModel] = Field(
        description=(
            "Pydantic model class defining the workflow outputs. Field title "
            "and description are the display metadata; array and file outputs "
            "are data fields (see :mod:`ess.reduce.spec.data`); a field may be "
            "optional when the workflow does not always produce it. Order is "
            "meaningful: consumers present outputs in this order and may "
            "auto-select the first, so put the primary output first."
        ),
    )

    def serialize(self) -> SerializedWorkflowSpec:
        """
        Project to the plain-data form with params and outputs as JSON Schema.

        The projection is one-way: pydantic validators do not survive it, so
        a consumer of the serialized form can validate only optimistically.
        Authoritative validation happens where the model classes live.
        """
        return SerializedWorkflowSpec(
            name=self.name,
            version=self.version,
            title=self.title,
            description=self.description,
            code_revision=self.code_revision,
            params_schema=self.params.model_json_schema(),
            outputs_schema=self.outputs.model_json_schema(),
        )


class SerializedWorkflowSpec(_SpecFields, frozen=True):
    """
    Plain-data form of a workflow spec, safe to send across process boundaries.

    Produced by :meth:`WorkflowSpec.serialize`; round-trips through JSON. Params
    and outputs are represented as JSON Schema, sufficient for form generation
    and optimistic validation but not for authoritative validation — that
    remains with the process owning the model classes. Data fields carry a
    ``dataField`` key with their kind and array structure.
    """

    params_schema: dict[str, Any] = Field(
        description="JSON Schema of the workflow's params model."
    )
    outputs_schema: dict[str, Any] = Field(
        description="JSON Schema of the workflow's outputs model."
    )
