# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Implementation-independent workflow specifications for UI generation.

See :mod:`ess.reduce.spec._workflow_spec` for the design, :mod:`~.data` for
data fields, and ADR 0001 (docs/developer/adr) for the rationale.
"""

from ._workflow_spec import NoParams, SerializedWorkflowSpec, WorkflowSpec
from .data import (
    Array,
    ArraySpec,
    DataField,
    DatasetRef,
    Kind,
    NexusFile,
    OpaqueFile,
    OutputRef,
    Ref,
    as_ref,
    data_fields,
    ref_fields,
    walk_refs,
)
from .parameters import Quantity

__all__ = [
    'Array',
    'ArraySpec',
    'DataField',
    'DatasetRef',
    'Kind',
    'NexusFile',
    'NoParams',
    'OpaqueFile',
    'OutputRef',
    'Quantity',
    'Ref',
    'SerializedWorkflowSpec',
    'WorkflowSpec',
    'as_ref',
    'data_fields',
    'ref_fields',
    'walk_refs',
]
