# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Implementation-independent workflow specifications for UI generation.

See :mod:`ess.reduce.spec._workflow_spec` for the design;
ADR 0001 (docs/developer/adr) for the rationale.
"""

from ._workflow_spec import (
    ArraySpec,
    NoParams,
    OutputSpec,
    SerializedWorkflowSpec,
    WorkflowSpec,
)

__all__ = [
    'ArraySpec',
    'NoParams',
    'OutputSpec',
    'SerializedWorkflowSpec',
    'WorkflowSpec',
]
