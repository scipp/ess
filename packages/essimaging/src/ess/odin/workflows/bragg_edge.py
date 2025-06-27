# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters and workflow for Odin.
"""

import sciline

from ...imaging.conversion import providers as conversion_providers
from ..masking import providers as masking_providers
from .generic import OdinGenericWorkflow


def OdinBraggEdgeWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.
    """
    workflow = OdinGenericWorkflow(**kwargs)
    for provider in (*conversion_providers, *masking_providers):
        workflow.insert(provider)
    return workflow


__all__ = [
    "OdinBraggEdgeWorkflow",
]
