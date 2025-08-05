# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the TBL workflow.
"""

import sciline

from ess.reduce.time_of_flight.workflow import GenericTofWorkflow

from ..imaging.conversion import providers as conversion_providers
from ..imaging.types import (
    BeamMonitor1,
    NeXusMonitorName,
    PulseStrideOffset,
    SampleRun,
)


def default_parameters() -> dict:
    return {
        NeXusMonitorName[BeamMonitor1]: "monitor_1",
        PulseStrideOffset: None,
    }


providers = (*conversion_providers,)


def TblWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for TBL.
    """
    workflow = GenericTofWorkflow(
        run_types=[SampleRun], monitor_types=[BeamMonitor1], **kwargs
    )
    for provider in providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow
