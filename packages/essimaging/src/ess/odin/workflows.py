# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters and workflow for Odin.
"""

import sciline

from ess.reduce.time_of_flight.workflow import GenericTofWorkflow

from ..imaging.conversion import providers as conversion_providers
from ..imaging.types import (
    BeamMonitor1,
    BeamMonitor2,
    BeamMonitor3,
    BeamMonitor4,
    DarkBackgroundRun,
    NeXusMonitorName,
    OpenBeamRun,
    PulseStrideOffset,
    SampleRun,
)
from .masking import providers as masking_providers


def default_parameters() -> dict:
    return {
        NeXusMonitorName[BeamMonitor1]: "beam_monitor_1",
        NeXusMonitorName[BeamMonitor2]: "beam_monitor_2",
        NeXusMonitorName[BeamMonitor3]: "beam_monitor_3",
        NeXusMonitorName[BeamMonitor4]: "beam_monitor_4",
        PulseStrideOffset: None,
    }


def OdinWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.
    """
    workflow = GenericTofWorkflow(
        run_types=[SampleRun, OpenBeamRun, DarkBackgroundRun],
        monitor_types=[BeamMonitor1, BeamMonitor2, BeamMonitor3, BeamMonitor4],
        **kwargs,
    )
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow


def OdinBraggEdgeWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.
    """
    workflow = OdinWorkflow(**kwargs)
    for provider in (*conversion_providers, *masking_providers):
        workflow.insert(provider)
    return workflow


__all__ = [
    "OdinBraggEdgeWorkflow",
    "OdinWorkflow",
]
