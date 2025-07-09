# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters and workflow for Odin.
"""

import sciline
import scipp as sc

from ess.reduce.time_of_flight.workflow import GenericTofWorkflow

from ..imaging.conversion import providers as conversion_providers
from ..imaging.types import (
    # DistanceResolution,
    # FrameMonitor0,
    BeamMonitor1,
    BeamMonitor2,
    BeamMonitor3,
    BeamMonitor4,
    DarkBackgroundRun,
    # LookupTableRelativeErrorThreshold,
    # LtotalRange,
    NeXusMonitorName,
    # NumberOfSimulatedNeutrons,
    OpenBeamRun,
    # PulsePeriod,
    # PulseStride,
    PulseStrideOffset,
    SampleRun,
    # TimeResolution,
)
from .masking import providers as masking_providers


def default_parameters() -> dict:
    return {
        NeXusMonitorName[BeamMonitor1]: "beam_monitor_1",
        NeXusMonitorName[BeamMonitor2]: "beam_monitor_2",
        NeXusMonitorName[BeamMonitor3]: "beam_monitor_3",
        NeXusMonitorName[BeamMonitor4]: "beam_monitor_4",
        # PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
        # PulseStride: 2,
        PulseStrideOffset: None,
        # LookupTableRelativeErrorThreshold: 0.1,
        # LtotalRange: (sc.scalar(55.0, unit="m"), sc.scalar(65.0, unit="m")),
        # DistanceResolution: sc.scalar(0.1, unit="m"),
        # TimeResolution: sc.scalar(250.0, unit='us'),
        # NumberOfSimulatedNeutrons: 1_000_000,
    }


# providers = (*conversion_providers, *masking_providers)


def OdinWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.
    """
    workflow = GenericTofWorkflow(
        run_types=[SampleRun, OpenBeamRun, DarkBackgroundRun],
        monitor_types=[BeamMonitor1, BeamMonitor2, BeamMonitor3, BeamMonitor4],
        **kwargs,
    )
    # for provider in providers:
    #     workflow.insert(provider)
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
