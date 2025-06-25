# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the TBL workflow.
"""

import sciline
import scipp as sc

from ess.reduce import time_of_flight
from ess.reduce.time_of_flight.workflow import GenericTofWorkflow

from ..imaging.conversion import providers as conversion_providers
from ..imaging.types import (
    DistanceResolution,
    FrameMonitor1,
    LookupTableRelativeErrorThreshold,
    LtotalRange,
    NeXusMonitorName,
    NumberOfSimulatedNeutrons,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    SampleRun,
    TimeResolution,
)


def default_parameters() -> dict:
    return {
        NeXusMonitorName[FrameMonitor1]: "monitor_1",
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
        PulseStride: 1,
        PulseStrideOffset: None,
        LookupTableRelativeErrorThreshold: 1.0,
        LtotalRange: (sc.scalar(25.0, unit="m"), sc.scalar(35.0, unit="m")),
        DistanceResolution: sc.scalar(0.1, unit="m"),
        TimeResolution: sc.scalar(250.0, unit='us'),
        NumberOfSimulatedNeutrons: 200_000,
    }


providers = (*conversion_providers,)


def TblWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for TBL.
    """
    workflow = GenericTofWorkflow(
        run_types=[SampleRun],
        monitor_types=[FrameMonitor1],
        **kwargs,
    )
    for provider in providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow
