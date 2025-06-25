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
    EmptyBeamRun,
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
from .masking import providers as masking_providers


def default_parameters() -> dict:
    return {
        NeXusMonitorName[FrameMonitor1]: "beam_monitor_3",
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
        PulseStride: 2,
        PulseStrideOffset: None,
        LookupTableRelativeErrorThreshold: 0.1,
        LtotalRange: (sc.scalar(55.0, unit="m"), sc.scalar(65.0, unit="m")),
        DistanceResolution: sc.scalar(0.1, unit="m"),
        TimeResolution: sc.scalar(250.0, unit='us'),
        NumberOfSimulatedNeutrons: 1_000_000,
    }


providers = (*conversion_providers, *masking_providers)


def OdinWorkflow(**kwargs) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.
    """
    workflow = GenericTofWorkflow(
        tof_lut_provider=time_of_flight.TofLutProvider.TOF,
        run_types=[SampleRun, EmptyBeamRun],
        monitor_types=[FrameMonitor1],
        **kwargs,
    )
    for provider in providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow
