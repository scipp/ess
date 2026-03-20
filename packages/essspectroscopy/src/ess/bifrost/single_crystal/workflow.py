# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""BIFROST Bragg peak monitor workflows."""

import sciline

from ess.reduce import unwrap as reduce_unwrap
from ess.spectroscopy.indirect.time_of_flight import TofWorkflow
from ess.spectroscopy.types import (
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
    SampleRun,
)

from ..cutting import group_by_rotation
from ..io import nexus
from ..io.mcstas import convert_simulated_time_to_event_time_offset
from ..workflow import default_parameters, simulation_default_parameters
from . import conversion, q_map, time_of_flight

_PROVIDERS = (
    *nexus.providers,
    *conversion.providers,
    *q_map.providers,
    *time_of_flight.providers,
    group_by_rotation,
)

_SIMULATION_PROVIDERS = (
    *nexus.providers,
    *conversion.providers,
    *q_map.providers,
    *time_of_flight.providers,
    convert_simulated_time_to_event_time_offset,
    group_by_rotation,
)


def BifrostBraggPeakMonitorWorkflow() -> sciline.Pipeline:
    workflow = TofWorkflow(
        run_types=(SampleRun,),
        monitor_types=(FrameMonitor1, FrameMonitor2, FrameMonitor3),
    )
    # Use the vanilla implementation instead of the indirect geometry one:
    workflow.insert(reduce_unwrap.to_wavelength.detector_wavelength_data)
    for provider in _PROVIDERS:
        workflow.insert(provider)
    for key, val in default_parameters().items():
        workflow[key] = val
    return workflow


def BifrostSimulationBraggPeakMonitorWorkflow() -> sciline.Pipeline:
    workflow = TofWorkflow(
        run_types=(SampleRun,),
        monitor_types=(FrameMonitor1, FrameMonitor2, FrameMonitor3),
    )
    # Use the vanilla implementation instead of the indirect geometry one:
    workflow.insert(reduce_unwrap.to_wavelength.detector_wavelength_data)
    for provider in _SIMULATION_PROVIDERS:
        workflow.insert(provider)
    for key, val in simulation_default_parameters().items():
        workflow[key] = val
    return workflow
