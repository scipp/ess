# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters and workflow for Odin.
"""

import sciline

from ess.reduce.unwrap import GenericUnwrapWorkflow, WavelengthLutMode

from ..imaging.types import (
    BeamMonitor1,
    BeamMonitor2,
    BeamMonitor3,
    BeamMonitor4,
    DarkBackgroundRun,
    LookupTableRelativeErrorThreshold,
    NeXusMonitorName,
    OpenBeamRun,
    PulseStrideOffset,
    SampleRun,
)
from .masking import providers as masking_providers


def default_parameters() -> dict:
    """Return the default workflow parameters for Odin."""
    return {
        NeXusMonitorName[BeamMonitor1]: "beam_monitor_1",
        NeXusMonitorName[BeamMonitor2]: "beam_monitor_2",
        NeXusMonitorName[BeamMonitor3]: "beam_monitor_3",
        NeXusMonitorName[BeamMonitor4]: "beam_monitor_4",
        PulseStrideOffset: None,
        LookupTableRelativeErrorThreshold: {
            "event_mode_detectors/timepix3": float("inf"),
            "histogram_mode_detectors/orca": float("inf"),
            "beam_monitor_1": float("inf"),
            "beam_monitor_2": float("inf"),
            "beam_monitor_3": float("inf"),
            "beam_monitor_4": float("inf"),
        },
    }


def OdinWorkflow(
    wavelength_from: WavelengthLutMode = "file", **kwargs
) -> sciline.Pipeline:
    """
    Workflow with default parameters for Odin.

    Parameters
    ----------
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`."""
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun, OpenBeamRun, DarkBackgroundRun],
        monitor_types=[BeamMonitor1, BeamMonitor2, BeamMonitor3, BeamMonitor4],
        wavelength_from=wavelength_from,
        **kwargs,
    )
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow


def OdinBraggEdgeWorkflow(
    wavelength_from: WavelengthLutMode = "file", **kwargs
) -> sciline.Pipeline:
    """
    Workflow with default parameters and masking providers
    for Odin Bragg-edge reduction.

    Parameters
    ----------
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`."""
    workflow = OdinWorkflow(wavelength_from=wavelength_from, **kwargs)
    for provider in (*masking_providers,):
        workflow.insert(provider)
    return workflow


__all__ = [
    "OdinBraggEdgeWorkflow",
    "OdinWorkflow",
]
