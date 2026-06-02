# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the TBL workflow.
"""

from typing import Literal

import sciline

from ess.reduce.unwrap.workflow import GenericUnwrapWorkflow

from ..imaging.types import (
    BeamMonitor1,
    LookupTableRelativeErrorThreshold,
    NeXusMonitorName,
    PulseStrideOffset,
    SampleRun,
)


def default_parameters() -> dict:
    """Return the default workflow parameters for TBL."""
    return {
        NeXusMonitorName[BeamMonitor1]: "monitor_1",
        PulseStrideOffset: None,
        LookupTableRelativeErrorThreshold: {
            "ngem_detector": float("inf"),
            "he3_detector_bank0": float("inf"),
            "he3_detector_bank1": float("inf"),
            "multiblade_detector": float("inf"),
            "timepix3_detector": float("inf"),
            "monitor_1": float("inf"),
        },
    }


def TblWorkflow(
    wavelength_from: WavelengthLutMode = "file", **kwargs
) -> sciline.Pipeline:
    """
    Workflow with default parameters for TBL.

    Parameters
    ----------
    wavelength_from:
        Mode for creating the wavelength lookup table. The 'analytical' mode uses
        analytical calculations to propagate and chop a pulse through the chopper
        cascade and build the lookup table. The 'simulation' mode uses ``tof`` to trace
        individual neutrons through the chopper system and build the table.
        The 'file' mode loads a pre-computed table from a file.
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`."""
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun],
        monitor_types=[BeamMonitor1],
        wavelength_from=wavelength_from,
        **kwargs,
    )
    for key, param in default_parameters().items():
        workflow[key] = param
    return workflow
