# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the TBL workflow.
"""

import sciline

from ess.reduce.unwrap import GenericUnwrapWorkflow, WavelengthLutMode

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
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html
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
