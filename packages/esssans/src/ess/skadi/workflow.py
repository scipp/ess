# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Default providers and parameters for the SKADI SANS workflow."""

import sciline
from ess import sans
from ess.sans.normalization import rectangular_pixel_solid_angle
from ess.sans.parameters import typical_outputs

from ess.reduce.uncertainty import UncertaintyBroadcastMode
from ess.reduce.unwrap import WavelengthLutMode
from ess.reduce.workflow import register_workflow

from ..sans.types import (
    DetectorMasks,
    DirectBeam,
    ReturnEvents,
)
from .mcstas import mcstas_providers


def skadi_default_parameters() -> dict:
    """Return defaults for a minimal SKADI reduction."""
    return {
        DetectorMasks: {},
        DirectBeam: None,
        ReturnEvents: False,
    }


@register_workflow
def SkadiWorkflow(
    wavelength_from: WavelengthLutMode = "file",
) -> sciline.Pipeline:
    """Create a basic, data-source-independent SKADI reduction workflow.

    Parameters
    ----------
    wavelength_from:
        Mode used by the common SANS workflow to obtain wavelength.

    Returns
    -------
    :
        The SKADI reduction workflow.
    """
    workflow = sans.SansWorkflow(wavelength_from=wavelength_from)
    workflow.insert(rectangular_pixel_solid_angle)
    for key, value in skadi_default_parameters().items():
        workflow[key] = value
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
def SkadiMcStasWorkflow() -> sciline.Pipeline:
    """Create the SKADI McStas workflow for relative intensity reductions.

    In the absence of monitor data, assume a flat incident spectrum with unit
    intensity per angstrom and unit transmission.
    """
    workflow = SkadiWorkflow()
    for provider in mcstas_providers:
        workflow.insert(provider)
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    return workflow
