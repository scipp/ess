# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the Loki workflow.
"""

import sciline
import scipp as sc
import scippnexus as snx

from ess import sans
from ess.reduce.workflow import register_workflow
from ess.sans.parameters import typical_outputs

from ..sans.types import (
    DetectorBankSizes,
    DetectorPixelShape,
    DirectBeam,
    DirectBeamFilename,
    Incident,
    LookupTableRelativeErrorThreshold,
    NeXusComponent,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    PixelShapePath,
    RunType,
    Transmission,
)

DETECTOR_BANK_SIZES = {
    'loki_detector_0': {'layer': 4, 'tube': 56, 'straw': 7, 'pixel': -1},
    'loki_detector_1': {'layer': 4, 'tube': 16, 'straw': 7, 'pixel': -1},
    'loki_detector_2': {'layer': 4, 'tube': 12, 'straw': 7, 'pixel': -1},
    'loki_detector_3': {'layer': 4, 'tube': 16, 'straw': 7, 'pixel': -1},
    'loki_detector_4': {'layer': 4, 'tube': 12, 'straw': 7, 'pixel': -1},
    'loki_detector_5': {'layer': 4, 'tube': 28, 'straw': 7, 'pixel': -1},
    'loki_detector_6': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': -1},
    'loki_detector_7': {'layer': 4, 'tube': 20, 'straw': 7, 'pixel': -1},
    'loki_detector_8': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': -1},
}


def loki_default_parameters() -> dict:
    return {
        DetectorBankSizes: DETECTOR_BANK_SIZES,
        NeXusMonitorName[Incident]: 'beam_monitor_1',
        NeXusMonitorName[Transmission]: 'beam_monitor_3',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
        LookupTableRelativeErrorThreshold: {
            **{f'loki_detector_{i}': float('inf') for i in range(9)},
            # Monitors where renamed, beam_monitor_i is the old name
            # and beam_monitor_mi is the new.
            **{f'beam_monitor_{i}': float('inf') for i in range(5)},
            **{f'beam_monitor_m{i}': float('inf') for i in range(5)},
        },
    }


def detector_pixel_shape(
    detector: NeXusComponent[snx.NXdetector, RunType],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[RunType]:
    return DetectorPixelShape[RunType](detector[pixel_shape_path])


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    """Load direct beam from file."""
    return DirectBeam(sc.io.load_hdf5(filename))


loki_providers = (detector_pixel_shape, load_direct_beam)


@register_workflow
def LokiWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for Loki.

    Returns
    -------
    :
        Loki workflow as a sciline.Pipeline
    """
    workflow = sans.SansWorkflow()
    for provider in loki_providers:
        workflow.insert(provider)
    for key, param in loki_default_parameters().items():
        workflow[key] = param
    workflow.typical_outputs = typical_outputs
    return workflow
