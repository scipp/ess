# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the Loki workflow.
"""

import sciline
import scipp as sc
import scippnexus as snx
from ess import sans
from ess.sans.parameters import typical_outputs

from ess.reduce.unwrap import SourceBounds, WavelengthLutMode
from ess.reduce.workflow import register_workflow

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
        NeXusMonitorName[Incident]: 'beam_monitor_m1',
        NeXusMonitorName[Transmission]: 'beam_monitor_m3',
        PixelShapePath: 'pixel_shape',
        NonBackgroundWavelengthRange: None,
        # Used by analytical lookup tables, which treat the pulse as flat in time.
        # LoKI has no pulse-shaping choppers, so the window sets the mean and standard
        # deviation of the table wavelengths. A flat window mu +- sqrt(3) sigma has the
        # mean (mu = 1.76 ms) and standard deviation (sigma = 0.88 ms) of the emission
        # time of the McStas ESS source for 1-10 Angstrom.
        SourceBounds: SourceBounds(
            time=(sc.scalar(0.24, unit='ms'), sc.scalar(3.29, unit='ms')),
            wavelength=(
                sc.scalar(0.001, unit='angstrom'),
                sc.scalar(15.0, unit='angstrom'),
            ),
        ),
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
def LokiWorkflow(
    wavelength_from: WavelengthLutMode = "file",
) -> sciline.Pipeline:
    """
    Workflow with default parameters for Loki.

    Parameters
    ----------
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html

    Returns
    -------
    :
        Loki workflow as a sciline.Pipeline
    """
    workflow = sans.SansWorkflow(wavelength_from=wavelength_from)
    for provider in loki_providers:
        workflow.insert(provider)
    for key, param in loki_default_parameters().items():
        workflow[key] = param
    workflow.typical_outputs = typical_outputs
    return workflow
