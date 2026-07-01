# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Components for BEER
"""

import importlib.metadata

from .mcstas import load_beer_mcstas
from .peakfinding import dhkl_peaks_from_cif
from .workflow import (
    BeerMcStasWorkflowPulseShaping,
    BeerMcStasWorkflowPulseShapingAnalytical,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
    BeerPowderMcStasWorkflow,
    BeerPowderMcStasWorkflowAnalytical,
    BeerPowderWorkflow,
    BeerPowderWorkflowAnalytical,
    default_parameters,
)

try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'BeerMcStasWorkflowPulseShaping',
    'BeerMcStasWorkflowPulseShapingAnalytical',
    'BeerModMcStasWorkflow',
    'BeerModMcStasWorkflowKnownPeaks',
    'BeerPowderMcStasWorkflow',
    'BeerPowderMcStasWorkflowAnalytical',
    'BeerPowderWorkflow',
    'BeerPowderWorkflowAnalytical',
    '__version__',
    'default_parameters',
    'dhkl_peaks_from_cif',
    'load_beer_mcstas',
]
