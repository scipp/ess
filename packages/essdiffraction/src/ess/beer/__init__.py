# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Components for BEER
"""

import importlib.metadata

import lazy_loader as lazy

from .mcstas import load_beer_mcstas
from .peakfinding import dhkl_peaks_from_cif
from .workflow import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
    BeerModulationAutoMcStasWorkflow,
    BeerModulationKnownPeaksMcStasWorkflow,
    BeerPowderMcStasWorkflow,
    BeerPowderWorkflow,
    BeerPowderWorkflowAnalytical,
    default_parameters,
)

try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# The diagnostics subpackage needs ipywidgets, which is optional. Import it only on
# first access so that the workflows can be used without ipywidgets.
__getattr__, _, _ = lazy.attach(
    __name__,
    submodules=['diagnostics'],
)

del _, importlib, lazy

__all__ = [
    'BeerMcStasWorkflowPulseShaping',
    'BeerModMcStasWorkflow',
    'BeerModMcStasWorkflowKnownPeaks',
    'BeerModulationAutoMcStasWorkflow',
    'BeerModulationKnownPeaksMcStasWorkflow',
    'BeerPowderMcStasWorkflow',
    'BeerPowderWorkflow',
    'BeerPowderWorkflowAnalytical',
    '__version__',
    'default_parameters',
    'dhkl_peaks_from_cif',
    'load_beer_mcstas',
]
