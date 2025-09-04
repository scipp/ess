# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Components for BEER
"""

import importlib.metadata

from .io import load_beer_mcstas
from .workflow import (
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
    default_parameters,
)

try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'BeerModMcStasWorkflow',
    'BeerModMcStasWorkflowKnownPeaks',
    '__version__',
    'default_parameters',
    'load_beer_mcstas',
]
