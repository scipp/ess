# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""BIFROST data reduction."""

import importlib.metadata

import lazy_loader as lazy

from .io import nexus
from .workflow import BifrostSimulationWorkflow, BifrostWorkflow

try:
    __version__ = importlib.metadata.version("essspectroscopy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# The diagnostics subpackage needs ipywidgets, which is optional. Import it only on
# first access so that the workflows can be used without ipywidgets.
__getattr__, _, _ = lazy.attach(
    __name__,
    submodules=['diagnostics'],
    submod_attrs={'diagnostics': ['InstrumentView', 'instrument_view']},
)

del _, importlib, lazy

__all__ = [
    'BifrostSimulationWorkflow',
    'BifrostWorkflow',
    'nexus',
]
