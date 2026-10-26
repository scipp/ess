# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import importlib.metadata

import lazy_loader as lazy

from . import workflow
from .larmor_workflow import (
    LokiAtLarmorTutorialWorkflow,
    LokiAtLarmorWorkflow,
    larmor_default_parameters,
)
from .workflow import LokiWorkflow, loki_default_parameters

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# The diagnostics subpackage needs ipywidgets, which is optional. Import it only on
# first access so that the workflows can be used without ipywidgets.
__getattr__, _, _ = lazy.attach(
    __name__,
    submodules=['diagnostics'],
    submod_attrs={
        'diagnostics': ['InstrumentView', 'LokiBankViewer', 'instrument_view']
    },
)

del _, importlib, lazy

__all__ = [
    'InstrumentView',
    'LokiAtLarmorTutorialWorkflow',
    'LokiAtLarmorWorkflow',
    'LokiBankViewer',
    'LokiWorkflow',
    'instrument_view',
    'larmor_default_parameters',
    'loki_default_parameters',
    'workflow',
]
