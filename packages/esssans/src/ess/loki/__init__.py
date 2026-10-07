# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import workflow
from .diagnostics import InstrumentView, LokiBankViewer, instrument_view
from .larmor_workflow import (
    LokiAtLarmorTutorialWorkflow,
    LokiAtLarmorWorkflow,
    larmor_default_parameters,
)
from .safe_load import load_detectors, load_monitors
from .workflow import LokiWorkflow, loki_default_parameters

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'InstrumentView',
    'LokiAtLarmorTutorialWorkflow',
    'LokiAtLarmorWorkflow',
    'LokiBankViewer',
    'LokiWorkflow',
    'instrument_view',
    'larmor_default_parameters',
    'load_detectors',
    'load_monitors',
    'loki_default_parameters',
    'workflow',
]
