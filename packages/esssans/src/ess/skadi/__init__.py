# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import mcstas, workflow
from .mcstas import load_skadi_mcstas
from .workflow import SkadiMcStasWorkflow, SkadiWorkflow, skadi_default_parameters

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'SkadiMcStasWorkflow',
    'SkadiWorkflow',
    'load_skadi_mcstas',
    'mcstas',
    'skadi_default_parameters',
    'workflow',
]
