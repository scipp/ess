# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from .workflow import MagicMcStasWorkflow, MagicWorkflow, default_parameters

try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'MagicMcStasWorkflow',
    'MagicWorkflow',
    '__version__',
    'default_parameters',
]
