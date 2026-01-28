# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: RUF100, E402, F401, I

import importlib.metadata

from . import orca
from .orca import OrcaNormalizedImagesWorkflow
from .workflow import TblWorkflow, default_parameters

try:
    __version__ = importlib.metadata.version("esstbl")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    "OrcaNormalizedImagesWorkflow",
    "TblWorkflow",
    "default_parameters",
    "orca",
]
