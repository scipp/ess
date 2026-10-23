# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from ..imaging import orca
from . import beamline
from .beamline import nexus_name
from .workflow import OrcaNormalizedImagesWorkflow, TblWorkflow, default_parameters

try:
    __version__ = importlib.metadata.version("esstbl")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    "OrcaNormalizedImagesWorkflow",
    "TblWorkflow",
    "beamline",
    "default_parameters",
    "nexus_name",
    "orca",
]
