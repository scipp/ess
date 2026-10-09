# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import beamline
from .beamline import nexus_name
from .workflows import OdinBraggEdgeWorkflow, OdinOrcaWorkflow, OdinWorkflow

try:
    __version__ = importlib.metadata.version("essodin")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    "OdinBraggEdgeWorkflow",
    "OdinOrcaWorkflow",
    "OdinWorkflow",
    "beamline",
    "nexus_name",
]
