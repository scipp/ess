# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

"""Components for BIFROST."""

import importlib.metadata

from .io import nexus
from .workflow import BifrostSimulationWorkflow
from .detector import providers

try:
    __version__ = importlib.metadata.version("essspectroscopy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = ['BifrostSimulationWorkflow', 'nexus', 'providers']
