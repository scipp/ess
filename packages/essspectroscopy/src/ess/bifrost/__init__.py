# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

"""BIFROST data reduction."""

import importlib.metadata

from .detector import providers
from .io import nexus
from .workflow import BifrostSimulationWorkflow, BifrostWorkflow

try:
    __version__ = importlib.metadata.version("essspectroscopy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = ['BifrostSimulationWorkflow', 'BifrostWorkflow', 'nexus', 'providers']
