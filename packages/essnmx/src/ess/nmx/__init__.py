# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: RUF100, E402, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version("essnmx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .mcstas import NMXMcStasWorkflow

__all__ = ["NMXMcStasWorkflow"]
