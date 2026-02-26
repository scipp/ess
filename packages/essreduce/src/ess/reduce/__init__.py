# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import kinematics, nexus, normalization, uncertainty

try:
    __version__ = importlib.metadata.version("essreduce")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = ["kinematics", "nexus", "normalization", "uncertainty"]
