# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .data import small_mcstas_3_sample
from .reduction import NMXData, NMXReducedData
from .types import MaximumProbability

default_parameters = {MaximumProbability: 10000}

del MaximumProbability

__all__ = [
    "small_mcstas_3_sample",
    "NMXReducedData",
    "NMXData",
    "default_parameters",
]
