# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version("essnmx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .data import small_mcstas_3_sample
from .reduction import NMXReducedDataGroup
from .types import MaximumCounts, NMXRawEventCountsDataGroup

default_parameters = {MaximumCounts: 10000}

del MaximumCounts

__all__ = [
    "NMXRawEventCountsDataGroup",
    "NMXReducedDataGroup",
    "default_parameters",
    "small_mcstas_3_sample",
]
