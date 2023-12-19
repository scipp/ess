# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .data import small_mcstas_sample
from .mcstas_loader import InputFilepath, NMXData, load_mcstas_nexus
from .workflow import collect_default_parameters

__all__ = [
    "small_mcstas_sample",
    "NMXData",
    "InputFilepath",
    "load_mcstas_nexus",
    "collect_default_parameters",
]
