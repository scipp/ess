# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import data, general, io
from .general import default_parameters

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

providers = general.providers + io.providers

del importlib

__all__ = ['data', 'general', 'io', 'providers', 'default_parameters']
