# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

# flake8: noqa E402
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import data, general, io
from .general import default_parameters

providers = general.providers + io.providers

__all__ = ['data', 'general', 'io', 'providers', 'default_parameters']
