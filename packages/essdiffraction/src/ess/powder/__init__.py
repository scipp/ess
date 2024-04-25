# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F401
"""
Components for powder diffraction experiments.
"""

import importlib.metadata

from . import conversion, correction, filtering, grouping, smoothing, uncertainty

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

providers = (
    *conversion.providers,
    *correction.providers,
    *filtering.providers,
    *grouping.providers,
)
"""Sciline providers for powder diffraction."""
