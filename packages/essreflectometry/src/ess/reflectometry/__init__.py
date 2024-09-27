# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

import importlib.metadata

from . import calibrations, conversions, corrections, normalize, orso
from .load import load_reference, save_reference

try:
    __version__ = importlib.metadata.version("essreflectometry")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

providers = (
    *conversions.providers,
    *corrections.providers,
    *calibrations.providers,
    *normalize.providers,
    *orso.providers,
)
"""
List of providers for setting up a Sciline pipeline.

This does not constitute a complete workflow but only
a skeleton for a generic reflectometry setting.
For an example of a complete workflow
see :py:data:`essreflectometry.amor.providers`.
"""

del importlib


__all__ = [
    "load_reference",
    "save_reference",
]
