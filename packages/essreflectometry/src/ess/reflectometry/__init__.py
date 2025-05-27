# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version("essreflectometry")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


from . import conversions, corrections, figures, normalization, orso
from .load import load_reference, save_reference

providers = (
    *corrections.providers,
    *conversions.providers,
    *orso.providers,
    *normalization.providers,
    *figures.providers,
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
    "figures",
    "load_reference",
    "save_reference",
]
