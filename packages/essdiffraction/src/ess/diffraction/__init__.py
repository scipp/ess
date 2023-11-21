# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for diffraction experiments (powder and single crystal).
"""

import importlib.metadata

from . import filtering
from .correction import normalize_by_monitor, normalize_by_vanadium
from .correction import providers as correction_providers
from .grouping import group_by_two_theta
from .smoothing import lowpass

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

providers = (*filtering.providers, *correction_providers)
"""Sciline providers for setting up a diffraction pipeline.

These implement basic diffraction data-reduction functionality and need to be
extended with instrument-specific and sub-technique-specific providers.
"""
del correction_providers

__all__ = [
    'lowpass',
    'group_by_two_theta',
    'normalize_by_monitor',
    'normalize_by_vanadium',
]
