# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for diffraction experiments (powder and single crystal).
"""

import importlib.metadata

from .corrections import normalize_by_monitor, normalize_by_vanadium
from .grouping import group_by_two_theta
from .smoothing import lowpass

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'lowpass',
    'group_by_two_theta',
    'normalize_by_monitor',
    'normalize_by_vanadium',
]
