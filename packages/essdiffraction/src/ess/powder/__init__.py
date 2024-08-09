# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Components for powder diffraction experiments.
"""

import importlib.metadata

from . import (
    conversion,
    correction,
    filtering,
    grouping,
    masking,
    smoothing,
    transform,
)
from .masking import with_pixel_mask_filenames
from . import nexus

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
    *masking.providers,
)

"""Sciline providers for powder diffraction."""

__all__ = [
    "conversion",
    "correction",
    "filtering",
    "grouping",
    "masking",
    "nexus",
    "transform",
    "providers",
    "smoothing",
    "with_pixel_mask_filenames",
]
