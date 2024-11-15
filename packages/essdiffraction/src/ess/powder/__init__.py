# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Components for powder diffraction experiments.
"""

import importlib.metadata

from . import (
    calibration,
    conversion,
    correction,
    filtering,
    grouping,
    masking,
    smoothing,
    transform,
)
from .masking import with_pixel_mask_filenames
from .correction import RunNormalization

try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

providers = (
    *calibration.providers,
    *conversion.providers,
    *correction.providers,
    *filtering.providers,
    *grouping.providers,
    *masking.providers,
)

"""Sciline providers for powder diffraction."""

__all__ = [
    "__version__",
    "RunNormalization",
    "conversion",
    "correction",
    "filtering",
    "grouping",
    "masking",
    "transform",
    "providers",
    "smoothing",
    "with_pixel_mask_filenames",
]
