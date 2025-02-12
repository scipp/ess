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
from .correction import RunNormalization
from .masking import with_pixel_mask_filenames

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
    "RunNormalization",
    "__version__",
    "conversion",
    "correction",
    "filtering",
    "grouping",
    "masking",
    "providers",
    "smoothing",
    "transform",
    "with_pixel_mask_filenames",
]
