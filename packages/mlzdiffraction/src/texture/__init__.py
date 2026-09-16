# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Components for POWTEX powder reduction."""

from . import conversion, correction, grouping, histogramming
from .workflow import TextureReduce

providers = (
    *conversion.providers,
    *correction.providers,
    *grouping.providers,
    *histogramming.providers,
)

__all__ = [
    "TextureReduce",
    "conversion",
    "correction",
    "grouping",
    "histogramming",
]
