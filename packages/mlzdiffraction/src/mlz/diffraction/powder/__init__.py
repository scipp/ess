# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Components for POWTEX powder reduction."""

from . import conversion, correction, data, grouping, masking, workflow
from .workflow import PowderReduceP2D

providers = (
    *conversion.providers,
    *correction.providers,
    *data.providers,
    *grouping.providers,
    *masking.providers,
    *workflow.providers,
)

__all__ = [
    "PowderReduceP2D",
    "conversion",
    "correction",
    "data",
    "grouping",
    "masking",
    "providers",
    "workflow",
]
