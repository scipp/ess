# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import importlib.metadata

from ..reflectometry import supermirror
from . import conversions, load, maskings, normalization, orso, resolution, workflow
from .types import (
    AngularResolution,
    SampleSizeResolution,
    WavelengthResolution,
)
from .workflow import EstiaMcStasWorkflow, EstiaWorkflow

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AngularResolution",
    "EstiaMcStasWorkflow",
    "EstiaWorkflow",
    "SampleSizeResolution",
    "WavelengthResolution",
    "conversions",
    "load",
    "maskings",
    "normalization",
    "orso",
    "resolution",
    "supermirror",
    "workflow",
]
