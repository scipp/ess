# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version("esspolarization")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib


from .correction import (
    CorrectionWorkflow,
    HalfPolarizedWorkflow,
    PolarizationAnalysisWorkflow,
)
from .he3 import (
    Depolarized,
    DirectBeamBackgroundQRange,
    DirectBeamNoCell,
    DirectBeamQRange,
    He3CellLength,
    He3CellPressure,
    He3CellWorkflow,
    He3DirectBeam,
    He3FillingTime,
    He3Opacity0,
    He3OpacityFunction,
    He3PolarizationFunction,
    He3TransmissionEmptyGlass,
    He3TransmissionFunction,
    Polarized,
)
from .supermirror import (
    SecondDegreePolynomialEfficiency,
    SupermirrorEfficiencyFunction,
    SupermirrorWorkflow,
)
from .types import (
    Analyzer,
    Down,
    HalfPolarizedCorrectedData,
    NoAnalyzer,
    PolarizationCorrectedData,
    Polarizer,
    PolarizingElement,
    ReducedSampleDataBySpinChannel,
    TransmissionFunction,
    Up,
)

__all__ = [
    "Analyzer",
    "CorrectionWorkflow",
    "Depolarized",
    "DirectBeamBackgroundQRange",
    "DirectBeamNoCell",
    "DirectBeamQRange",
    "Down",
    "HalfPolarizedCorrectedData",
    "HalfPolarizedWorkflow",
    "He3CellLength",
    "He3CellPressure",
    "He3CellWorkflow",
    "He3DirectBeam",
    "He3FillingTime",
    "He3Opacity0",
    "He3OpacityFunction",
    "He3PolarizationFunction",
    "He3TransmissionEmptyGlass",
    "He3TransmissionFunction",
    "NoAnalyzer",
    "PolarizationAnalysisWorkflow",
    "PolarizationCorrectedData",
    "Polarizer",
    "PolarizingElement",
    "ReducedSampleDataBySpinChannel",
    "SecondDegreePolynomialEfficiency",
    "SupermirrorEfficiencyFunction",
    "SupermirrorWorkflow",
    "TransmissionFunction",
    "Up",
]
