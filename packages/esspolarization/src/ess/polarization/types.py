# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, NewType, TypeVar

import sciline as sl
import scipp as sc

Up = NewType('Up', int)
Down = NewType('Down', int)
PolarizerSpin = TypeVar('PolarizerSpin', Up, Down)
AnalyzerSpin = TypeVar('AnalyzerSpin', Up, Down)


class ReducedSampleDataBySpinChannel(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """Sample data for a given spin channel."""


@dataclass
class AnalyzerCorrectedData(Generic[PolarizerSpin]):
    """Sample data with analyzer correction, prior to polarizer correction."""

    analyzer_up: sc.DataArray
    analyzer_down: sc.DataArray


@dataclass
class PolarizationCorrectedData:
    """Polarization-corrected sample data."""

    upup: sc.DataArray
    updown: sc.DataArray
    downup: sc.DataArray
    downdown: sc.DataArray


Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)
PolarizingElement = TypeVar('PolarizingElement', Analyzer, Polarizer)


class TransmissionFunction(Generic[PolarizingElement], ABC):
    """Wavelength- and time-dependent transmission for a given cell."""

    @abstractmethod
    def apply(
        self, data: sc.DataArray, plus_minus: Literal['plus', 'minus']
    ) -> sc.DataArray:
        ...


class PolarizingElementCorrection(
    Generic[PolarizerSpin, AnalyzerSpin, PolarizingElement]
):
    """Correction factors for polarizing element."""

    diag: sc.DataArray
    off_diag: sc.DataArray
