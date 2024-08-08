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
NoAnalyzer = Up
"""Indicates that element is not in use, e.g., analyzer in half-polarized case."""


class ReducedSampleDataBySpinChannel(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """Sample data for a given spin channel."""


Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)
PolarizingElement = TypeVar('PolarizingElement', Analyzer, Polarizer)

PlusMinus = Literal['plus', 'minus']


class TransmissionFunction(Generic[PolarizingElement], ABC):
    """Wavelength- and time-dependent transmission for a given cell."""

    @abstractmethod
    def apply(self, data: sc.DataArray, plus_minus: PlusMinus) -> sc.DataArray: ...


@dataclass
class PolarizingElementCorrection(
    Generic[PolarizerSpin, AnalyzerSpin, PolarizingElement]
):
    """Correction factors for polarizer or analyzer."""

    diag: sc.DataArray
    off_diag: sc.DataArray

    def get(self, *, up: bool) -> tuple[sc.DataArray, sc.DataArray]:
        """Get the correction factors for up or down spin."""
        if up:
            return self.diag, self.off_diag
        return self.off_diag, self.diag


@dataclass
class PolarizationCorrection(Generic[PolarizerSpin, AnalyzerSpin]):
    """Combined correction factors for polarizer and analyzer."""

    upup: sc.DataArray
    updown: sc.DataArray
    downup: sc.DataArray
    downdown: sc.DataArray


@dataclass
class HalfPolarizedCorrection(Generic[PolarizerSpin]):
    """Combined correction factors for half-polarized case."""

    up: sc.DataArray
    down: sc.DataArray


@dataclass
class PolarizationCorrectedData(Generic[PolarizerSpin, AnalyzerSpin]):
    """
    Polarization-corrected sample data.

    The PolarizerSpin and AnalyzerSpin type parameters refer to the measurement. The
    fields in this class give the resulting data after applying the corrections. For a
    given measurement with polarizer spin `PolarizerSpin` and analyzer spin
    `AnalyzerSpin`, there will be resulting intensity in all four output fields.
    """

    upup: sc.DataArray
    updown: sc.DataArray
    downup: sc.DataArray
    downdown: sc.DataArray


@dataclass
class HalfPolarizedCorrectedData(Generic[PolarizerSpin]):
    """
    Polarization-corrected sample data for half-polarized case.

    The PolarizerSpin type parameter refers to the measurement. The fields in this class
    give the resulting data after applying the corrections. For a given measurement with
    polarizer spin `PolarizerSpin`, there will be resulting intensity in both output
    fields.
    """

    up: sc.DataArray
    down: sc.DataArray


@dataclass
class FlipperEfficiency(Generic[PolarizingElement]):
    """Efficiency of a flipper"""

    value: float
