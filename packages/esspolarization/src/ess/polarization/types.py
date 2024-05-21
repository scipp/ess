# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import sciline as sl
import scipp as sc

Up = NewType('Up', int)
Down = NewType('Down', int)
PolarizerSpin = TypeVar('PolarizerSpin', Up, Down)
AnalyzerSpin = TypeVar('AnalyzerSpin', Up, Down)


PolarizationCorrectedSampleData = NewType(
    'PolarizationCorrectedSampleData', sc.DataArray
)
"""Polarization-corrected sample data."""


class ReducedSampleDataBySpinChannel(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """Sample data for a given spin channel."""
