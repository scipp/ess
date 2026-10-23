# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from enum import Enum
from typing import NewType

import sciline as sl
import scipp as sc

from ess.reduce.nexus.types import (
    EmptyDetector,
    Filename,
    RawDetector,
    RunType,
    SampleRun,
)

EmptyDetector = EmptyDetector
RawDetector = RawDetector
Filename = Filename
RunType = RunType
SampleRun = SampleRun


class DetectorBank(Enum):
    detector_a = 'detector_a'
    detector_b = 'detector_b'


class RawMcStasEvents(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Flat weighted absorption-logger rows before geometry or timing corrections.

    Coordinates retain the McStas logger's local frame and original arrival time.
    These events have not been assigned to detector pixels.
    """


class PolarizationState(Enum):
    up = 'up'
    down = 'down'


IncidentPolarization = NewType("IncidentPolarization", sc.Variable)
"""Polarisation of the incident beam."""

ScatteredPolarization = NewType("ScatteredPolarization", sc.Variable)
"""Polarisation of the scattered beam (polarisation analysis mode)."""
