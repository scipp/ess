# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from enum import Enum
from typing import NewType

import scipp as sc

from ess.reduce.nexus.types import Filename, RawDetector, RunType, SampleRun

RawDetector = RawDetector
Filename = Filename
RunType = RunType
SampleRun = SampleRun


class DetectorBank(Enum):
    detector_a = 'detector_a'
    detector_b = 'detector_b'


class PolarizationState(Enum):
    up = 'up'
    down = 'down'


IncidentPolarization = NewType("IncidentPolarization", sc.Variable)
"""Polarisation of the incident beam."""

ScatteredPolarization = NewType("ScatteredPolarization", sc.Variable)
"""Polarisation of the scattered beam (polarisation analysis mode)."""
