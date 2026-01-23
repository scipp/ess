# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for BEER workflows.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus.types import Filename, RawDetector, RunType, SampleRun
from ess.reduce.time_of_flight.types import TofDetector


class StreakClusteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data binned by streak"""


RawDetector = RawDetector
Filename = Filename
SampleRun = SampleRun
TofDetector = TofDetector

DetectorBank = NewType('DetectorBank', int)

TwoThetaLimits = NewType("TwoThetaLimits", tuple[sc.Variable, sc.Variable])

TofCoordTransformGraph = NewType("TofCoordTransformGraph", dict)
GeometryCoordTransformGraph = NewType("GeometryCoordTransformGraph", dict)

PulseLength = NewType("PulseLength", sc.Variable)
"""Length of the neutron source pulse in time."""

ModulationPeriod = NewType("ModulationPeriod", sc.Variable)
"""The effective period of the modulating chopper:
``1 / (K * F)`` where ``K`` is the number of chopper openings and
``F`` is the chopper frequency."""

WavelengthDefinitionChopperDelay = NewType(
    "WavelengthDefinitionChopperDelay", sc.Variable
)
"""Wavelength definition chopper time delay relative to source pulse."""

DHKLList = NewType("DHKLList", sc.Variable)
"""List of peak position estimates."""

CIFIdentifierForPeakPositions = NewType("CIFIdentifierForPeakPositions", object)
"""CIF file or identifier used to find :py:`DHKLList`."""

CIFPeaksMinIntensity = NewType("CIFPeaksMinIntensity", sc.Variable)
"""Minimum peak intensity for peaks from CIF file to be included in :py:`DHKLList`."""
