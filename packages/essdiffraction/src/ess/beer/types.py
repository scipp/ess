# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for BEER workflows.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from enum import Enum
from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus.types import Filename, RawDetector, RunType, SampleRun
from ess.reduce.unwrap.types import WavelengthDetector


class StreakClusteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data binned by streak"""


class BeamlineParameters(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """BEER wavelength, chopper, and detector geometry parameters.

    Computed without event data. See :func:`ess.beer.parameters.beamline_parameters`
    for the entries and their conventions.
    """


RawDetector = RawDetector
Filename = Filename
SampleRun = SampleRun
WavelengthDetector = WavelengthDetector


class DetectorBank(Enum):
    north = 'north'
    south = 'south'
    both = 'both'


PulseLength = NewType("PulseLength", sc.Variable)
"""Length of the neutron source pulse in time."""

DHKLList = NewType("DHKLList", sc.Variable)
"""List of peak position estimates."""

CIFIdentifierForPeakPositions = NewType("CIFIdentifierForPeakPositions", object)
"""CIF file or identifier used to find :py:`DHKLList`."""

CIFPeaksMinIntensity = NewType("CIFPeaksMinIntensity", sc.Variable)
"""Minimum peak intensity for peaks from CIF file to be included in :py:`DHKLList`."""
