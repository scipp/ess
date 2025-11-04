# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from types import MappingProxyType
from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.reduce.time_of_flight import types as tof_t

# 1 TypeVars used to parametrize the generic parts of the workflow

Filename = reduce_t.Filename
GravityVector = reduce_t.GravityVector
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent
Position = reduce_t.Position
RawDetector = reduce_t.RawDetector
RawMonitor = reduce_t.RawMonitor

DetectorLtotal = tof_t.DetectorLtotal
TofDetector = tof_t.TofDetector
PulseStrideOffset = tof_t.PulseStrideOffset
TimeOfFlightLookupTable = tof_t.TimeOfFlightLookupTable
TimeOfFlightLookupTableFilename = tof_t.TimeOfFlightLookupTableFilename


SampleRun = NewType('SampleRun', int)
"""Sample run; a run with a sample in the beam."""

DarkBackgroundRun = NewType('DarkBackgroundRun', int)
"""Dark background run; a run with no sample in the beam, and the shutter closed, to
measure the dark current of the detector."""

OpenBeamRun = NewType('OpenBeamRun', int)
"""Open beam run; a run with no sample in the beam, and the shutter open, to measure the
beam profile."""

BeamMonitor1 = NewType('BeamMonitor1', int)
"""Beam monitor number 1"""

BeamMonitor2 = NewType('BeamMonitor2', int)
"""Beam monitor number 2"""

BeamMonitor3 = NewType('BeamMonitor3', int)
"""Beam monitor number 3"""

BeamMonitor4 = NewType('BeamMonitor4', int)
"""Beam monitor number 4"""

RunType = reduce_t.RunType
MonitorType = reduce_t.MonitorType


class CoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """
    Graph of coordinate transformations used to compute the wavelength from the
    time-of-flight.
    """


class WavelengthDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts with wavelength information."""


MaskingRules = NewType('MaskingRules', MappingProxyType[str, Callable])
"""Functions to mask different dimensions of Odin data."""


class CorrectedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with masks."""


del sc, sciline, NewType
