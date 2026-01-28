# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from types import MappingProxyType
from typing import NewType

import sciline
import scipp as sc
from ess.reduce.nexus import types as reduce_t
from ess.reduce.time_of_flight import types as tof_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

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

UncertaintyBroadcastMode = _UncertaintyBroadcastMode


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
    """Corrected detector counts with masking applied."""


class FluxNormalizedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts normalized to proton charge."""


class BackgroundSubtractedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts with dark background subtracted."""


NormalizedImage = NewType('NormalizedImage', sc.DataArray)
"""Final image: background-subtracted sample run divided by background-subtracted open
beam run."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Proton charge data for a run."""


class ExposureTime(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Exposure time of each frame recorded by the camera detector."""


del sc, sciline, NewType
