# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from types import MappingProxyType
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.reduce.time_of_flight import types as tof_t

# 1 TypeVars used to parametrize the generic parts of the workflow

DetectorData = reduce_t.DetectorData
Filename = reduce_t.Filename
MonitorData = reduce_t.MonitorData
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent

DetectorLtotal = tof_t.DetectorLtotal
DetectorTofData = tof_t.DetectorTofData
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

RunType = TypeVar("RunType", SampleRun, DarkBackgroundRun, OpenBeamRun)
MonitorType = TypeVar(
    "MonitorType", BeamMonitor1, BeamMonitor2, BeamMonitor3, BeamMonitor4
)

CoordTransformGraph = NewType("CoordTransformGraph", dict)
"""
Graph of coordinate transformations used to compute the wavelength from the
time-of-flight.
"""


class CountsWavelength(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts with wavelength information."""


MaskingRules = NewType('MaskingRules', MappingProxyType[str, Callable])
"""Functions to mask different dimensions of Odin data."""


class CountsMasked(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with masks."""


ImageDetectorName = NewType('ImageDetectorName', str)
"""Histogram mode detector name."""

ImageKeyLogs = NewType('ImageKeyLogs', sc.DataArray)
"""Image key logs."""

RotationMotionSensorName = NewType('RotationMotionSensorName', str)
"""Rotation sensor name."""

RotationLogs = NewType('RotationLogs', sc.DataArray)
"""Rotation logs data."""

HistogramModeDetectorsPath = NewType('HistogramModeDetectorsPath', str)
"""Path to the histogram mode detectors in a nexus file."""

DEFAULT_HISTOGRAM_PATH = HistogramModeDetectorsPath(
    "/entry/instrument/histogram_mode_detectors"
)


del sc, sciline, NewType, TypeVar
