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

DarkBackgroundRun = reduce_t.BackgroundRun
DetectorData = reduce_t.DetectorData
DiskChoppers = reduce_t.DiskChoppers
OpenBeamRun = reduce_t.EmptyBeamRun
Filename = reduce_t.Filename
FrameMonitor0 = reduce_t.FrameMonitor0
FrameMonitor1 = reduce_t.FrameMonitor1
FrameMonitor2 = reduce_t.FrameMonitor2
FrameMonitor3 = reduce_t.FrameMonitor3
MonitorData = reduce_t.MonitorData
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent
SampleRun = reduce_t.SampleRun

DetectorLtotal = tof_t.DetectorLtotal
DetectorTofData = tof_t.DetectorTofData
PulsePeriod = tof_t.PulsePeriod
PulseStride = tof_t.PulseStride
PulseStrideOffset = tof_t.PulseStrideOffset
DistanceResolution = tof_t.DistanceResolution
TimeResolution = tof_t.TimeResolution
LtotalRange = tof_t.LtotalRange
LookupTableRelativeErrorThreshold = tof_t.LookupTableRelativeErrorThreshold
NumberOfSimulatedNeutrons = tof_t.NumberOfSimulatedNeutrons
TimeOfFlightLookupTable = tof_t.TimeOfFlightLookupTable
SimulationResults = tof_t.SimulationResults

RunType = TypeVar("RunType", SampleRun, DarkBackgroundRun, OpenBeamRun)
MonitorType = TypeVar(
    "MonitorType", FrameMonitor0, FrameMonitor1, FrameMonitor2, FrameMonitor3
)

CoordTransformGraph = NewType("CoordTransformGraph", dict)
"""
Graph of coordinate transformations used to compute the wavelength from the
time-of-flight.
"""


class DetectorWavelengthData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with wavelength coordinate."""


MaskingRules = NewType('MaskingRules', MappingProxyType[str, Callable])
"""Functions to mask different dimensions of Odin data."""


class MaskedDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
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
