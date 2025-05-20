# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for spectroscopy."""

from typing import Any, NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce import time_of_flight
from ess.reduce.nexus import types as reduce_t

# NeXus types

CalibratedBeamline = reduce_t.CalibratedBeamline
CalibratedDetector = reduce_t.CalibratedDetector
DetectorData = reduce_t.DetectorData
DetectorPositionOffset = reduce_t.DetectorPositionOffset
GravityVector = reduce_t.GravityVector
Filename = reduce_t.Filename
MonitorData = reduce_t.MonitorData
NeXusClass = reduce_t.NeXusClass
NeXusComponentLocationSpec = reduce_t.NeXusComponentLocationSpec
NeXusComponent = reduce_t.NeXusComponent
NeXusData = reduce_t.NeXusData
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusFileSpec = reduce_t.NeXusFileSpec
NeXusMonitorName = reduce_t.NeXusName
NeXusTransformation = reduce_t.NeXusTransformation
Position = reduce_t.Position
PreopenNeXusFile = reduce_t.PreopenNeXusFile


SampleRun = reduce_t.SampleRun
VanadiumRun = reduce_t.VanadiumRun

FrameMonitor0 = reduce_t.FrameMonitor0
FrameMonitor1 = reduce_t.FrameMonitor1
FrameMonitor2 = reduce_t.FrameMonitor2
FrameMonitor3 = reduce_t.FrameMonitor3

# Type vars

RunType = TypeVar("RunType", SampleRun, VanadiumRun)
MonitorType = TypeVar(
    "MonitorType",
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
)

# Time-of-flight types

DetectorTofData = time_of_flight.DetectorTofData
MonitorTofData = time_of_flight.MonitorTofData
MonitorLtotal = time_of_flight.MonitorLtotal
PulseStride = time_of_flight.PulseStride
PulseStrideOffset = time_of_flight.PulseStrideOffset
PulsePeriod = time_of_flight.PulsePeriod
TimeOfFlightLookupTable = time_of_flight.TimeOfFlightLookupTable
TimeOfFlightLookupTableFilename = time_of_flight.TimeOfFlightLookupTableFilename

L1Range = NewType("L1Range", tuple[sc.Variable, sc.Variable])
"""
Range (min, max) of the length of the flight path from the source to the sample.

This type corresponds to :class:`ess.reduce.time_of_flight.types.LtotalRange`
for other instruments.
But for indirect geometry spectrometers, we compute time of flight
to the sample, not the detectors.
"""

# Custom types


class Analyzer(sciline.Scope[RunType, sc.DataGroup[Any]], sc.DataGroup[Any]): ...


class BeamlineWithSpectrometerCoords(
    sciline.Scope[RunType, sc.DataArray], sc.DataArray
): ...


class DataAtSample(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class DataGroupedByRotation(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class EnergyData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


InelasticCoordTransformGraph = NewType('InelasticCoordTransformGraph', dict)


class InstrumentAngles(
    sciline.Scope[RunType, sc.DataGroup[sc.DataArray]], sc.DataGroup[sc.DataArray]
):
    """Instrument angles for the sample orientation as a function of time."""


MonitorCoordTransformGraph = NewType('MonitorCoordTransformGraph', dict)


NXspeFileName = NewType('NXspeFileName', str)
NXspeFileNames = NewType('NXspeFileNames', list[NXspeFileName])


class PrimarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


class SecondarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


class WavelengthMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...
