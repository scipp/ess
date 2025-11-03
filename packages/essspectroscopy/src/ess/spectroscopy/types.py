# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for spectroscopy."""

from typing import Any, NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce import time_of_flight
from ess.reduce.nexus import types as reduce_t

# NeXus types

EmptyDetector = reduce_t.EmptyDetector
RawDetector = reduce_t.RawDetector
DetectorPositionOffset = reduce_t.DetectorPositionOffset
GravityVector = reduce_t.GravityVector
Filename = reduce_t.Filename
RawMonitor = reduce_t.RawMonitor
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

TofDetector = time_of_flight.TofDetector
TofMonitor = time_of_flight.TofMonitor
MonitorLtotal = time_of_flight.MonitorLtotal
PulseStride = time_of_flight.PulseStride
PulseStrideOffset = time_of_flight.PulseStrideOffset
PulsePeriod = time_of_flight.PulsePeriod
TimeOfFlightLookupTable = time_of_flight.TimeOfFlightLookupTable
TimeOfFlightLookupTableFilename = time_of_flight.TimeOfFlightLookupTableFilename

# Custom types


class Analyzer(sciline.Scope[RunType, sc.DataGroup[Any]], sc.DataGroup[Any]):
    """A single wavelength analyzer loaded from an NXcrystal.

    Can be obtained from ``Analyzers``.
    """


class Analyzers(sciline.Scope[RunType, sc.DataGroup[Any]], sc.DataGroup[Any]):
    """All wavelength analyzers loaded from a NXcrystals."""


class DataAtSample(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class DataGroupedByRotation(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class DetectorCountsWithQ(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class QDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class EnergyQDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class SampleAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Rotation angle of the sample, possibly as a function of time.

    For BIFROST, this is angle "a3".
    """


class InstrumentAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Rotation angle of the instrument, possibly as a function of time.

    For BIFROST, this is angle "a4".
    """


MonitorCoordTransformGraph = NewType('MonitorCoordTransformGraph', dict)


NXspeFileName = NewType('NXspeFileName', str)
NXspeFileNames = NewType('NXspeFileNames', list[NXspeFileName])


class PrimarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


class SecondarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


InelasticCoordTransformGraph = NewType('InelasticCoordTransformGraph', dict)


class ElasticCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


SQWBinSizes = NewType('SQWBinSizes', dict[str, int])


class WavelengthMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...
