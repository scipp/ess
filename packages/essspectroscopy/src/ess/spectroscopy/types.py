# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
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

# Type vars

# Include BackgroundRun because a single constraint is not allowed.
# We will eventually have more than one...
RunType = TypeVar("RunType", SampleRun, reduce_t.BackgroundRun)
# Monitor types include all monitors used by instrument packages.
MonitorType = TypeVar(
    "MonitorType",
    reduce_t.Monitor1,
    reduce_t.Monitor2,
    reduce_t.Monitor3,
    reduce_t.Monitor4,
)

# Time-of-flight types

PulseStride = time_of_flight.PulseStride
PulsePeriod = time_of_flight.PulsePeriod
TimeOfFlightLookupTable = time_of_flight.TimeOfFlightLookupTable

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


class TofData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class TofMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...


class WavelengthMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...
