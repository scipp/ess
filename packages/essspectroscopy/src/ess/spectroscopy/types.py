# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Any, NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce import time_of_flight
from ess.reduce.nexus import types as reduce_t


def make_scipp_named_typer(scipp_type):
    def typer(named: str) -> type[scipp_type]:
        return NewType(named, scipp_type)

    return typer


variable_type = make_scipp_named_typer(sc.Variable)
data_array_type = make_scipp_named_typer(sc.DataArray)

AllNeXusComponents = reduce_t.AllNeXusComponents
Analyzers = reduce_t.Analyzers
CalibratedBeamline = reduce_t.CalibratedBeamline
CalibratedDetector = reduce_t.CalibratedDetector
Choppers = reduce_t.Choppers
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

PulseStride = time_of_flight.PulseStride
PulsePeriod = time_of_flight.PulsePeriod
TimeOfFlightLookupTable = time_of_flight.TimeOfFlightLookupTable


class InstrumentAngles(
    sciline.Scope[RunType, sc.DataGroup[sc.DataArray]], sc.DataGroup[sc.DataArray]
):
    """Instrument angles for the sample orientation as a function of time."""


class Analyzer(sciline.Scope[RunType, sc.DataGroup[Any]], sc.DataGroup[Any]): ...


AnalyzerPosition = variable_type('AnalyzerPosition')
DetectorPosition = variable_type('DetectorPosition')
SamplePosition = variable_type('SamplePosition')
SourcePosition = variable_type('SourcePosition')
AnalyzerOrientation = variable_type('AnalyzerOrientation')
SampleAnalyzerVector = variable_type('SampleAnalyzerVector')
AnalyzerDetectorVector = variable_type('AnalyzerDetectorVector')
SampleAnalyzerDirection = variable_type('SampleAnalyzerDirection')
ReciprocalLatticeVectorAbsolute = variable_type('ReciprocalLatticeVectorAbsolute')
AnalyzerDspacing = variable_type('Dspacing')
IncidentDirection = variable_type('IncidentDirection')
IncidentEnergy = variable_type('IncidentEnergy')
FinalDirection = variable_type('FinalDirection')
FinalSlowness = variable_type('FinalSlowness')
FinalWavelength = variable_type('FinalWavelength')
FinalWavenumber = variable_type('FinalWavenumber')
FinalWavevector = variable_type('FinalWavevector')
FinalEnergy = variable_type('FinalEnergy')
SourceSamplePathLength = variable_type('SourceSamplePathLength')
SampleDetectorPathLength = variable_type('SampleDetectorPathLength')
SampleDetectorFlightTime = variable_type('SampleDetectorFlightTime')
SampleFrameTime = variable_type('SampleFrameTime')
DetectorFrameTime = variable_type('DetectorFrameTime')
DetectorTime = variable_type('DetectorTime')
SourceTime = variable_type('SourceTime')

SampleName = NewType('SampleName', str)
SourceName = NewType('SourceName', str)

MonitorName = NewType('MonitorName', str)
MonitorPosition = variable_type('MonitorPosition')
SourceMonitorPathLength = variable_type('SourceMonitorPathLength')
WavelengthMonitor = data_array_type('WavelengthMonitor')

LabMomentumTransfer = variable_type('LabMomentumTransfer')

SampleTableAngle = variable_type("SampleTableAngle")
TableMomentumTransfer = variable_type('TableMomentumTransfer')

SampleMomentumTransfer = variable_type('SampleMomentumTransfer')
SampleMomentumTransferX = variable_type("SampleMomentumTransferX")
SampleMomentumTransferY = variable_type("SampleMomentumTransferY")
SampleMomentumTransferZ = variable_type("SampleMomentumTransferZ")

EnergyTransfer = variable_type('EnergyTransfer')

DetectorGeometricA4 = variable_type("DetectorGeometricA4")

SlothEvents = data_array_type('SlothEvents')
SlothBins = variable_type('SlothBins')
NormSlothEvents = data_array_type('NormSlothEvents')

WavelengthEvents = data_array_type('WavelengthEvents')
WavelengthBins = variable_type('WavelengthBins')
NormWavelengthEvents = data_array_type('NormWavelengthEvents')

NXspeFileName = NewType('NXspeFileName', str)
NXspeFileNames = NewType('NXspeFileNames', list[NXspeFileName])


class TofMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...


class TofData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class EnergyData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class SecondarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


class PrimarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


InelasticCoordTransformGraph = NewType('InelasticCoordTransformGraph', dict)
MonitorCoordTransformGraph = NewType('MonitorCoordTransformGraph', dict)


class BeamlineWithSpectrometerCoords(
    sciline.Scope[RunType, sc.DataArray], sc.DataArray
): ...


class DataAtSample(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class DataGroupedByRotation(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...
