# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import NewType

from choppera.primary import PrimarySpectrometer
from scipp import DataArray, Variable


def make_scipp_named_typer(scipp_type):
    def typer(named: str) -> type[scipp_type]:
        return NewType(named, scipp_type)

    return typer


variable_type = make_scipp_named_typer(Variable)
data_array_type = make_scipp_named_typer(DataArray)


NeXusFileName = NewType('NeXusFileName', str)

SourcePosition = variable_type('SourcePosition')
SamplePosition = variable_type('SamplePosition')
AnalyzerPosition = variable_type('AnalyzerPosition')
DetectorPosition = variable_type('DetectorPosition')
AnalyzerOrientation = variable_type('AnalyzerOrientation')
SampleAnalyzerVector = variable_type('SampleAnalyzerVector')
AnalyzerDetectorVector = variable_type('AnalyzerDetectorVector')
SampleAnalyzerDirection = variable_type('SampleAnalyzerDirection')
ReciprocalLatticeVectorAbsolute = variable_type('ReciprocalLatticeVectorAbsolute')
ReciprocalLatticeSpacing = variable_type('ReciprocalLatticeSpacing')
IncidentDirection = variable_type('IncidentDirection')
IncidentSlowness = variable_type('IncidentSlowness')
IncidentWavelength = variable_type('IncidentWavelength')
IncidentWavenumber = variable_type('IncidentWavenumber')
IncidentWavevector = variable_type('IncidentWavevector')
IncidentEnergy = variable_type('IncidentEnergy')
FinalDirection = variable_type('FinalDirection')
FinalSlowness = variable_type('FinalSlowness')
FinalWavelength = variable_type('FinalWavelength')
FinalWavenumber = variable_type('FinalWavenumber')
FinalWavevector = variable_type('FinalWavevector')
FinalEnergy = variable_type('FinalEnergy')
SourceSamplePathLength = variable_type('SourceSamplePathLength')
SourceSampleFlightTime = variable_type('SourceSampleFlightTime')
SampleDetectorPathLength = variable_type('SampleDetectorPathLength')
SampleDetectorFlightTime = variable_type('SampleDetectorFlightTime')
SampleFrameTime = variable_type('SampleFrameTime')
SampleTime = variable_type('SampleTime')
DetectorFrameTime = variable_type('DetectorFrameTime')
DetectorTime = variable_type('DetectorTime')
SourceTime = variable_type('SourceTime')

SourceName = NewType('SourceName', str)
SampleName = NewType('SampleName', str)
SourceFrequency = variable_type('SourceFrequency')
SourceDuration = variable_type('SourceDuration')
SourceDelay = variable_type('SourceDelay')
SourceVelocities = variable_type('SourceVelocities')

PrimarySpectrometerObject = NewType('PrimarySpectrometerObject', PrimarySpectrometer)

FocusComponentName = NewType('FocusComponentName', str)
FocusComponentNames = NewType('FocusComponentNames', list[FocusComponentName])
PrimaryFocusDistance = variable_type('PrimaryFocusDistance')
PrimaryFocusTime = variable_type('PrimaryFocusTime')

SourceMonitorPathLength = variable_type('SourceMonitorPathLength')
SourceMonitorFlightTime = variable_type('SourceMonitorFlightTime')
FrameTimeMonitor = data_array_type('FrameTimeMonitor')
WallTimeMonitor = data_array_type('WallTimeMonitor')
SlownessMonitor = data_array_type('SlownessMonitor')

MonitorNormalisation = variable_type('MonitorNormalisation')

LabMomentumTransfer = variable_type('LabMomentumTransfer')
LabMomentumTransferX = variable_type("LabMomentumTransferX")
LabMomentumTransferY = variable_type("LabMomentumTransferY")
LabMomentumTransferZ = variable_type("LabMomentumTransferZ")

SampleTableAngle = variable_type("SampleTableAngle")
TableMomentumTransfer = variable_type('TableMomentumTransfer')
TableMomentumTransferX = variable_type("TableMomentumTransferX")
TableMomentumTransferY = variable_type("TableMomentumTransferY")
TableMomentumTransferZ = variable_type("TableMomentumTransferZ")

SampleMomentumTransfer = variable_type('SampleMomentumTransfer')
SampleMomentumTransferX = variable_type("SampleMomentumTransferX")
SampleMomentumTransferY = variable_type("SampleMomentumTransferY")
SampleMomentumTransferZ = variable_type("SampleMomentumTransferZ")

EnergyTransfer = variable_type('EnergyTransfer')

# Debugging types, likely to be removed
DetectorGeometricA4 = variable_type("DetectorGeometricA4")
