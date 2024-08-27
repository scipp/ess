from typing import Type, NewType
from scipp import Variable
from choppera.primary import PrimarySpectrometer


def variable_type(named: str) -> Type[Variable]:
    return NewType(named, Variable)


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
IncidentWavevectorDirection = variable_type('IncidentWavevectorDirection')
IncidentWavenumber = variable_type('IncidentWavenumber')
IncidentWavevector = variable_type('IncidentWavevector')
IncidentEnergy = variable_type('IncidentEnergy')
FinalWavevectorDirection = variable_type('FinalWavevectorDirection')
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

LabMomentumTransfer = variable_type('MonmentumTransfer')
LabMomentumTransferX = variable_type("LabMomentumTransferX")
LabMomentumTransferY = variable_type("LabMomentumTransferY")
LabMomentumTransferZ = variable_type("LabMomentumTransferZ")

EnergyTransfer = variable_type('EnergyTransfer')