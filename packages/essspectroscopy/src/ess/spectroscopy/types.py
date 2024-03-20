from typing import Type
from scipp import Variable


def variable_type(named: str) -> Type[Variable]:
    from typing import NewType
    return NewType(named, Variable)


SamplePosition = variable_type('SamplePosition')
AnalyzerPosition = variable_type('AnalyzerPosition')
DetectorPosition = variable_type('DetectorPosition')
AnalyzerOrientation = variable_type('AnalyzerOrientation')
SampleAnalyzerVector = variable_type('SampleAnalyzerVector')
AnalyzerDetectorVector = variable_type('AnalyzerDetectorVector')
SampleAnalyzerDirection = variable_type('SampleAnalyzerDirection')
ReciprocalLatticeVectorAbsolute = variable_type('ReciprocalLatticeVectorAbsolute')
ReciprocalLatticeSpacing = variable_type('ReciprocalLatticeSpacing')
WavevectorDirection = variable_type('WavevectorDirection')
Wavenumber = variable_type('Wavenumber')
Wavevector = variable_type('Wavevector')
SampleDetectorPathLength = variable_type('SampleDetectorPathLength')
SampleDetectorFlightTime = variable_type('SampleDetectorFlightTime')
SampleTime = variable_type('SampleTime')
DetectorTime = variable_type('DetectorTime')
SourceTime = variable_type('SourceTime')
