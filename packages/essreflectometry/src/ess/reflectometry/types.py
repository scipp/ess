from typing import NewType, TypeVar

import sciline
import scipp as sc

Reference = NewType('Reference', str)
Sample = NewType('Sample', str)
Run = TypeVar('Run', Reference, Sample)


class NeXusDetectorName(sciline.Scope[Run, str], str):
    """Name of the detector in the nexus file containing the events of the run"""


class DetectorPosition(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Positions of the detector pixels, relative to the source(?), as a 3d-vector"""


class SamplePosition(sciline.Scope[Run, sc.Variable], sc.Variable):
    """The position of the sample relative to the source(?)."""


class IncidentBeam(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Incident beam vector."""


class SpecularReflectionCoordTransformGraph(sciline.Scope[Run, dict], dict):
    """Coordinate transformation graph for specular reflection"""


class RawEvents(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event time data from nexus file,
    binned by `detector_number` (pixel of the detector frame)."""


class RawDetector(sciline.Scope[Run, sc.DataGroup], sc.DataGroup):
    """NXdetector loaded from file"""


class ChopperCorrectedTofEvents(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event time data after correcting tof for choppers."""


class EventData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data with added coordinates such as incident angle (theta),
    wavelength, and momentum transfer (Q)"""


class MaskedEventData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data that has been masked in wavelength and logical detector coordinates"""


class FootprintCorrectedData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data with weights corrected for the footprint of the beam
    on the sample for the incidence angle of the event."""


HistogrammedReference = NewType('HistogrammedReference', sc.DataArray)
'''Intensity distribution of the reference measurement in (z, wavelength)'''

CorrectionMatrix = NewType('CorrectionMatrix', sc.DataArray)
'''Intensity distribution on the detector for a sample with :math`R(Q) = 1`'''

NormalizationFactor = NewType('NormalizationFactor', sc.DataArray)
''':code`CorrectionMatrix` with added coordinate "sample" Q'''

NormalizedIofQ = NewType('NormalizedIofQ', sc.DataArray)
'''Intensity histogram over momentum transfer
normalized by the calibrated reference measurement.'''

QResolution = NewType('QResolution', sc.Variable)
'''Resolution term for the momentum transfer for each bin of QBins.'''


''' Parameters for the workflow '''

QBins = NewType('QBins', sc.Variable)
'''Bins for the momentum transfer histogram.'''

WavelengthBins = NewType('WavelengthBins', sc.Variable)
'''Bins for the wavelength histogram, also used to filter the event data.'''


class PoochFilename(sciline.Scope[Run, str], str):
    """Name of an event data nexus file in the pooch data repository."""


class FilePath(sciline.Scope[Run, str], str):
    """File path of an event data nexus file."""


class SampleRotation(sciline.Scope[Run, sc.Variable], sc.Variable):
    """The rotation of the sample relative to the center of the incoming beam."""


class DetectorRotation(sciline.Scope[Run, sc.Variable], sc.Variable):
    '''The rotation of the detector relative to the horizon'''


class BeamSize(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Full-Width-Half-maximum of the incoming beam."""


class DetectorSpatialResolution(sciline.Scope[Run, sc.Variable], sc.Variable):
    # TODO what is the definition of this?
    """Spatial resolution of the detector."""


class SampleSize(sciline.Scope[Run, sc.Variable], sc.Variable):
    # TODO is this radius or total length?
    """Size of the sample. If None it is assumed to be the same as the reference."""


Gravity = NewType('Gravity', sc.Variable)
"""This parameter determines if gravity is taken into account
when computing the scattering angle and momentum transfer."""


YIndexLimits = NewType('YIndexLimits', tuple[sc.Variable, sc.Variable])
'''Limit of the (logical) 'y' detector pixel index'''


ZIndexLimits = NewType('ZIndexLimits', tuple[sc.Variable, sc.Variable])
'''Limit of the (logical) 'z' detector pixel index'''


ReferenceFilePath = NewType('ReferenceFilePath', str)
'''Path to the cached normalization matrix'''
