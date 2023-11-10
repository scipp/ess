from typing import NewType, TypeVar

import sciline
import scipp as sc

Reference = NewType('Reference', str)
Sample = NewType('Sample', str)
Run = TypeVar('Run', Reference, Sample)


class RawData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event time data from nexus file,
    binned by `detector_number` (pixel of the detector frame)."""


class WavelengthData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data with wavelengths computed for every event,
    binned by `detector_number` (pixel of the detector frame)"""


class ThetaData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data with wavelengths and scattering angle computed for each event,
    binned by `detector_number` (pixel of the detector frame)"""


class QData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data with wavelength, incidence angle,
    and momentum transfer computed per event.
    Binned by momentum transfer according to the QBins provider,
    and by `detector_number` (pixel of the detector frame).
    """


class FootprintCorrectedData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Event data with weight corrected for the footprint of the beam
    on the sample for the incidence angle of the event."""


SpecularReflectionCoordTransformGraph = NewType(
    'SpecularReflectionCoordTransformGraph', dict
)


class HistogrammedQData(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Histogram of event weights by momentum transfer and detector_number."""


class IofQ(sciline.Scope[Run, sc.DataArray], sc.DataArray):
    """Normalization of the histogram.
    The normalization for the sample consists of scaling
    with the inverse of the total weight.
    The normalization for the reference consists of
        1. multiplication with the calibration factor
           from the supermirror calibration and
        2. scaling with the inverse of the total weight.
    """


NormalizedIofQ = NewType('NormalizedIofQ', sc.DataArray)
'''Normalized histogram over momentrum transfer and detector number,
normalized by the calibrated reference measurement.'''
QResolution = NewType('QResolution', sc.Variable)
'''Resolution term for the momentum transfer for each bin of QBins.'''


''' Parameters for the workflow '''

QBins = NewType('QBins', sc.Variable)
'''Bins for the momentum transfer histogram.'''


class Filename(sciline.Scope[Run, str], str):
    """Filename of the event data nexus file."""


class SampleRotation(sciline.Scope[Run, sc.Variable], sc.Variable):
    """The rotation of the sample relative to the center of the incoming beam."""


class BeamSize(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Full-Width-Half-maximum of the incoming beam."""


class DetectorSpatialResolution(sciline.Scope[Run, sc.Variable], sc.Variable):
    # TODO what is the definition of this?
    """Spatial resolution of the detector."""


class SampleSize(sciline.Scope[Run, sc.Variable], sc.Variable):
    # TODO is this radius or total length?
    """Size of the sample."""


Gravity = NewType('Gravity', sc.Variable)
"""This parameter determines if gravity is taken into account
when computing the scattering angle and momentum transfer."""
