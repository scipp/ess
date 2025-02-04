from typing import NewType, TypeVar

import sciline
import scipp as sc

ReferenceRun = NewType("ReferenceRun", str)
SampleRun = NewType("SampleRun", str)
RunType = TypeVar("RunType", ReferenceRun, SampleRun)


class NeXusDetectorName(sciline.Scope[RunType, str], str):
    """Name of the detector in the nexus file containing the events of the RunType"""


class DetectorPosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Positions of the detector pixels, relative to the source(?), as a 3d-vector"""


class SamplePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The position of the sample relative to the source(?)."""


class SpecularReflectionCoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """Coordinate transformation graph for specular reflection"""


class RawDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Event time data from nexus file,
    binned by `detector_number` (pixel of the detector frame)."""


class LoadedNeXusDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """NXdetector loaded from file"""


class ReducibleData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Event data with common coordinates added"""


ReducedReference = NewType("ReducedReference", sc.DataArray)
"""Intensity distribution on the detector for a sample with :math`R(Q) = 1`"""

Reference = NewType("Reference", sc.DataArray)
""":code`ReducedReference` histogrammed in sample ``Q``"""

Sample = NewType("Sample", sc.DataArray)
""":code`Sample` measurement prepared for reduction"""

ReflectivityOverQ = NewType("ReflectivityOverQ", sc.DataArray)
"""Intensity histogram over momentum transfer
normalized by the calibrated reference measurement."""

ReflectivityOverZW = NewType("ReflectivityOverZW", sc.DataArray)
"""Intensity histogram over z- and wavelength- grid.
normalized by the calibrated reference measurement."""

QResolution = NewType("QResolution", sc.Variable)
"""Resolution term for the momentum transfer for each bin of QBins."""


""" Parameters for the workflow """

QBins = NewType("QBins", sc.Variable)
"""Bins for the momentum transfer histogram."""

WavelengthBins = NewType("WavelengthBins", sc.Variable)
"""Bins for the wavelength histogram, also used to filter the event data."""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of an event data nexus file."""


class SampleRotation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The rotation of the sample relative to the center of the incoming beam."""


class DetectorRotation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The rotation of the detector relative to the horizon"""


class BeamSize(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Full-Width-Half-maximum of the incoming beam."""


class DetectorSpatialResolution(sciline.Scope[RunType, sc.Variable], sc.Variable):
    # TODO what is the definition of this?
    """Spatial resolution of the detector."""


class SampleSize(sciline.Scope[RunType, sc.Variable], sc.Variable):
    # TODO is this radius or total length?
    """Size of the sample. If None it is assumed to be the same as the reference."""


class ProtonCurrent(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Proton current log from file"""


YIndexLimits = NewType("YIndexLimits", tuple[sc.Variable, sc.Variable])
"""Limit of the (logical) 'y' detector pixel index"""


ZIndexLimits = NewType("ZIndexLimits", tuple[sc.Variable, sc.Variable])
"""Limit of the (logical) 'z' detector pixel index"""


BeamDivergenceLimits = NewType("BeamDivergenceLimits", tuple[sc.Variable, sc.Variable])
"""Limit of the beam divergence"""


ReferenceFilePath = NewType("ReferenceFilePath", str)
"""Path to the cached normalization matrix"""
