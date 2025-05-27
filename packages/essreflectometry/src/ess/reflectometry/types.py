# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Any, NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t

SampleRun = reduce_t.SampleRun
ReferenceRun = NewType("ReferenceRun", int)
RunType = TypeVar("RunType", ReferenceRun, SampleRun)

Beamline = reduce_t.Beamline
CalibratedDetector = reduce_t.CalibratedDetector
DetectorData = reduce_t.DetectorData
DetectorPositionOffset = reduce_t.DetectorPositionOffset
Filename = reduce_t.Filename
Measurement = reduce_t.Measurement
NeXusComponent = reduce_t.NeXusComponent
NeXusDetectorName = reduce_t.NeXusDetectorName
Position = reduce_t.Position

CoordTransformationGraph = NewType("CoordTransformationGraph", dict)


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


class ThetaBins(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Binning in theta that takes into consideration that some
    detector pixels have the same theta value."""


class RawSampleRotation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The rotation of the sample registered in the NeXus file."""


class SampleRotation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The rotation of the sample relative to the center of the incoming beam."""


class SampleRotationOffset(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The difference between the true slope of the sample surface
    and the sample rotation value in the file."""


class DetectorRotation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """The rotation of the detector relative to the horizon"""


class BeamSize(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Full-Width-Half-maximum of the incoming beam."""


class DetectorSpatialResolution(sciline.Scope[RunType, sc.Variable], sc.Variable):
    # TODO what is the definition of this?
    """Spatial resolution of the detector."""


class SampleSize(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Diameter of the sample. If None it is assumed to be the same as the reference."""


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


WavelengthThetaFigure = NewType("WavelengthThetaFigure", Any)
WavelengthZIndexFigure = NewType("WavelengthZIndexFigure", Any)
QThetaFigure = NewType("QThetaFigure", Any)
ReflectivityDiagnosticsView = NewType("ReflectivityDiagnosticsView", Any)
