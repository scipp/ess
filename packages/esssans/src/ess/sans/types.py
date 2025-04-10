# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
This modules defines the domain types uses in esssans.

The domain types are used to define parameters and to request results from a Sciline
pipeline."""

from collections.abc import Sequence
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

BackgroundRun = reduce_t.BackgroundRun
CalibratedDetector = reduce_t.CalibratedDetector
CalibratedMonitor = reduce_t.CalibratedMonitor
DetectorData = reduce_t.DetectorData
DetectorPositionOffset = reduce_t.DetectorPositionOffset
EmptyBeamRun = reduce_t.EmptyBeamRun
Filename = reduce_t.Filename
Incident = reduce_t.IncidentMonitor
MonitorData = reduce_t.MonitorData
MonitorPositionOffset = reduce_t.MonitorPositionOffset
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent
SampleRun = reduce_t.SampleRun
ScatteringRunType = reduce_t.ScatteringRunType
Transmission = reduce_t.TransmissionMonitor
TransmissionRun = reduce_t.TransmissionRun

DetectorBankSizes = reduce_t.DetectorBankSizes
NeXusDetectorName = reduce_t.NeXusDetectorName

MonitorType = TypeVar('MonitorType', Incident, Transmission)
RunType = TypeVar(
    'RunType',
    SampleRun,
    BackgroundRun,
    EmptyBeamRun,
    TransmissionRun[SampleRun],
    TransmissionRun[BackgroundRun],
)

UncertaintyBroadcastMode = _UncertaintyBroadcastMode

# 1.3  Numerator and denominator of IofQ
Numerator = NewType('Numerator', sc.DataArray)
"""Numerator of IofQ"""
Denominator = NewType('Denominator', sc.DataArray)
"""Denominator of IofQ"""
IofQPart = TypeVar('IofQPart', Numerator, Denominator)
"""TypeVar used for specifying Numerator or Denominator of IofQ"""

# 1.4  Entry paths in NeXus files
PixelShapePath = NewType('PixelShapePath', str)
"""
Name of the entry where the pixel shape is stored in the NeXus file
"""

TransformationPath = NewType('TransformationPath', str)
"""
Name of the entry where the transformation computed from the
transformation chain is stored, for the detectors and the monitors
"""

# 2  Workflow parameters

ReturnEvents = NewType('ReturnEvents', bool)
"""Whether to return events in the output I(Q)"""

WavelengthBins = NewType('WavelengthBins', sc.Variable)
"""Wavelength binning"""

WavelengthBands = NewType('WavelengthBands', sc.Variable | None)
"""Wavelength bands. Typically a single band, set to first and last value of
WavelengthBins.

The wavelength bands can however be used to compute the scattering cross section inside
multiple wavelength bands. In this case, the wavelength bands must be either one- or
two-dimensional.

In the case of a one-dimensional array, the values represent a set of non-overlapping
bins in the wavelength dimension.

In the case of a two-dimensional array, the values represent a set of (possibly
overlapping or non-contiguous) wavelength bands, with the first dimension being the
band index and the second dimension being the wavelength. For each band, there must be
two wavelength values defining the start and end wavelength of the band.
"""

ProcessedWavelengthBands = NewType('ProcessedWavelengthBands', sc.Variable)
"""Processed wavelength bands, as a two-dimensional variable, with the first dimension
being the band index and the second dimension being the wavelength. For each band, there
must be two wavelength values defining the start and end wavelength of the band."""


QBins = NewType('QBins', sc.Variable)
"""Q binning used when computing IofQ"""

QxBins = NewType('QxBins', sc.Variable)
"""Qx binning used when computing IofQxy"""

QyBins = NewType('QyBins', sc.Variable)
"""Qy binning used when computing IofQxy"""

NonBackgroundWavelengthRange = NewType(
    'NonBackgroundWavelengthRange', sc.Variable | None
)
"""Range of wavelengths that are not considered background in the monitor"""

DirectBeamFilename = NewType('DirectBeamFilename', str)
"""Filename of direct beam correction"""

BeamCenter = NewType('BeamCenter', sc.Variable)
"""Beam center, may be set directly or computed using beam-center finder"""

WavelengthMask = NewType('WavelengthMask', sc.DataArray | None)
"""Optional wavelength mask"""

CorrectForGravity = NewType('CorrectForGravity', bool)
"""Whether to correct for gravity when computing wavelength and Q."""

DimsToKeep = NewType('DimsToKeep', Sequence[str])
"""Dimensions that should not be reduced and thus still be present in the final
I(Q) result (this is typically the layer dimension)."""

OutFilename = NewType('OutFilename', str)
"""Filename of the output"""


PixelMaskFilename = NewType('PixelMaskFilename', str)

DetectorIDs = NewType('DetectorIDs', sc.Variable)
"""1-D variable listing all detector IDs."""

MaskedDetectorIDs = NewType('MaskedDetectorIDs', sc.Variable)
"""1-D variable listing all masked detector IDs."""


DetectorMasks = NewType('DetectorMasks', dict[str, sc.Variable])
"""Masks for detector pixels"""


# 3  Workflow (intermediate) results


DirectBeam = NewType('DirectBeam', sc.DataArray | None)
"""Direct beam"""


class TransmissionFraction(
    sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray
):
    """Transmission fraction"""


CleanDirectBeam = NewType('CleanDirectBeam', sc.DataArray)
"""Direct beam after resampling to required wavelength bins, else and array of ones."""


class DetectorPixelShape(sciline.Scope[ScatteringRunType, sc.DataGroup], sc.DataGroup):
    """Geometry of the detector from description in nexus file."""


class SolidAngle(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Solid angle of detector pixels seen from sample position"""


class MaskedSolidAngle(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Same as :py:class:`SolidAngle`, but with pixel masks applied"""


class TofData(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Data with a time-of-flight coordinate"""


class TofMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataGroup], sc.DataGroup
):
    """Monitor data with a time-of-flight coordinate"""


PixelMask = NewType('PixelMask', sc.Variable)


class MaskedData(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied"""


class MonitorTerm(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Monitor-dependent factor of the Normalization term (numerator) for IofQ."""


class CleanWavelength(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """
    Prerequisite for IofQ numerator or denominator.

    This can either be the sample or background counts, converted to wavelength,
    or the respective normalization terms computed from the respective solid angle,
    direct beam, and monitors.
    """


class WavelengthScaledQ(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of applying wavelength scaling/masking to :py:class:`CleanSummedQ`"""


class WavelengthScaledQxy(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of applying wavelength scaling/masking to :py:class:`CleanSummedQxy`"""


class CleanQ(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of converting :py:class:`CleanWavelengthMasked` to Q"""


class CleanQxy(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of converting :py:class:`CleanWavelengthMasked` to Qx and Qy"""


class CleanSummedQ(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of histogramming/binning :py:class:`CleanQ` over all pixels into Q bins"""


class CleanSummedQxy(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of histogramming/binning :py:class:`CleanQxy` over all pixels into Qx and
    Qy bins"""


class ReducedQ(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of reducing :py:class:`CleanSummedQ` over the wavelength dimensions"""


class ReducedQxy(
    sciline.Scope[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of reducing :py:class:`CleanSummedQxy` over the wavelength dimensions"""


class IofQ(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """I(Q)"""


class IofQxy(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """I(Qx, Qy)"""


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', sc.DataArray)
"""I(Q) with background (given by I(Q) of the background run) subtracted"""

BackgroundSubtractedIofQxy = NewType('BackgroundSubtractedIofQxy', sc.DataArray)
"""I(Qx, Qy) with background (given by I(Qx, Qy) of the background run) subtracted"""


class WavelengthMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data converted to wavelength"""


class CleanMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data cleaned of background counts"""


# 4  Metadata

Beamline = reduce_t.Beamline
Measurement = reduce_t.Measurement
