# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
This modules defines the domain types uses in esssans.

The domain types are used to define parameters and to request results from a Sciline
pipeline."""
from collections.abc import Sequence
from enum import Enum
from typing import NewType, TypeVar

import sciline
import scipp as sc

# 1  TypeVars used to parametrize the generic parts of the workflow

# 1.1  Run types
BackgroundRun = NewType('BackgroundRun', int)
"""Background run"""
EmptyBeamRun = NewType('EmptyBeamRun', int)
"""Run where the sample holder was empty (sometimes called 'direct run')"""
SampleRun = NewType('SampleRun', int)
"""Sample run"""

RunType = TypeVar(
    'RunType',
    BackgroundRun,
    EmptyBeamRun,
    SampleRun,
)
"""TypeVar used for specifying BackgroundRun, EmptyBeamRun or SampleRun"""


class TransmissionRun(sciline.Scope[RunType, int], int):
    """Mapping between RunType and transmission run.
    In the case where no transmission run is provided, the transmission run should be
    the same as the measurement (sample or background) run."""


# 1.2  Monitor types
Incident = NewType('Incident', int)
"""Incident monitor"""
Transmission = NewType('Transmission', int)
"""Transmission monitor"""
MonitorType = TypeVar('MonitorType', Incident, Transmission)
"""TypeVar used for specifying Incident or Transmission monitor type"""

# 1.3  Numerator and denominator of IofQ
Numerator = NewType('Numerator', sc.DataArray)
"""Numerator of IofQ"""
Denominator = NewType('Denominator', sc.DataArray)
"""Denominator of IofQ"""
IofQPart = TypeVar('IofQPart', Numerator, Denominator)
"""TypeVar used for specifying Numerator or Denominator of IofQ"""

# 1.4  Entry paths in Nexus files
NexusInstrumentPath = NewType('NexusInstrumentPath', str)

NexusSampleName = NewType('NexusSampleName', str)

NexusSourceName = NewType('NexusSourceName', str)

NexusDetectorName = NewType('NexusDetectorName', str)

TransformationChainPath = NewType('TransformationChainName', str)

# 2  Workflow parameters

UncertaintyBroadcastMode = Enum(
    'UncertaintyBroadcastMode', ['drop', 'upper_bound', 'fail']
)
"""
Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""

WavelengthBins = NewType('WavelengthBins', sc.Variable)
"""Wavelength binning"""

WavelengthBands = NewType('WavelengthBands', sc.Variable)
"""Wavelength bands. Typically a single band, set to first and last value of
WavelengthBins."""

QBins = NewType('QBins', sc.Variable)
"""Q binning"""

NonBackgroundWavelengthRange = NewType('NonBackgroundWavelengthRange', sc.Variable)
"""Range of wavelengths that are not considered background in the monitor"""

DirectBeamFilename = NewType('DirectBeamFilename', str)
"""Filename of direct beam correction"""

BeamCenter = NewType('BeamCenter', sc.Variable)
"""Beam center, may be set directly or computed using beam-center finder"""

WavelengthMask = NewType('WavelengthMask', sc.DataArray)
"""Optional wavelength mask"""

CorrectForGravity = NewType('CorrectForGravity', bool)
"""Whether to correct for gravity when computing wavelength and Q."""

FinalDims = NewType('FinalDims', Sequence[str])
"""Final dimensions of IofQ"""

OutFilename = NewType('OutFilename', str)
"""Filename of the output"""


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    """Name of Incident|Transmission monitor in NeXus file"""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of BackgroundRun|EmptyBeamRun|SampleRun"""


class RunID(sciline.Scope[RunType, int], int):
    """Sample run ID when multiple runs are used"""


# 3  Workflow (intermediate) results


class SamplePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Sample position"""


class SourcePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Source position"""


class DataWithLogicalDims(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data reshaped to have logical dimensions"""


DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
"""Detector edge mask"""

SampleHolderMask = NewType('SampleHolderMask', sc.Variable)
"""Sample holder mask"""

DirectBeam = NewType('DirectBeam', sc.DataArray)
"""Direct beam"""


class TransmissionFraction(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Transmission fraction"""


CleanDirectBeam = NewType('CleanDirectBeam', sc.DataArray)
"""Direct beam after resampling to required wavelength bins"""


class DetectorPixelShape(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Geometry of the detector from description in nexus file."""


class LabFrameTransform(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Coordinate transformation from detector local coordinates
    to the sample frame of reference."""


class SolidAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Solid angle of detector pixels seen from sample position"""


class LoadedDetectorContents(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """The entire contents of a loaded detector data"""


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data"""


class UnmergedPatchedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Patched with added sample and source positions data"""


class TofData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data converted to time-of-flight"""


class UnmergedRawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Single raw data run"""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied"""


class CalibratedMaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied and calibrated pixel positions"""


class NormWavelengthTerm(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Normalization term (numerator) for IofQ before scaling with solid-angle."""


class CleanMasked(
    sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Data with calibrated pixel positions and pixel-specific masks applied"""


class CleanWavelength(
    sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray
):
    """
    Prerequisite for IofQ numerator or denominator.

    This can either be the sample or background counts, converted to wavelength,
    or the respective normalization terms computed from the respective solid angle,
    direct beam, and monitors.
    """


class CleanWavelengthMasked(
    sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of applying wavelength masking to :py:class:`CleanWavelength`"""


class CleanQ(sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray):
    """Result of converting :py:class:`CleanWavelengthMasked` to Q"""


class CleanSummedQ(
    sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of histogramming/binning :py:class:`CleanQ` over all pixels into Q bins"""


class IofQ(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """I(Q)"""


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', sc.DataArray)
"""I(Q) with background (given by I(Q) of the background run) subtracted"""


class LoadedMonitorContents(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataGroup], sc.DataGroup
):
    """The entire contents of a loaded monitor"""


class UnmergedRawMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Unmerged raw monitor data"""


class RawMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data"""


class UnmergedPatchedMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Patched monitor data with source position"""


class TofMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data converted to time-of-flight"""


class WavelengthMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data converted to wavelength"""


class CleanMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data cleaned of background counts"""


# 4  Metadata

RunTitle = NewType('RunTitle', str)
"""Title of the run."""

RunNumber = NewType('RunNumber', int)
"""Run number."""
