# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

import sciline
import scipp as sc

# 1  TypeVars used to parametrize the generic parts of the workflow

# 1.1  Run types
BackgroundRun = NewType('BackgroundRun', int)
"""Background run""" ""
DirectRun = NewType('DirectRun', int)
"""Direct run"""
SampleRun = NewType('SampleRun', int)
"""Sample run"""
RunType = TypeVar('RunType', BackgroundRun, DirectRun, SampleRun)
"""TypeVar used for specifying BackgroundRun, DirectRun or SampleRun"""

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
"""TypeVar used for specifying Numerator or Denominator of IofQ""" ""

# 2  Workflow parameters

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


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    """Name of Incident|Transmission monitor in NeXus file"""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of BackgroundRun|DirectRun|SampleRun"""


# 3  Workflow (intermediate) results

DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
"""Detector edge mask"""
SampleHolderMask = NewType('SampleHolderMask', sc.Variable)
"""Sample holder mask"""

DirectBeam = NewType('DirectBeam', sc.DataArray)
"""Direct beam"""

TransmissionFraction = NewType('TransmissionFraction', sc.DataArray)
"""
Transmission fraction for inspection purposes.

The IofQ computation does not use this, it uses the monitors directly.
"""


CleanDirectBeam = NewType('CleanDirectBeam', sc.DataArray)
"""Direct beam after resampling to required wavelength bins"""


class SolidAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Solid angle of detector pixels seen from sample position"""


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data"""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied"""


# TODO Need Scope with multiple params, see scipp/sciline#42
@dataclass
class Clean(Generic[RunType, IofQPart]):
    """
    Prerequisite for IofQ numerator or denominator.

    This can either be the sample or background counts, converted to wavelength,
    or the respective normalization terms computed from the respective solid angle,
    direct beam, and monitors.
    """

    value: sc.DataArray


@dataclass
class CleanMasked(Generic[RunType, IofQPart]):
    """Result of applying wavelength masking to :py:class:`Clean`"""

    value: sc.DataArray


@dataclass
class CleanQ(Generic[RunType, IofQPart]):
    """Result of converting :py:class:`CleanMasked` to Q"""

    value: sc.DataArray


@dataclass
class CleanSummedQ(Generic[RunType, IofQPart]):
    """Result of histogramming/binning :py:class:`CleanQ` over all pixels into Q bins"""

    value: sc.DataArray


class IofQ(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """I(Q)"""


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', sc.DataArray)
"""I(Q) with background (given by I(Q) of the background run) subtracted"""


@dataclass
class RawMonitor(Generic[RunType, MonitorType]):
    """Raw monitor data"""

    value: sc.DataArray


@dataclass
class WavelengthMonitor(Generic[RunType, MonitorType]):
    """Monitor data converted to wavelength"""

    value: sc.DataArray


@dataclass
class CleanMonitor(Generic[RunType, MonitorType]):
    """Monitor data cleaned of background counts"""

    value: sc.DataArray
