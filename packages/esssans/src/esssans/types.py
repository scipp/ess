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
"""Background run: the run with only the solvent which the sample is placed in."""
EmptyBeamRun = NewType('EmptyBeamRun', int)
"""Run (sometimes called 'direct run') where the sample holder was empty.
It is used for reading the data from the transmission monitor."""
SampleRun = NewType('SampleRun', int)
"""Sample run: the run with the sample placed in the solvent inside the sample holder.
"""

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

# 1.4  Entry paths in NeXus files
NeXusSampleName = NewType('NeXusSampleName', str)
"""Name of sample entry in NeXus file"""

NeXusSourceName = NewType('NeXusSourceName', str)
"""Name of source entry in NeXus file"""

NeXusDetectorName = NewType('NeXusDetectorName', str)
"""Name of detector entry in NeXus file"""

TransformationPath = NewType('TransformationPath', str)
"""
Name of the entry under which to store the transformation computed from the
transformation chain, for the detectors and the monitors
"""

# 2  Workflow parameters

UncertaintyBroadcastMode = Enum(
    'UncertaintyBroadcastMode', ['drop', 'upper_bound', 'fail']
)
"""
Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""

ReturnEvents = NewType('ReturnEvents', bool)
"""Whether to return events in the output I(Q)"""

WavelengthBins = NewType('WavelengthBins', sc.Variable)
"""Wavelength binning"""

WavelengthBands = NewType('WavelengthBands', sc.Variable)
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
"""Q binning"""

QxyBins = NewType('QxyBins', dict[str, sc.Variable])
"""Binning for 'Qx' and 'Qy'. If set this overrides QBins."""

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

DimsToKeep = NewType('DimsToKeep', Sequence[str])
"""Dimensions that should not be reduced and thus still be present in the final
I(Q) result (this is typically the layer dimension)."""

OutFilename = NewType('OutFilename', str)
"""Filename of the output"""


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    """Name of Incident|Transmission monitor in NeXus file"""


class FileList(sciline.Scope[RunType, list], list):
    """Filenames of BackgroundRun|EmptyBeamRun|SampleRun"""


BeamStopPosition = NewType('BeamStopPosition', sc.Variable)
"""Approximate center of the beam stop position"""

BeamStopRadius = NewType('BeamStopRadius', sc.Variable)
"""Radius of the beam stop"""

# 3  Workflow (intermediate) results


class SamplePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Sample position"""


class SourcePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Source position"""


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


class LoadedFileContents(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """The entire contents of a loaded file"""


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data"""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied"""


class CalibratedMaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied and calibrated pixel positions"""


class NormWavelengthTerm(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Normalization term (numerator) for IofQ before scaling with solid-angle."""


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


class RawMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data"""


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
