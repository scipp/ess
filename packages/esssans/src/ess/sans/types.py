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

ScatteringRunType = TypeVar(
    'ScatteringRunType',
    SampleRun,
    BackgroundRun,
)


class TransmissionRun(sciline.Scope[ScatteringRunType, int], int):
    """Mapping between ScatteringRunType and transmission run.
    In the case where no transmission run is provided, the transmission run should be
    the same as the measurement (sample or background) run."""


RunType = TypeVar(
    'RunType',
    BackgroundRun,
    EmptyBeamRun,
    SampleRun,
    # Note that mypy does not seem to like this nesting, may need to find a workaround
    TransmissionRun[SampleRun],
    TransmissionRun[BackgroundRun],
)
"""TypeVar used for specifying BackgroundRun, EmptyBeamRun or SampleRun"""

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
NeXusDetectorName = NewType('NeXusDetectorName', str)
"""Name of detector entry in NeXus file"""

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


PixelMaskFilename = NewType('PixelMaskFilename', str)

FilenameType = TypeVar('FilenameType', bound=str)


DataFolder = NewType('DataFolder', str)


class FilePath(sciline.Scope[FilenameType, str], str):
    """Path to a file"""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of a run"""


MaskedDetectorIDs = NewType('MaskedDetectorIDs', sc.Variable)
"""1-D variable listing all masked detector IDs."""


# 3  Workflow (intermediate) results


class RawSource(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw source from NeXus file"""


class RawSample(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw sample from NeXus file"""


class SamplePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Sample position"""


class SourcePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Source position"""


DirectBeam = NewType('DirectBeam', sc.DataArray)
"""Direct beam"""


class TransmissionFraction(
    sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray
):
    """Transmission fraction"""


CleanDirectBeam = NewType('CleanDirectBeam', sc.DataArray)
"""Direct beam after resampling to required wavelength bins"""


class DetectorPixelShape(sciline.Scope[ScatteringRunType, sc.DataGroup], sc.DataGroup):
    """Geometry of the detector from description in nexus file."""


class LabFrameTransform(sciline.Scope[ScatteringRunType, sc.Variable], sc.Variable):
    """Coordinate transformation from detector local coordinates
    to the sample frame of reference."""


class SolidAngle(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Solid angle of detector pixels seen from sample position"""


class LoadedNeXusDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Detector data, loaded from a NeXus file, containing not only neutron events
    but also pixel shape information, transformations, ..."""


class LoadedNeXusMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataGroup], sc.DataGroup
):
    """Monitor data loaded from a NeXus file, containing not only neutron events
    but also transformations, ..."""


class RawData(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Raw detector data"""


class ConfiguredReducibleDataData(
    sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray
):
    """Raw event data where variances and necessary coordinates
    (e.g. sample and source position) have been added, and where optionally some
    user configuration was applied to some of the coordinates."""


class TofData(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Data with a time-of-flight coordinate"""


class TofMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataGroup], sc.DataGroup
):
    """Monitor data with a time-of-flight coordinate"""


PixelMask = NewType('PixelMask', sc.Variable)


class MaskedData(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied"""


class CalibratedMaskedData(
    sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray
):
    """Raw data with pixel-specific masks applied and calibrated pixel positions"""


class NormWavelengthTerm(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """Normalization term (numerator) for IofQ before scaling with solid-angle."""


class CleanWavelength(
    sciline.ScopeTwoParams[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """
    Prerequisite for IofQ numerator or denominator.

    This can either be the sample or background counts, converted to wavelength,
    or the respective normalization terms computed from the respective solid angle,
    direct beam, and monitors.
    """


class CleanWavelengthMasked(
    sciline.ScopeTwoParams[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of applying wavelength masking to :py:class:`CleanWavelength`"""


class CleanQ(
    sciline.ScopeTwoParams[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of converting :py:class:`CleanWavelengthMasked` to Q"""


class CleanSummedQ(
    sciline.ScopeTwoParams[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of histogramming/binning :py:class:`CleanQ` over all pixels into Q bins"""


class CleanSummedQMergedBanks(
    sciline.ScopeTwoParams[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """CleanSummedQ with merged banks"""


class FinalSummedQ(
    sciline.ScopeTwoParams[ScatteringRunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Final data into Q bins, in a state that is ready to be normalized."""


class IofQ(sciline.Scope[ScatteringRunType, sc.DataArray], sc.DataArray):
    """I(Q)"""


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', sc.DataArray)
"""I(Q) with background (given by I(Q) of the background run) subtracted"""


class RawMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data"""


class ConfiguredReducibleMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data where variances and necessary coordinates
    (e.g. source position) have been added, and where optionally some
    user configuration was applied to some of the coordinates."""


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
