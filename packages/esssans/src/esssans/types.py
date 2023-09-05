# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

import sciline
import scipp as sc

# Note: Unused by IofQ workflow, it uses monitors directly?
TransmissionFraction = NewType('TransmissionFraction', sc.DataArray)

WavelengthBands = NewType('WavelengthBands', sc.Variable)
WavelengthBins = NewType('WavelengthBins', sc.Variable)
QBins = NewType('QBins', sc.Variable)
NonBackgroundWavelengthRange = NewType('NonBackgroundWavelengthRange', sc.Variable)
DirectBeamFilename = NewType('DirectBeamFilename', str)
BeamCenter = NewType('BeamCenter', sc.Variable)
DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
SampleHolderMask = NewType('SampleHolderMask', sc.Variable)
WavelengthMask = NewType('WavelengthMask', sc.DataArray)
DirectBeam = NewType('DirectBeam', sc.DataArray)
CleanDirectBeam = NewType('CleanDirectBeam', sc.DataArray)  # after resample
BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', sc.DataArray)
CorrectForGravity = NewType('CorrectForGravity', bool)

BackgroundRun = NewType('BackgroundRun', int)
DirectRun = NewType('DirectRun', int)
SampleRun = NewType('SampleRun', int)
RunType = TypeVar('RunType', BackgroundRun, DirectRun, SampleRun)

# TODO Need Scope with multiple params, see scipp/sciline#42
Incident = NewType('Incident', int)
Transmission = NewType('Transmission', int)
MonitorType = TypeVar('MonitorType', Incident, Transmission)

Numerator = NewType('Numerator', sc.DataArray)
Denominator = NewType('Denominator', sc.DataArray)
IofQPart = TypeVar('IofQPart')


class SolidAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    ...


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    ...


class Filename(sciline.Scope[RunType, str], str):
    ...


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    ...


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    ...


@dataclass
class Clean(Generic[RunType, IofQPart]):
    value: sc.DataArray


@dataclass
class CleanMasked(Generic[RunType, IofQPart]):
    value: sc.DataArray


@dataclass
class CleanQ(Generic[RunType, IofQPart]):
    value: sc.DataArray


@dataclass
class CleanSummedQ(Generic[RunType, IofQPart]):
    value: sc.DataArray


class IofQ(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    ...


@dataclass
class RawMonitor(Generic[RunType, MonitorType]):
    value: sc.DataArray


@dataclass
class WavelengthMonitor(Generic[RunType, MonitorType]):
    value: sc.DataArray


@dataclass
class CleanMonitor(Generic[RunType, MonitorType]):
    value: sc.DataArray
