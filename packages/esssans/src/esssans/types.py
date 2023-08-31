# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType, TypeVar, Generic, Tuple
import scipp as sc
import sciline

WavelengthBins = NewType('WavelengthBins', sc.Variable)
QBins = NewType('QBins', sc.Variable)
NonBackgroundWavelengthRange = NewType('NonBackgroundWavelengthRange', sc.Variable)
DirectBeamFilename = NewType('DirectBeamFilename', str)
BeamCenter = NewType('BeamCenter', Tuple[sc.Variable, sc.Variable])
DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
SampleHolderMask = NewType('SampleHolderMask', sc.Variable)

SampleRun = NewType('SampleRun', int)
DirectRun = NewType('DirectRun', int)
RunType = TypeVar('RunType', SampleRun, DirectRun)

# TODO Need Scope with multiple params, see scipp/sciline#42
Incident = NewType('Incident', int)
Transmission = NewType('Transmission', int)
MonitorType = TypeVar('MonitorType', Incident, Transmission)


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    ...


class Filename(sciline.Scope[RunType, str], str):
    ...


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    ...


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
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
