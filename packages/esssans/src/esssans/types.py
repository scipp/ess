# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType, TypeVar, Generic
import scipp as sc
import sciline

WavelengthBins = NewType('WavelengthBins', sc.Variable)
QBins = NewType('QBins', sc.Variable)
NonBackgroundWavelengthRange = NewType('NonBackgroundWavelengthRange', sc.Variable)
DirectBeamFilename = NewType('DirectBeamFilename', str)

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


@dataclass
class RawMonitor(Generic[RunType, MonitorType]):
    value: sc.DataArray


@dataclass
class WavelengthMonitor(Generic[RunType, MonitorType]):
    value: sc.DataArray


@dataclass
class CleanMonitor(Generic[RunType, MonitorType]):
    value: sc.DataArray
