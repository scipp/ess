# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for spectroscopy."""

from typing import Any, NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce import unwrap
from ess.reduce.nexus import types as reduce_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

# NeXus types

Beamline = reduce_t.Beamline
DetectorPositionOffset = reduce_t.DetectorPositionOffset
EmptyDetector = reduce_t.EmptyDetector
Filename = reduce_t.Filename
GravityVector = reduce_t.GravityVector
Measurement = reduce_t.Measurement
NeXusClass = reduce_t.NeXusClass
NeXusComponentLocationSpec = reduce_t.NeXusComponentLocationSpec
NeXusComponent = reduce_t.NeXusComponent
NeXusData = reduce_t.NeXusData
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusFileSpec = reduce_t.NeXusFileSpec
NeXusMonitorName = reduce_t.NeXusName
NeXusTransformation = reduce_t.NeXusTransformation
Position = reduce_t.Position
PreopenNeXusFile = reduce_t.PreopenNeXusFile
RawDetector = reduce_t.RawDetector
RawMonitor = reduce_t.RawMonitor

SampleRun = reduce_t.SampleRun
VanadiumRun = reduce_t.VanadiumRun

# BIFROST monitors
ElasticMonitor = NewType('ElasticMonitor', int)
NormalizationMonitor = NewType('NormalizationMonitor', int)
PSCMonitor = NewType('PSCMonitor', int)
OverlapMonitor = NewType('OverlapMonitor', int)

# Type vars

RunType = TypeVar("RunType", SampleRun, VanadiumRun)
MonitorType = TypeVar(
    "MonitorType", ElasticMonitor, NormalizationMonitor, PSCMonitor, OverlapMonitor
)

# Time-of-flight types

WavelengthDetector = unwrap.WavelengthDetector
WavelengthMonitor = unwrap.WavelengthMonitor
LookupTableRelativeErrorThreshold = unwrap.LookupTableRelativeErrorThreshold
MonitorLtotal = unwrap.MonitorLtotal
PulseStride = unwrap.PulseStride
PulseStrideOffset = unwrap.PulseStrideOffset
PulsePeriod = unwrap.PulsePeriod
ErrorLimitedLookupTable = unwrap.ErrorLimitedLookupTable
LookupTable = unwrap.LookupTable
LookupTableFilename = unwrap.LookupTableFilename


# Custom types


class Analyzer(sciline.Scope[RunType, sc.DataGroup[Any]], sc.DataGroup[Any]):
    """A single wavelength analyzer loaded from an NXcrystal.

    Can be obtained from ``Analyzers``.
    """


class Analyzers(sciline.Scope[RunType, sc.DataGroup[Any]], sc.DataGroup[Any]):
    """All wavelength analyzers loaded from a NXcrystals."""


class DataAtSample(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class DataGroupedByRotation(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class QDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts with a momentum transfer coordinate."""


EnergyBins = NewType('EnergyBins', sc.Variable)


class EnergyData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class EnergyQDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts with a momentum transfer and energy transfer coordinates."""


class IncidentEnergyDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector counts with an incident energy coordinate."""


class NormalizedIncidentEnergyDetector(
    sciline.Scope[RunType, sc.DataArray], sc.DataArray
):
    """Normalized detector counts with an incident energy coordinate."""


class SampleAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Rotation angle of the sample, possibly as a function of time.

    For BIFROST, this is angle "a3".
    """


class InstrumentAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Rotation angle of the instrument, possibly as a function of time.

    For BIFROST, this is angle "a4".
    """


NXspeFileName = NewType('NXspeFileName', str)
NXspeFileNames = NewType('NXspeFileNames', list[NXspeFileName])

OutFilename = NewType("OutFilename", str)


class PrimarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


class SecondarySpecCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


InelasticCoordTransformGraph = NewType('InelasticCoordTransformGraph', dict)


class ElasticCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


class MonitorCoordTransformGraph(sciline.Scope[RunType, dict], dict): ...


SQWBinSizes = NewType('SQWBinSizes', dict[str, int])


UncertaintyBroadcastMode = _UncertaintyBroadcastMode


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Accumulated proton charge for a measurement."""
