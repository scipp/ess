# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used in ess.powder.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from collections.abc import Callable
from typing import Any, NewType, TypeVar

import sciline
import scipp as sc
from ess.reduce.nexus import generic_types as reduce_gt
from ess.reduce.nexus import types as reduce_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

BackgroundRun = reduce_gt.BackgroundRun
CalibratedDetector = reduce_gt.CalibratedDetector
CalibratedMonitor = reduce_gt.CalibratedMonitor
DetectorData = reduce_gt.DetectorData
DetectorPositionOffset = reduce_gt.DetectorPositionOffset
EmptyBeamRun = reduce_gt.EmptyBeamRun
Filename = reduce_gt.Filename
Incident = reduce_gt.Incident
MonitorData = reduce_gt.MonitorData
MonitorPositionOffset = reduce_gt.MonitorPositionOffset
MonitorType = reduce_gt.MonitorType
NeXusMonitorName = reduce_gt.NeXusMonitorName
NeXusDetector = reduce_gt.NeXusDetector
NeXusMonitor = reduce_gt.NeXusMonitor
RunType = reduce_gt.RunType
SampleRun = reduce_gt.SampleRun
ScatteringRunType = reduce_gt.ScatteringRunType
Transmission = reduce_gt.Transmission
TransmissionRun = reduce_gt.TransmissionRun
SamplePosition = reduce_gt.SamplePosition
SourcePosition = reduce_gt.SourcePosition

DetectorBankSizes = reduce_t.DetectorBankSizes
NeXusDetectorName = reduce_t.NeXusDetectorName


# 1 TypeVars used to parametrize the generic parts of the workflow

# 1.1 Run types
BackgroundRun = NewType("BackgroundRun", int)
"""Empty sample can run."""
EmptyBeamRun = NewType("EmptyBeamRun", int)
"""Empty instrument run."""
SampleRun = NewType("SampleRun", int)
"""Sample run."""
VanadiumRun = NewType("VanadiumRun", int)
"""Vanadium run."""
RunType = TypeVar("RunType", EmptyBeamRun, SampleRun, VanadiumRun)
"""TypeVar used for specifying the run."""

# 1.2  Monitor types
Monitor1 = NewType('Monitor1', int)
"""Placeholder for monitor 1."""
Monitor2 = NewType('Monitor2', int)
"""Placeholder for monitor 2."""
MonitorType = TypeVar('MonitorType', Monitor1, Monitor2)
"""TypeVar used for identifying a monitor"""

# 2 Workflow parameters

DetectorBankSizes = NewType("DetectorBankSizes", dict[str, dict[str, int | Any]])

CalibrationFilename = NewType("CalibrationFilename", str | None)
"""Filename of the instrument calibration file."""


NeXusDetectorName = NewType("NeXusDetectorName", str)
"""Name of detector entry in NeXus file"""


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    """Name of Incident|Transmission monitor in NeXus file"""


DspacingBins = NewType("DSpacingBins", sc.Variable)
"""Bin edges for d-spacing."""


class Filename(sciline.Scope[RunType, str], str):
    """Name of an input file."""


OutFilename = NewType("OutFilename", str)
"""Filename of the output."""

TwoThetaBins = NewType("TwoThetaBins", sc.Variable)
"""Bin edges for grouping in 2theta.

This is used by an alternative focussing step that groups detector
pixels by scattering angle into bins given by these edges.
"""

UncertaintyBroadcastMode = _UncertaintyBroadcastMode
"""Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""

ValidTofRange = NewType("ValidTofRange", sc.Variable)
"""Min and max tof value of the instrument."""

# 3 Workflow (intermediate) results


class AccumulatedProtonCharge(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Total proton charge."""

    # Override the docstring of super().__init__ because if contains a broken link
    # when used by Sphinx in ESSdiffraction.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


CalibrationData = NewType("CalibrationData", sc.Dataset | None)
"""Detector calibration data."""

DataFolder = NewType("DataFolder", str)


class DataWithScatteringCoordinates(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with scattering coordinates computed for all events: wavelength, 2theta,
    d-spacing."""


NeXusDetectorDimensions = NewType("NeXusDetectorDimensions", dict[str, int])
"""Logical detector dimensions."""


class DspacingData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing."""


class DspacingDataWithoutVariances(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing where variances where removed."""


DspacingHistogram = NewType("DspacingHistogram", sc.DataArray)
"""Histogrammed intensity vs d-spacing."""

ElasticCoordTransformGraph = NewType("ElasticCoordTransformGraph", dict)
"""Graph for transforming coordinates in elastic scattering."""


class FilteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data without invalid events."""


class FocussedDataDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs d-spacing after focussing pixels."""


class FocussedDataDspacingTwoTheta(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs (d-spacing, 2theta) after focussing pixels."""


IofDspacing = NewType("IofDspacing", sc.DataArray)
"""Data that has been normalized by a vanadium run."""

IofDspacingTwoTheta = NewType("IofDspacingTwoTheta", sc.DataArray)
"""Data that has been normalized by a vanadium run, and grouped into 2theta bins."""


class NeXusDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """
    Detector loaded from a NeXus file, without event data.

    Contains detector numbers, pixel shape information, transformations, ...
    """


class NeXusMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataGroup], sc.DataGroup
):
    """
    Monitor loaded from a NeXus file, without event data.

    Contains detector numbers, pixel shape information, transformations, ...
    """


class DetectorEventData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Event data loaded from a detector in a NeXus file"""


class MonitorEventData(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Event data loaded from a monitor in a NeXus file"""


class RawMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data"""


class RawMonitorData(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data where variances and necessary coordinates
    (e.g. source position) have been added, and where optionally some
    user configuration was applied to some of the coordinates."""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with masked pixels, tof regions, wavelength regions, 2theta regions, or
    dspacing regions."""


MaskedDetectorIDs = NewType("MaskedDetectorIDs", dict[str, sc.Variable])
"""1-D variable listing all masked detector IDs."""


class NormalizedByProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by proton charge."""


PixelMaskFilename = NewType("PixelMaskFilename", str)
"""Filename of a pixel mask."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Time-dependent proton charge."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


class RawDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data (events / histogram) extracted from a RawDetector."""


class RawSample(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from a loaded sample."""


class RawSource(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from a loaded neutron source."""


class ReducibleDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that is in a state ready for reduction."""


class SamplePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Sample position"""


class SourcePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Source position"""


TofMask = NewType("TofMask", Callable | None)
"""TofMask is a callable that returns a mask for a given TofData."""


TwoThetaMask = NewType("TwoThetaMask", Callable | None)
"""TwoThetaMask is a callable that returns a mask for a given TwoThetaData."""


WavelengthMask = NewType("WavelengthMask", Callable | None)
"""WavelengthMask is a callable that returns a mask for a given WavelengthData."""


del sc, sciline, NewType, TypeVar
