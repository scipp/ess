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
from scippneutron.io import cif

from ess.reduce.nexus import generic_types as reduce_gt
from ess.reduce.nexus import types as reduce_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

# 1 TypeVars used to parametrize the generic parts of the workflow

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
NeXusSample = reduce_gt.NeXusSample
NeXusSource = reduce_gt.NeXusSource
RunType = reduce_gt.RunType
SampleRun = reduce_gt.SampleRun
ScatteringRunType = reduce_gt.ScatteringRunType
Transmission = reduce_gt.Transmission
TransmissionRun = reduce_gt.TransmissionRun
SamplePosition = reduce_gt.SamplePosition
SourcePosition = reduce_gt.SourcePosition
VanadiumRun = reduce_gt.VanadiumRun

DetectorBankSizes = reduce_t.DetectorBankSizes
NeXusDetectorName = reduce_t.NeXusDetectorName


# 2 Workflow parameters

CalibrationFilename = NewType("CalibrationFilename", str | None)
"""Filename of the instrument calibration file."""


DspacingBins = NewType("DSpacingBins", sc.Variable)
"""Bin edges for d-spacing."""


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


class DataWithScatteringCoordinates(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with scattering coordinates computed for all events: wavelength, 2theta,
    d-spacing."""


class DspacingData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing."""


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


TofMask = NewType("TofMask", Callable | None)
"""TofMask is a callable that returns a mask for a given TofData."""


TwoThetaMask = NewType("TwoThetaMask", Callable | None)
"""TwoThetaMask is a callable that returns a mask for a given TwoThetaData."""


WavelengthMask = NewType("WavelengthMask", Callable | None)
"""WavelengthMask is a callable that returns a mask for a given WavelengthData."""


CIFAuthors = NewType('CIFAuthors', list[cif.Author])
"""List of authors to save to output CIF files."""

ReducedDspacingCIF = NewType('ReducedDspacingCIF', cif.CIF)
"""Reduced data in d-spacing, ready to be saved to a CIF file."""

del sc, sciline, NewType, TypeVar
