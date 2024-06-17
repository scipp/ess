# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used in ess.powder.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from enum import Enum
from typing import Any, Callable, Dict, NewType, TypeVar

import sciline
import scipp as sc

# 1 TypeVars used to parametrize the generic parts of the workflow

# 1.1 Run types
EmptyCanRun = NewType("EmptyCanRun", int)
"""Empty sample can run."""
EmptyInstrumentRun = NewType("EmptyInstrumentRun", int)
"""Empty instrument run."""
SampleRun = NewType("SampleRun", int)
"""Sample run."""
VanadiumRun = NewType("VanadiumRun", int)
"""Vanadium run."""
RunType = TypeVar("RunType", EmptyInstrumentRun, SampleRun, VanadiumRun)
"""TypeVar used for specifying the run."""


# 2 Workflow parameters

CalibrationFilename = NewType("CalibrationFilename", str)
"""Filename of the instrument calibration file."""


NeXusDetectorName = NewType("NeXusDetectorName", str)
"""Name of detector entry in NeXus file"""

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

UncertaintyBroadcastMode = Enum(
    "UncertaintyBroadcastMode", ["drop", "upper_bound", "fail"]
)
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


CalibrationData = NewType("CalibrationData", sc.Dataset)
"""Detector calibration data."""

DataFolder = NewType("DataFolder", str)


class DataWithScatteringCoordinates(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with scattering coordinates computed for all events: wavelength, 2theta,
    d-spacing."""


class NeXusDetectorDimensions(
    sciline.Scope[NeXusDetectorName, Dict[str, int]], Dict[str, int]
):
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


class LoadedNeXusDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Detector data, loaded from a NeXus file, containing not only neutron events
    but also pixel shape information, transformations, ..."""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with masked pixels, tof regions, wavelength regions, 2theta regions, or
    dspacing regions."""


MaskedDetectorIDs = NewType("MaskedDetectorIDs", Dict[str, sc.Variable])
"""1-D variable listing all masked detector IDs."""


class NormalizedByProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by proton charge."""


PixelMaskFilename = NewType("PixelMaskFilename", str)
"""Filename of a pixel mask."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Time-dependent proton charge."""


RawCalibrationData = NewType("RawCalibrationData", sc.Dataset)
"""Calibration data as loaded from file, needs preprocessing before using."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


class RawDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Full raw data for a detector."""


class RawDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
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
