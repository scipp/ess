# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX powder workflows."""

from collections.abc import Callable
from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ..powtex import types as powtex_t

CalibrationData = powtex_t.CalibrationData
CalibrationRun = powtex_t.CalibrationRun
EmptyCanRun = powtex_t.EmptyCanRun
EmptyVanadiumCanRun = powtex_t.EmptyVanadiumCanRun
Filename = powtex_t.Filename
Filepath = powtex_t.Filepath
KeepEvents = powtex_t.KeepEvents
NonEmptyCanRunType = powtex_t.NonEmptyCanRunType
RunType = powtex_t.RunType
SampleRun = powtex_t.SampleRun
ShiftNegativeIntensity = powtex_t.ShiftNegativeIntensity
NegativeIntensityResetValue = powtex_t.NegativeIntensityResetValue
TofDetector = powtex_t.TofDetector
TofMask = powtex_t.TofMask
TwoThetaMask = powtex_t.TwoThetaMask
UncertaintyBroadcastMode = powtex_t.UncertaintyBroadcastMode
VanadiumRun = powtex_t.VanadiumRun
WavelengthDetector = powtex_t.WavelengthDetector
WavelengthMask = powtex_t.WavelengthMask

DSpacingMask = NewType("DSpacingMask", Callable | None)
"""Callable masking d-spacing data, or ``None`` if no mask is applied."""

DperpMask = NewType("DperpMask", Callable | None)
"""Callable masking dperp data, or ``None`` if no mask is applied."""

MaskType = TypeVar(
    "MaskType", TwoThetaMask, TofMask, WavelengthMask, DSpacingMask, DperpMask
)

P2DOutputFilename = NewType("P2DOutputFilename", Path)
"""Filename of the P2D output file."""

DspacingBins = NewType("DspacingBins", sc.Variable)
"""Bin edges for d-spacing."""

DperpBins = NewType("DperpBins", sc.Variable)
"""Bin edges for dperp."""

NormalisedIntensity2D = NewType("NormalisedIntensity2D", sc.DataArray)
"""2-D intensity of normalised sample data."""

ExtendedData2D = NewType("ExtendedData2D", sc.Dataset)
"""2-D intensity with auxiliary 2-D two-theta, wavelength, dspacing, and dperp data."""

MaskedData2D = NewType("MaskedData2D", sc.Dataset)
"""Extended 2-D data with masks."""

Intensity2D = NewType("Intensity2D", sc.DataArray)
"""2-D intensity with combined masks applied."""

IntensityP2D = NewType("IntensityP2D", sc.DataArray)
"""Intensity data ready for export into a P2D file."""

DataFrameP2D = NewType("DataFrameP2D", object)
"""Pandas dataframe containing P2D export columns."""


class MaskingData(sciline.Scope[MaskType, type], type):
    """Type for a mask function."""


class ElasticCoordTransformGraph(sciline.Scope[dict], dict):
    """Coordinate transformation graph for MLZ/POWTEX reduction."""


class SpectralDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data flattened to a spectrum dimension."""


class Dspacing2D(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data binned in d-spacing and dperp."""


class EmptyCanSubtracted2D(
    sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray
):
    """2-D intensity after empty-can subtraction."""


class AdjustedIntensity2D(
    sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray
):
    """2-D intensity after empty-can subtraction and negative-value handling."""
