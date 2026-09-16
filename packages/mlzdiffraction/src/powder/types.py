# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX powder workflows."""

from collections.abc import Callable
from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ess.powder import types as powder_t
from ess.reduce.unwrap import types as unwrap_t

AccumulatedProtonCharge = powder_t.AccumulatedProtonCharge
CalibrationData = powder_t.CalibrationData
CorrectedDetector = powder_t.CorrectedDetector
CorrectedDspacing = powder_t.CorrectedDspacing
DspacingDetector = powder_t.DspacingDetector
EmptyCanRun = powder_t.EmptyCanRun
NormalizedDspacing = powder_t.NormalizedDspacing
Position = powder_t.Position
RawDataAndMetadata = powder_t.RawDataAndMetadata
SampleRun = powder_t.SampleRun
TofMask = powder_t.TofMask
TwoThetaMask = powder_t.TwoThetaMask
UncertaintyBroadcastMode = powder_t.UncertaintyBroadcastMode
VanadiumRun = powder_t.VanadiumRun
WavelengthMask = powder_t.WavelengthMask

TofDetector = unwrap_t.TofDetector
TofMonitor = unwrap_t.TofMonitor
WavelengthDetector = unwrap_t.WavelengthDetector

EmptyVanadiumCanRun = NewType("EmptyVanadiumCanRun", int)
"""Empty-can run for a vanadium measurement."""

CalibrationRun = NewType("CalibrationRun", int)
"""Calibration run."""

RunType = TypeVar("RunType", SampleRun, EmptyCanRun, VanadiumRun, EmptyVanadiumCanRun)
NonEmptyCanRunType = TypeVar("NonEmptyCanRunType", SampleRun, VanadiumRun)
EmptyCanRunType = TypeVar("EmptyCanRunType", EmptyCanRun, EmptyVanadiumCanRun)
GeneralRunType = TypeVar(
    "GeneralRunType",
    SampleRun,
    EmptyCanRun,
    VanadiumRun,
    EmptyVanadiumCanRun,
    CalibrationRun,
)

DSpacingMask = NewType("DSpacingMask", Callable | None)
"""Callable masking d-spacing data, or ``None`` if no mask is applied."""

DperpMask = NewType("DperpMask", Callable | None)
"""Callable masking dperp data, or ``None`` if no mask is applied."""

MaskType = TypeVar(
    "MaskType", TwoThetaMask, TofMask, WavelengthMask, DSpacingMask, DperpMask
)

P2DOutputFilename = NewType("P2DOutputFilename", str)
"""Filename of the P2D output file."""

DspacingBins = NewType("DspacingBins", sc.Variable)
"""Bin edges for d-spacing."""

DperpBins = NewType("DperpBins", sc.Variable)
"""Bin edges for dperp."""

KeepEventsMLZ = NewType("KeepEventsMLZ", bool)
"""Flag indicating whether focussed P2D data should keep events instead of histogramming."""

NormalizedIntensity2D = NewType("NormalizedIntensity2D", sc.DataArray)
"""2-D intensity of normalized sample data."""

ExtendedIntensity2D = NewType("ExtendedIntensity2D", sc.Dataset)
"""2-D intensity with auxiliary two-theta and wavelength data."""

MaskedExtendedIntensity2D = NewType("MaskedExtendedIntensity2D", sc.Dataset)
"""Extended 2-D intensity with masks."""

MaskedIntensity2D = NewType("MaskedIntensity2D", sc.DataArray)
"""2-D intensity with combined masks applied."""

IntensityP2D = NewType("IntensityP2D", sc.DataArray)
"""Intensity data ready for export into a P2D file."""

DataFrameP2D = NewType("DataFrameP2D", object)
"""Pandas dataframe containing P2D export columns."""


class MaskingData(sciline.Scope[MaskType, type], type):
    """Type for a mask function."""


class ElasticCoordTransformGraphMLZ(sciline.Scope[dict], dict):
    """Coordinate transformation graph for MLZ/POWTEX reductions."""

class SpectralDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data flattened to a spectrum dimension."""


class Dspacing2D(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data binned in d-spacing and dperp."""


class EmptyCanSubtracted2D(
    sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray
):
    """2-D intensity after empty-can subtraction."""


class EmptyCanSubtractedNoNegative2D(
    sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray
):
    """2-D intensity after empty-can subtraction and negative-value handling."""


class ShiftNegativeIntensity(sciline.Scope[NonEmptyCanRunType, int], int):
    """Flag controlling whether negative intensity values are shifted."""


class NegativeIntensityResetValue(sciline.Scope[NonEmptyCanRunType, float], float):
    """Replacement value for negative intensities when shifting is disabled."""


class FilenameMLZ(sciline.Scope[GeneralRunType, Path], Path):
    """Filename for local MLZ/POWTEX data."""


class FilepathMLZ(sciline.Scope[GeneralRunType, Path], Path):
    """Directory containing local MLZ/POWTEX data."""


del sc, sciline, NewType, TypeVar
