# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX powder workflows."""

from collections.abc import Callable
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ..powtex.types import (
    NonEmptyCanRunType,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
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


class EmptyCanSubtractedNoNegative2D(
    sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray
):
    """2-D intensity after empty-can subtraction and negative-value handling."""
