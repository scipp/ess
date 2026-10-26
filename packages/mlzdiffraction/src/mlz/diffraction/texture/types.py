# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX texture workflows."""

from dataclasses import dataclass
from typing import NewType

import sciline as sl
import scipp as sc

from ..powtex.types import NonEmptyCanRunType, RunType


class DspacingCylSurfaceDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data projected to the POWTEX cylindrical surface."""


class CylindricalSurfaceTransformGraph(sl.Scope[dict], dict):
    """Graph for projecting 3-D POWTEX data onto a cylindrical surface."""


class Texture3D(sl.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Texture data binned in gamma, two-theta, and d-spacing."""


DetectorViewRawData = NewType("DetectorViewRawData", sc.DataArray)
FlatViewRawData = NewType("FlatViewRawData", sc.DataArray)
TwoThetaRingsViewRawData = NewType("TwoThetaRingsViewRawData", sc.DataArray)
GammaSlicesViewRawData = NewType("GammaSlicesViewRawData", sc.DataArray)
SampleRawData2D = NewType("SampleRawData2D", sc.DataArray)

GammaSlicesViewData = NewType("GammaSlicesViewData", sc.DataArray)
TwoThetaRingsViewData = NewType("TwoThetaRingsViewData", sc.DataArray)
StatisticsData = NewType("StatisticsData", sc.DataArray)

GammaCutStart = NewType("GammaCutStart", sc.Variable)
GammaCutEnd = NewType("GammaCutEnd", sc.Variable)
DropPartialBins = NewType("DropPartialBins", bool)
IncludeGapBins = NewType("IncludeGapBins", bool)  # maybe not needed in future

GammaBinWidth = NewType("GammaBinWidth", sc.Variable)
TwoThetaBinWidth = NewType("TwoThetaBinWidth", sc.Variable)
DspacingBinWidth = NewType("DspacingBinWidth", sc.Variable)

GammaBinningData = NewType("GammaBinningData", sc.DataGroup)
TwoThetaBinningData = NewType("TwoThetaBinningData", sc.Variable)
DspacingBinningData = NewType("DspacingBinningData", sc.Variable)


# frozen=True makes it impossible to change the values of the fields after
# initialization.
@dataclass
class BinningSettings:
    """Texture workflow binning settings."""

    gamma_cut_start: sc.Variable
    gamma_cut_end: sc.Variable
    drop_partial_bins: bool
    two_theta_min: sc.Variable
    two_theta_max: sc.Variable


class EmptyCanSubtracted3D(sl.Scope[NonEmptyCanRunType, sc.DataGroup], sc.DataGroup):
    """Texture data after empty-can subtraction."""


NormalisedTexture2D = NewType("NormalisedTexture2D", sc.DataGroup)
NormalisedTexture3D = NewType("NormalisedTexture3D", sc.DataGroup)

GammaSliceWidth = NewType("GammaSliceWidth", sc.DataArray)

DspacingViewData = NewType("DspacingViewData", sc.DataArray)
TwoThetaViewData = NewType("TwoThetaViewData", sc.DataArray)
IntegratedTwoThetaViewData = NewType("IntegratedTwoThetaViewData", sc.DataArray)

ProcessedData = NewType("ProcessedData", sc.DataArray)
