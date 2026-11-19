# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX texture workflows."""

from dataclasses import dataclass
from typing import NewType

import sciline as sl
import scipp as sc

from ..powtex import types as powtex_t

CalibrationData = powtex_t.CalibrationData
CalibrationRun = powtex_t.CalibrationRun
EmptyCanRun = powtex_t.EmptyCanRun
EmptyVanadiumCanRun = powtex_t.EmptyVanadiumCanRun
Filename = powtex_t.Filename
Filepath = powtex_t.Filepath
NonEmptyCanRunType = powtex_t.NonEmptyCanRunType
RunType = powtex_t.RunType
SampleRun = powtex_t.SampleRun
TofDetector = powtex_t.TofDetector
VanadiumRun = powtex_t.VanadiumRun

class DspacingCylSurfaceDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data projected to the POWTEX cylindrical surface."""


class CylindricalSurfaceTransformGraph(sl.Scope[dict], dict):
    """Graph for projecting 3-D POWTEX data onto a cylindrical surface."""


class Texture3D(sl.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Texture data binned in gamma, two-theta, and d-spacing."""


DetectorViewRawData = NewType("DetectorViewRawData", sc.DataArray)
FlatViewRawData = NewType("FlatViewRawData", sc.DataArray)
GammaSlicesViewRawData = NewType("GammaSlicesViewRawData", sc.DataArray)
RawSampleData2D = NewType("RawSampleData2D", sc.DataArray)
TwoThetaRingsViewRawData = NewType("TwoThetaRingsViewRawData", sc.DataArray)

GammaSlicesViewData = NewType("GammaSlicesViewData", sc.DataArray)
StatisticsData = NewType("StatisticsData", sc.DataArray)
TwoThetaRingsViewData = NewType("TwoThetaRingsViewData", sc.DataArray)

DspacingBinWidth = NewType("DspacingBinWidth", sc.Variable)
GammaBinWidth = NewType("GammaBinWidth", sc.Variable)
TwoThetaBinWidth = NewType("TwoThetaBinWidth", sc.Variable)

DspacingBinningData = NewType("DspacingBinningData", sc.Variable)
GammaBinningData = NewType("GammaBinningData", sc.DataGroup)
TwoThetaBinningData = NewType("TwoThetaBinningData", sc.Variable)

# frozen=True makes it impossible to change the values of the fields after
# initialization.
@dataclass
class InstrumentViewSettings:
    """Texture workflow detector geometry settings for visualisations."""
    drop_partial_bins: bool
    gamma_cut_start: sc.Variable
    gamma_cut_end: sc.Variable
    two_theta_min: sc.Variable
    two_theta_max: sc.Variable


class EmptyCanSubtracted3D(sl.Scope[NonEmptyCanRunType, sc.DataGroup], sc.DataGroup):
    """Texture data after empty-can subtraction."""


CalibratedTexture2D = NewType("CalibratedTexture2D", sc.DataGroup)
CalibratedTexture3D = NewType("CalibratedTexture3D", sc.DataGroup)

GammaSliceWidth = NewType("GammaSliceWidth", sc.DataArray)

DspacingViewData = NewType("DspacingViewData", sc.DataArray)
TwoThetaViewData = NewType("TwoThetaViewData", sc.DataArray)
IntegratedTwoThetaViewData = NewType("IntegratedTwoThetaViewData", sc.DataArray)

ProcessedData = NewType("ProcessedData", sc.DataArray)
