import sciline as sl
import scipp as sc
from typing import TypeVar, NewType
from dataclasses import dataclass

from ess.packages.essdiffraction.src.ess.powder.types import (
    RunType,
)


class DspacingCylSurfaceDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with time-of-flight coordinate."""


class CylindricalSurfaceTransformGraph(sl.Scope[dict], dict):
    """Graph for transforming projecting 3D POWTEX data onto a cylindrical surface."""


class Texture3D(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """ToDo."""


class View1D(sl.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """ToDo."""


class View2D(sl.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """ToDo."""


class View3D(sl.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """ToDo."""


# binning settings types
GammaCutStart = NewType("GammaCutStart", sc.Variable)
GammaCutEnd = NewType("GammaCutEnd", sc.Variable)
DropPartialBins = NewType("DropPartialBins", bool)
IncludeGapBins = NewType("IncludeGapBins", bool) # maybe not needed in future

GammaBinWidth = NewType("GammaBinWidth", sc.Variable)
TwoThetaBinWidth = NewType("TwoThetaBinWidth", sc.Variable)
DspacingBinWidth = NewType("DspacingBinWidth", sc.Variable)

GammaBinningData = NewType("GammaBinningData", sc.DataGroup)
TwoThetaBinningData = NewType("TwoThetaBinningData", sc.Variable)
DspacingBinningData = NewType("DspacingBinningData", sc.Variable)


@dataclass #(frozen=True) <-- frozen makes it impossible to change the values of the fields after initialization
class BinningSettings:
    gamma_cut_start: sc.Variable
    gamma_cut_end: sc.Variable
    drop_partial_bins: bool
    include_gamma_gap_bins: bool
    two_theta_min: sc.Variable
    two_theta_max: sc.Variable
