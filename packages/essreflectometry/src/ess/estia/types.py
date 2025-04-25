# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import sciline
import scipp as sc

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

OffOff = NewType("OffOff", str)
OffOn = NewType("OffOn", str)
OnOff = NewType("OnOff", str)
OnOn = NewType("OnOn", str)
FlipperSetting = TypeVar("FlipperSetting", OffOff, OffOn, OnOff, OnOn)

MagneticSample = NewType("MagneticSample", str)
MagneticReference = NewType("MagneticReference", str)
NonMagneticReference = NewType("NonMagneticReference", str)
PolarizedRunType = TypeVar(
    "PolarizedRunType", MagneticSample, MagneticReference, NonMagneticReference
)


class Intensity(
    sciline.Scope[PolarizedRunType, FlipperSetting, sc.DataArray], sc.DataArray
):
    """Intensity distribution"""


PolarizedReflectivityOverQ = NewType("PolarizedReflectivityOverQ", list[sc.DataArray])
