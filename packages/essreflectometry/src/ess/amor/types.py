# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc

from ..reflectometry.types import RunType

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)


class ChopperFrequency(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Frequency of the choppers used in the run."""


class ChopperPhase(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Phase of the choppers in the run."""


class ChopperDistance(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Distance from the midpoint between the two choppers to the sample."""


class ChopperSeparation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Distance between the two choppers."""


class RawChopper(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Chopper data loaded from nexus file."""


GravityToggle = NewType("GravityToggle", bool)
