from typing import Any, NewType

import sciline
import scipp as sc

from ..reflectometry.types import RunType

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

CoordTransformationGraph = NewType("CoordTransformationGraph", dict)


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


class ThetaBins(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Binning in theta that takes into consideration that some
    detector pixels have the same theta value."""


class AngleCenterOfIncomingToHorizon(
    sciline.Scope[RunType, sc.DataGroup], sc.DataGroup
):
    """Angle from the center of the incoming beam to the horizon."""


WavelengthThetaFigure = NewType("WavelengthThetaFigure", Any)
WavelengthZIndexFigure = NewType("WavelengthZIndexFigure", Any)
QThetaFigure = NewType("QThetaFigure", Any)
ReflectivityDiagnosticsView = NewType("ReflectivityDiagnosticsView", Any)

GravityToggle = NewType("GravityToggle", bool)
