from typing import NewType

import sciline
import scipp as sc

from ..types import Run

WavelengthResolution = NewType('WavelengthResolution', sc.Variable)
AngularResolution = NewType('AngularResolution', sc.Variable)
SampleSizeResolution = NewType('SampleSizeResolution', sc.Variable)


class BeamlineParams(sciline.Scope[Run, dict], dict):
    """Parameters describing the beamline and chopper positions."""


class ChopperFrequency(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Frequency of the choppers used in the run."""


class ChopperPhase(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Phase of the choppers in the run."""


class Chopper1Position(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Position of the first chopper relative the source of the beam."""


class Chopper2Position(sciline.Scope[Run, sc.Variable], sc.Variable):
    """Position of the second chopper relative to the source of the beam."""
