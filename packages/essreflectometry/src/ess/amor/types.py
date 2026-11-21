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
    """Phase of the choppers in the run.

    Only used by :func:`ess.amor.AmorWorkflow2025`.
    """


class ChopperPhaseOffset(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Phase between the chopper trigger signal and the opening of the chopper."""


class ChopperDistance(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Distance from the midpoint between the two choppers to the sample."""


class ChopperSeparation(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Distance between the two choppers."""


class ProtonCurrent(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Log of the proton current on the target of the neutron source."""


class PulseTimes(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Start time of every chopper revolution of the run.

    The events are recorded relative to these times.
    Revolutions without events are included, in contrast to the
    ``event_time_zero`` of the event data.
    """


ProtonCurrentThreshold = NewType("ProtonCurrentThreshold", sc.Variable)
"""Proton current below which the neutrons of a chopper revolution are discarded."""

GravityToggle = NewType("GravityToggle", bool)
