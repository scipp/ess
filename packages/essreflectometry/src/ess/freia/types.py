# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc
from ess.reduce.unwrap.types import WavelengthMonitor as _WavelengthMonitor

from ..reflectometry.types import RunType

IncidentMonitor = NewType("IncidentMonitor", int)

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

# Plain assignment (not a PEP 695 ``type`` alias): sciline needs the concrete
# generic alias at runtime, and it is subscripted again as
# ``WavelengthMonitor[RunType]`` in providers.
WavelengthMonitor = _WavelengthMonitor[RunType, IncidentMonitor]


class SampleSurfaceNormal(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Normal pointing out of the reflecting surface, in global coordinates."""


class DetectorRegionOfInterest(sciline.Scope[RunType, dict], dict):
    """Pixel or event coordinates mapped to inclusive (lower, upper) bounds.

    Select corresponding reflected and direct peaks separately for SampleRun and
    ReferenceRun, for example using ``scattering_angle`` and ``height``.
    An empty dictionary explicitly selects the entire detector.
    """
