# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType

import sciline
import scipp as sc
from ess.reduce.unwrap.types import WavelengthMonitor as _WavelengthMonitor

from ..reflectometry.types import RunType

WBC1Monitor = NewType("WBC1Monitor", int)
"""Bunker monitor for wavelength-band chopper WBC1 diagnostics."""
PSCMonitor = NewType("PSCMonitor", int)
"""Bunker monitor for pulse-shaping choppers PSC1-3 diagnostics."""
WBC2Monitor = NewType("WBC2Monitor", int)
"""Out-of-bunker monitor for WBC2 and PSC4 diagnostics."""
WBC3Monitor = NewType("WBC3Monitor", int)
"""Out-of-bunker monitor for WBC3 and PSC5 diagnostics."""
ShutterMonitor = NewType("ShutterMonitor", int)
"""Pre-sample monitor for beam profiling and fast-shutter diagnostics."""
NormalizationMonitor = NewType("NormalizationMonitor", int)
"""Pre-sample flux monitor for normalization."""

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

# Plain assignment (not a PEP 695 ``type`` alias): sciline needs the concrete
# generic alias at runtime, and it is subscripted again as
# ``WavelengthMonitor[RunType]`` in providers.
WavelengthMonitor = _WavelengthMonitor[RunType, NormalizationMonitor]


class SampleSurfaceNormal(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Normal pointing out of the reflecting surface, in global coordinates."""


class MirrorPosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """A point on the inversion mirror's reflecting plane, in global coordinates."""


class MirrorNormal(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Normal to the inversion mirror in global coordinates; either sign is valid."""


@dataclass(frozen=True)
class MirrorGeometry[RunType]:
    """A point on the mirror plane and its unit normal, in laboratory coordinates."""

    position: sc.Variable
    normal: sc.Variable


class UpstreamSlitCenters(sciline.Scope[RunType, dict], dict):
    """Beam names mapped to fixed slit centers before the inversion mirror.

    Positions are in global coordinates, also when inversion is disabled.
    """


class DownstreamSlitCenters(sciline.Scope[RunType, dict], dict):
    """Beam names mapped to fixed slit centers after the inversion mirror.

    Positions are in global coordinates, also when inversion is disabled.
    """


class DetectorRegionOfInterest(sciline.Scope[RunType, dict], dict):
    """Pixel or event coordinates mapped to inclusive (lower, upper) bounds.

    Select corresponding reflected and direct peaks separately for SampleRun and
    ReferenceRun, for example using ``scattering_angle`` and ``height``.
    An empty dictionary explicitly selects the entire detector.
    """


class IncidentAngleGraph(sciline.Scope[RunType, dict], dict):
    """Coordinate graph for assigning incident angles and correcting for gravity."""


class ShutterPositions(sciline.Scope[RunType, dict], dict):
    """Incident beam names mapped to fixed shutter positions in global coordinates.

    Keys match UpstreamSlitCenters and DownstreamSlitCenters. Each beam has one
    shutter, whose position is independent of the slit centers used for angles.
    In inversion mode, shutters must be upstream of the mirror.
    """


class ShutterStatusLogs(sciline.Scope[RunType, dict], dict):
    """Shutter identifiers mapped to status logs.

    Each log contains boolean data (True means open, False means closed) and a
    datetime ``time`` coordinate.
    """
