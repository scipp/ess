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


@dataclass(frozen=True)
class MirrorGeometry[RunType]:
    """A point on the mirror plane and its unit normal, in laboratory coordinates.

    Both inputs must be finite scalar vectors. The normal is normalized on
    construction and must be nonzero; either sign is valid.
    """

    position: sc.Variable
    normal: sc.Variable

    def __post_init__(self) -> None:
        for name, vector in [('position', self.position), ('normal', self.normal)]:
            if vector.ndim != 0 or vector.dtype != sc.DType.vector3:
                raise ValueError(f'Mirror {name} must be a scalar vector.')
            if not sc.isfinite(vector).value:
                raise ValueError(f'Mirror {name} must be finite.')
        length = sc.norm(self.normal)
        if not sc.isfinite(length).value or length.value == 0:
            raise ValueError('Mirror normal must have a finite, nonzero norm.')
        object.__setattr__(self, 'normal', self.normal / length)


class UpstreamSlitCenters(sciline.Scope[RunType, dict], dict):
    """Beam names mapped to upstream slit centers in global coordinates."""


class DownstreamSlitCenters(sciline.Scope[RunType, dict], dict):
    """Beam names mapped to downstream slit centers in global coordinates."""


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
