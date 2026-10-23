# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc
from ess.reduce.unwrap.types import WavelengthMonitor as _WavelengthMonitor

from ess.reflectometry.types import RunType

BeamMonitor = NewType("BeamMonitor", int)
"""ESTIA beam-flux monitor."""

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

# Plain assignment (not a PEP 695 ``type`` alias): sciline needs the concrete
# generic alias at runtime, and it is subscripted again as
# ``WavelengthMonitor[RunType]`` in providers.
WavelengthMonitor = _WavelengthMonitor[RunType, BeamMonitor]
