# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeAlias

import scipp as sc
from ess.reduce.nexus.types import IncidentMonitor
from ess.reduce.unwrap.types import WavelengthMonitor

from ..reflectometry.types import RunType

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

WavelengthMonitor: TypeAlias = WavelengthMonitor[RunType, IncidentMonitor]
