# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeAlias

import scipp as sc
from ess.reduce.unwrap.types import WavelengthMonitor as _WavelengthMonitor

from ess.reflectometry.types import RunType

CaveMonitor = NewType("CaveMonitor", int)

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

WavelengthMonitor: TypeAlias = _WavelengthMonitor[RunType, CaveMonitor]
