# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

WavelengthResolution = NewType("WavelengthResolution", sc.Variable)
AngularResolution = NewType("AngularResolution", sc.Variable)
SampleSizeResolution = NewType("SampleSizeResolution", sc.Variable)

CoordTransformationGraph = NewType("CoordTransformationGraph", dict)
