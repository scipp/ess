# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

MaxNumberOfPixelsPerAxis = NewType("MaxNumberOfPixelsPerAxis", int)
PixelStep = NewType("PixelStep", int)
NumberOfDetectors = NewType("NumberOfDetectors", int)
NumberOfAxes = NewType("NumberOfAxes", int)


default_params = {
    MaxNumberOfPixelsPerAxis: MaxNumberOfPixelsPerAxis(1280),
    NumberOfAxes: NumberOfAxes(2),
    PixelStep: PixelStep(1),
    NumberOfDetectors: NumberOfDetectors(3),
}
