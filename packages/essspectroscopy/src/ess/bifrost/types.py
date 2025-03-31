# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.spectroscopy.types import RunType


class ArcNumber(sciline.Scope[RunType, sc.Variable], sc.Variable): ...


# See https://github.com/scipp/essreduce/issues/105 about monitor names
FrameMonitor0 = reduce_t.Monitor1
FrameMonitor1 = reduce_t.Monitor2
FrameMonitor2 = reduce_t.Monitor3
FrameMonitor3 = reduce_t.Monitor4


class McStasDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...
