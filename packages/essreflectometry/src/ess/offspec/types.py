# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t

from ..reflectometry.types import RunType

SpectrumLimits = NewType("SpectrumLimits", tuple[sc.Variable, sc.Variable])
BackgroundMinWavelength = NewType("BackgroundMinWavelength", sc.Variable)


class CoordTransformationGraph(sciline.Scope[RunType, dict], dict):
    """Coordinate transformation for the runtype"""


class MonitorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """ "Monitor data from the run file, with background subtracted"""


NeXusMonitorName = reduce_t.NeXusName
