# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for BIFROST.

This module supplements :mod:`ess.spectroscopy.types` with BIFROST-specific types.
"""

from typing import NewType

import sciline
import scipp as sc

from ess.spectroscopy.types import RunType


class ArcNumber(sciline.Scope[RunType, sc.Variable], sc.Variable): ...


class McStasRawDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


ArcEnergy = NewType('ArcEnergy', sc.Variable)
