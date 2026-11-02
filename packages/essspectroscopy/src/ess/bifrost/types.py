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

InstrumentAnglePath = NewType('InstrumentAnglePath', str)
"""Absolute NeXus path of the detector tank rotation.

This must be a transformation in the depends-on chain of the elastic monitor.
"""

SampleAnglePath = NewType('SampleAnglePath', str)
"""Absolute NeXus path of the ``NXlog`` (or ``NXpositioner``) of the sample rotation."""
