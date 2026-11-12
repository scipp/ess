# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for BIFROST.

This module supplements :mod:`ess.spectroscopy.types` with BIFROST-specific types.
"""

from typing import NewType

import sciline
import scipp as sc
import scippnexus as snx
from ess.spectroscopy.types import NeXusMultiNames, RunType


class ArcNumber(sciline.Scope[RunType, sc.Variable], sc.Variable): ...


class McStasRawDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


ArcEnergy = NewType('ArcEnergy', sc.Variable)


NeXusDetectorNames = NeXusMultiNames[snx.NXdetector]


InstrumentAnglePath = NewType('InstrumentAnglePath', str)
"""Absolute NeXus path of the detector tank rotation.

This must be a transformation in the depends-on chain of the elastic monitor.
"""

SampleAngleNeXusName = NewType('SampleAngleNeXusName', str)
"""(Part of) the NeXus path of the sample angle.

This is used to find the NXtransformation in the sample's transformation chain
that corresponds to the sample rotation.
Can be a complete nexus path, or just part of the path.
"""
