# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for single crystal diffraction on BIFROST."""

from dataclasses import dataclass
from typing import NewType

import sciline
import scipp as sc

from ess.spectroscopy.types import RunType


class CountsWithQMapCoords(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


@dataclass(frozen=True, slots=True)
class QProjection:
    """Projection vectors in Q."""

    parallel: sc.Variable
    perpendicular: sc.Variable


QParallelBins = NewType("QParallelBins", sc.Variable)
QPerpendicularBins = NewType("QPerpendicularBins", sc.Variable)


class Q2dHistogram(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


class QHistogram(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


QRange = NewType("QRange", tuple[sc.Variable, sc.Variable])

SampleRotationBins = NewType("SampleRotationBins", sc.Variable)
