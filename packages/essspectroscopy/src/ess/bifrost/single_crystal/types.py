# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for single crystal diffraction on BIFROST."""

from dataclasses import dataclass
from typing import NewType

import sciline
import scipp as sc

from ess.spectroscopy.types import RunType


class CountsWithQMapCoords(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Counts with various Q coordinates.

    Has (event) coordinates:

    - 'Q_parallel': projection along the beam direction, aka 'Qz'
    - 'Q_perpendicular': perpendicular component of Q to the beam, aka 'Qx'
    - 'Q': absolute value of Q
    """


@dataclass(frozen=True, slots=True)
class QProjection:
    """Projection vectors in Q."""

    parallel: sc.Variable
    perpendicular: sc.Variable


QParallelBins = NewType("QParallelBins", sc.Variable)
"""Binning in Q_parallel (aka Qz)."""

QPerpendicularBins = NewType("QPerpendicularBins", sc.Variable)
"""Binning in Q_perpendicular (aka Qx)."""


class IntensityQparQperp(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Histogrammed data in Q_parallel and Q_perpendicular.

    Here, Q_parallel is the component of Q along the beam direction and Q_perpendicular
    is the non-zero component of Q perpendicular to the beam direction.
    For BIFROST, these are Qz and Qx respectively. (Qy is zero because the
    detector is in the x-z plane with the sample.)
    """


class IntensitySampleRotation(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data integrated over |Q| and histogrammed in the sample rotation angle."""


QRange = NewType("QRange", tuple[sc.Variable, sc.Variable])
"""Range of |Q| to integrate over."""

SampleRotationBins = NewType("SampleRotationBins", sc.Variable)
"""Binning in the sample rotation angle."""
