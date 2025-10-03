# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for single crystal diffraction on BIFROST."""

from dataclasses import dataclass

import sciline
import scipp as sc

from ess.spectroscopy.types import RunType


class CountsWithQMapCoords(sciline.Scope[RunType, sc.DataArray], sc.DataArray): ...


@dataclass(frozen=True, slots=True)
class QProjection:
    """Projection vectors in Q."""

    parallel: sc.Variable
    perpendicular: sc.Variable
