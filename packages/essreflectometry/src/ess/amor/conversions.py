# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import IncidentBeam, RunType, SamplePosition
from .types import Chopper1Position, Chopper2Position


def incident_beam(
    source_chopper_1_position: Chopper1Position[RunType],
    source_chopper_2_position: Chopper2Position[RunType],
    sample_position: SamplePosition[RunType],
) -> IncidentBeam[RunType]:
    """
    Compute the incident beam vector from the source chopper position vector,
    instead of the source_position vector.
    """
    chopper_midpoint = (
        source_chopper_1_position + source_chopper_2_position
    ) * sc.scalar(0.5)
    return sample_position - chopper_midpoint


providers = (incident_beam,)
