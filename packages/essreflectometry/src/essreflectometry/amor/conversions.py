# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..conversions import specular_reflection as spec_relf_graph
from ..types import SpecularReflectionCoordTransformGraph


def incident_beam(
    *,
    source_chopper_1: sc.Variable,
    source_chopper_2: sc.Variable,
    sample_position: sc.Variable,
) -> sc.Variable:
    """
    Compute the incident beam vector from the source chopper position vector,
    instead of the source_position vector.
    """
    chopper_midpoint = (
        source_chopper_1.value['position'].data
        + source_chopper_2.value['position'].data
    ) * sc.scalar(0.5)
    return sample_position - chopper_midpoint


def specular_reflection() -> SpecularReflectionCoordTransformGraph:
    """
    Generate a coordinate transformation graph for Amor reflectometry.
    """
    graph = spec_relf_graph()
    graph['incident_beam'] = incident_beam
    return SpecularReflectionCoordTransformGraph(graph)


providers = [specular_reflection]
