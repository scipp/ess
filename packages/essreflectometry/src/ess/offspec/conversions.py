# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from scippneutron.conversion.graph import beamline, tof

from ..reflectometry.types import (
    ReferenceRun,
    SampleRun,
)
from .types import CoordTransformationGraph


def coordinate_transformation_graph_sample() -> CoordTransformationGraph[SampleRun]:
    return {
        **beamline.beamline(scatter=True),
        **tof.elastic_wavelength("tof"),
    }


def coordinate_transformation_graph_reference() -> CoordTransformationGraph[
    ReferenceRun
]:
    return {
        **beamline.beamline(scatter=False),
        **tof.elastic_wavelength("tof"),
    }


providers = (
    coordinate_transformation_graph_sample,
    coordinate_transformation_graph_reference,
)
