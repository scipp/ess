# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.spatial import rotations_from_rotvecs
from scippneutron.conversion.graph import beamline, tof

from ..reflectometry.types import (
    ReferenceRun,
    SampleRun,
)
from .types import CoordTransformationGraph


def adjust_pixel_positions_for_sample(data: sc.DataArray):
    rotation = rotations_from_rotvecs(
        rotation_vectors=sc.vector(
            value=[-2.0 * data.coords['theta'].value, 0, 0], unit=sc.units.deg
        )
    )
    return data.assign_coords(
        position=rotation * (data.coords['position'] - data.coords['sample_position'])
    )


def coordinate_transformation_graph_sample() -> CoordTransformationGraph[SampleRun]:
    return {
        **beamline.beamline(scatter=True),
        **tof.elastic_wavelength("tof"),
    }


def coordinate_transformation_graph_reference() -> (
    CoordTransformationGraph[ReferenceRun]
):
    return {
        **beamline.beamline(scatter=False),
        **tof.elastic_wavelength("tof"),
    }


providers = (
    coordinate_transformation_graph_sample,
    coordinate_transformation_graph_reference,
)
