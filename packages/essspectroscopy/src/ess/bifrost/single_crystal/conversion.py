# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Coordinate conversions for single crystal diffraction with BIFROST."""

from scippneutron.conversion.graph import beamline as beamline_graphs
from scippneutron.conversion.graph import tof as tof_graphs

from ess.spectroscopy.indirect.conversion import (
    rotate_to_sample_table_momentum_transfer,
)
from ess.spectroscopy.types import (
    DetectorCountsWithQ,
    DetectorTofData,
    ElasticCoordTransformGraph,
    GravityVector,
    RunType,
)


def single_crystal_coordinate_transformation_graph(
    gravity: GravityVector,
) -> ElasticCoordTransformGraph:
    """Return the coordinate transformation graph for single crystal diffraction."""
    base = tof_graphs.elastic_Q_vec(start='tof')
    base['lab_momentum_transfer'] = base['Q_vec']
    return ElasticCoordTransformGraph(
        {
            **beamline_graphs.beamline(scatter=True),
            **base,
            'gravity': lambda: gravity,
            'sample_table_momentum_transfer': rotate_to_sample_table_momentum_transfer,
        }
    )


def convert_tof_to_q(
    with_tof: DetectorTofData[RunType],
    *,
    graph: ElasticCoordTransformGraph,
) -> DetectorCountsWithQ[RunType]:
    """Convert ToF to Q."""
    transformed = with_tof.transform_coords(
        ['a3', 'sample_table_momentum_transfer'],
        graph=graph,
        keep_intermediate=False,
        keep_inputs=False,
        keep_aliases=False,
        rename_dims=False,  # because otherwise, it would rename a3 -> Q
    )
    return DetectorCountsWithQ[RunType](transformed)


providers = (single_crystal_coordinate_transformation_graph, convert_tof_to_q)
