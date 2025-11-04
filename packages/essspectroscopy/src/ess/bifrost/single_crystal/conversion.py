# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Coordinate conversions for single crystal diffraction with BIFROST."""

import scippnexus as snx
from scippneutron.conversion.graph import beamline as beamline_graphs
from scippneutron.conversion.graph import tof as tof_graphs

from ess.spectroscopy.indirect.conversion import (
    rotate_to_sample_table_momentum_transfer,
)
from ess.spectroscopy.types import (
    ElasticCoordTransformGraph,
    GravityVector,
    Position,
    QDetector,
    RunType,
    TofDetector,
)


def single_crystal_coordinate_transformation_graph(
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> ElasticCoordTransformGraph[RunType]:
    """Return the coordinate transformation graph for single crystal diffraction."""
    base = tof_graphs.elastic_Q_vec(start='tof')
    base['lab_momentum_transfer'] = base['Q_vec']
    return ElasticCoordTransformGraph[RunType](
        {
            **beamline_graphs.beamline(scatter=True),
            **base,
            'sample_position': lambda: sample_position,
            'source_position': lambda: source_position,
            'gravity': lambda: gravity,
            'sample_table_momentum_transfer': rotate_to_sample_table_momentum_transfer,
        }
    )


def convert_tof_to_q(
    with_tof: TofDetector[RunType],
    *,
    graph: ElasticCoordTransformGraph[RunType],
) -> QDetector[RunType]:
    """Convert ToF to Q."""
    transformed = with_tof.transform_coords(
        ['a3', 'sample_table_momentum_transfer'],
        graph=graph,
        keep_intermediate=False,
        keep_inputs=False,
        keep_aliases=False,
        rename_dims=False,  # because otherwise, it would rename a3 -> Q
    )
    return QDetector[RunType](transformed)


providers = (single_crystal_coordinate_transformation_graph, convert_tof_to_q)
