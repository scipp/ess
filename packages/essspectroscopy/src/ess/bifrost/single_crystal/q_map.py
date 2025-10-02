# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Build a Q-map for single crystal diffraction."""

from collections.abc import Callable

import scipp as sc

from ess.spectroscopy.types import DetectorCountsWithQ, RunType

from .types import QBinsParallel, QBinsPerpendicular, QMap, QProjection


def project_onto_q_plane(
    counts: DetectorCountsWithQ[RunType],
    *,
    q_projection: QProjection,
    parallel_bins: QBinsParallel,
    perpendicular_bins: QBinsPerpendicular,
) -> QMap[RunType]:
    transformed = counts.transform_coords(
        ['Q_parallel', 'Q_perpendicular', 'Q'],
        graph={
            'Q_parallel': _make_projection(q_projection.parallel),
            'Q_perpendicular': _make_projection(q_projection.perpendicular),
            'Q': lambda sample_table_momentum_transfer: sc.norm(
                sample_table_momentum_transfer
            ),
        },
        keep_inputs=False,
    )
    if transformed.bins is not None:
        transformed.bins.coords['a3'] = sc.bins_like(
            transformed, transformed.coords['a3']
        )
        transformed = transformed.bins.concat()

    binned = transformed.bin(
        Q_perpendicular=perpendicular_bins, Q_parallel=parallel_bins
    )
    return QMap[RunType](binned)


def _make_projection(vector: sc.Variable) -> Callable[..., sc.Variable]:
    def projection(sample_table_momentum_transfer: sc.Variable) -> sc.Variable:
        return sc.dot(sample_table_momentum_transfer, vector / sc.norm(vector))

    return projection


def default_q_projection() -> QProjection:
    return QProjection(
        parallel=sc.vector(value=[0, 0, 1]), perpendicular=sc.vector(value=[1, 0, 0])
    )


providers = (default_q_projection, project_onto_q_plane)
