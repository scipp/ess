# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Conversions from validated parameter models to scipp objects.

Consumed by workflow implementations only; kept out of
:mod:`ess.reduce.spec.parameters` so the parameter vocabulary itself stays
free of scipp and serializes cleanly to JSON Schema.
"""

import scipp as sc

from .parameters import EdgesModel, RangeModel, Scale


def edges_to_variable(edges: EdgesModel, dim: str) -> sc.Variable:
    """Return the bin edges described by the model as a scipp variable."""
    op = {Scale.LINEAR: sc.linspace, Scale.LOG: sc.geomspace}[edges.scale]
    return op(
        dim=dim,
        start=edges.start,
        stop=edges.stop,
        num=edges.num_bins + 1,
        unit=str(edges.unit),
    )


def range_to_variables(range_: RangeModel) -> tuple[sc.Variable, sc.Variable]:
    """Return the range bounds as a pair of scipp scalars."""
    unit = str(range_.unit)
    return sc.scalar(range_.start, unit=unit), sc.scalar(range_.stop, unit=unit)
