# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
The scipp side of the spec vocabulary.

Conversions from validated parameter models to scipp objects, and the
structural check of a scipp object against an :class:`ArraySpec`. Consumed by
workflow implementations and runners only; kept out of the rest of
:mod:`ess.reduce.spec` so the vocabulary itself stays free of scipp and
serializes cleanly to JSON Schema.
"""

import scipp as sc

from .data import ArraySpec
from .parameters import EdgesModel, RangeModel, Scale


def check_array(value: sc.Variable | sc.DataArray, spec: ArraySpec) -> None:
    """
    Raise ``ValueError`` unless ``value`` has the structure ``spec`` declares.

    Pydantic cannot inspect a scipp object, so a runner calls this on array
    outputs at completion, and may on the arrays it resolves for a workflow.
    """
    problems = []
    if tuple(value.dims) != spec.dims:
        problems.append(f'dims {value.dims} != {spec.dims}')
    unit = None if spec.unit is None else sc.Unit(spec.unit)
    if value.unit != unit:
        problems.append(f'unit {value.unit} != {unit}')
    if (value.bins is not None) != spec.binned:
        problems.append(f'binned={value.bins is not None} != {spec.binned}')
    coords = value.coords if isinstance(value, sc.DataArray) else {}
    for name, coord_unit in spec.coords.items():
        if name not in coords:
            problems.append(f'missing coord {name!r}')
            continue
        expected = None if coord_unit is None else sc.Unit(coord_unit)
        if coords[name].unit != expected:
            problems.append(f'coord {name!r} unit {coords[name].unit} != {expected}')
    if problems:
        raise ValueError('array does not match its spec: ' + '; '.join(problems))


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
