# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scipp.typing import VariableLike


def event_or_outer_coord(da: sc.DataArray, name: str) -> sc.Variable:
    """Return either an event coord or an outer coord with a given name."""
    try:
        return da.bins.coords[name]
    except (AttributeError, KeyError):
        # Either not binned or no event coord with this name.
        return da.coords[name]


def elem_unit(var: VariableLike) -> sc.Unit:
    return var.bins.unit if var.bins is not None else var.unit


def elem_dtype(var: VariableLike) -> sc.DType:
    return var.bins.constituents['data'].dtype if var.bins is not None else var.dtype
