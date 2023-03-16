# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from typing import Optional

import numpy as np
import scipp as sc
from scipp.constants import g


def gravity_vector() -> sc.Variable:
    """
    Return a vector of 3 components, defining the magnitude and direction of the Earth's
    gravitational field.
    """
    return sc.vector(value=[0, -1, 0]) * g


def mask_range(da: sc.DataArray,
               edges: sc.Variable,
               mask: sc.Variable,
               name: Optional[str] = None) -> sc.DataArray:
    """
    Mask a range on a data array.
    The provided edges are used to define the ranges to be masked.

    Parameters
    ----------
    da:
        The data array to be masked.
    edges:
        The edges of the ranges to be masked.
    mask:
        The mask values to be applied within the ranges defined by ``edges``.
    name:
        The name of the mask to be applied. If not provided, a random name will be used.

    Returns
    -------
    :
        A copy of the input data array with the mask applied.
    """
    dim = edges.dim
    if edges.sizes[dim] != mask.sizes[dim] + 1:
        raise RuntimeError(
            f"Size of {dim} in edges ({edges.sizes[dim]}) does not match "
            f"size of {dim} in mask ({mask.sizes[dim]})")
    if name is None:
        name = uuid.uuid4().hex
    lu = sc.DataArray(data=mask, coords={dim: edges})
    if da.bins is not None:
        if dim in da.coords:
            new_bins = sc.array(dims=[dim],
                                values=np.union1d(edges.values, da.coords[dim].values),
                                unit=edges.unit)
            out = da.bin({dim: new_bins})
            out.masks[name] = sc.lookup(lu, dim)[sc.midpoints(new_bins, dim=dim)]
        else:
            out = da.bin({dim: edges})
            out.masks[name] = mask
    else:
        out = da.copy(deep=False)
        mask_values = sc.lookup(lu, dim)[da.coords[dim]]
        if da.coords.is_edges(dim):
            out.masks[name] = mask_values[dim, 1:] | mask_values[dim, :-1]
        else:
            out.masks[name] = mask_values
    return out
