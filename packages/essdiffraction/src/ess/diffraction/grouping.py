# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scippneutron.conversion.graph import beamline


def group_by_two_theta(data: sc.DataArray, *, edges: sc.Variable) -> sc.DataArray:
    """
    Group data into two_theta bins.

    Parameters
    ----------
    data:
        Input data array with events. Must contain a coord called 'two_theta'
        or coords that can be used to compute it.
    edges:
        Bin edges in two_theta. `data` is grouped into those bins.

    Returns
    -------
    :
        `data` grouped into two_theta bins.
    """
    out = data.transform_coords('two_theta', graph=beamline.beamline(scatter=True))
    return out.bin(two_theta=edges.to(unit=out.coords['two_theta'].unit, copy=False))
