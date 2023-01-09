# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scippneutron.conversion.graph import beamline


def group_by_two_theta(data: sc.DataArray,
                       *,
                       edges: sc.Variable,
                       replace_dim: str = 'spectrum') -> sc.DataArray:
    """
    Group data into two_theta bins.

    Parameters
    ----------
    data:
        Input data array with events. Must contain a coord or attr called 'two_theta'
        or coords or attrs that can be used to compute it.
    edges:
        Bin edges in two_theta. `data` is grouped into those bins.
    replace_dim:
        Dimension that is replaced by two_theta.
        All events are concatenated along this dimension.

    Returns
    -------
    :
        `data` grouped into two_theta bins.
    """
    data = data.transform_coords('two_theta', graph=beamline.beamline(scatter=True))
    return sc.groupby(data,
                      'two_theta',
                      bins=edges.to(unit=data.coords['two_theta'].unit,
                                    copy=False)).bins.concat(replace_dim)
