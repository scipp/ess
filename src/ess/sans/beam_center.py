# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from .common import gravity_vector
from ..logging import get_logger
from .i_of_q import make_coordinate_transform_graphs


def _center_of_mass(data):
    """
    Find the center of mass of the data counts.
    """
    summed = data.sum(list(set(data.dims) - set(data.coords['position'].dims)))
    v = sc.values(summed.data)
    com = sc.sum(summed.coords['position'] * v) / v.sum()
    return com.fields.x, com.fields.y


def _refine(xy, data, graph, q_bins, masking_radius):
    """
    Compute the intensity as a function of Q inside 4 quadrants in Phi.
    Return the sum of the squares of the relative differences between the 4 quadrants.
    """
    da_init = data.copy(deep=False)
    da_init.coords['position'] = data.coords['position'].copy(deep=True)
    u = da_init.coords['position'].unit
    da_init.coords['position'].fields.x -= sc.scalar(xy[0], unit=u)
    da_init.coords['position'].fields.y -= sc.scalar(xy[1], unit=u)
    r = sc.sqrt(da_init.coords['position'].fields.x**2 +
                da_init.coords['position'].fields.y**2)
    da_init.masks['m'] = r > masking_radius
    da_q = da_init.transform_coords('Q', graph=graph)
    da_phi = da_q.transform_coords('phi', graph=graph)
    # Offset the phi angle by 45 degrees to get left,right,top,bottom quandrants.
    pi = sc.constants.pi.value
    phi_offset = sc.scalar(pi / 4, unit='rad')
    da_phi.coords['phi'] += phi_offset
    da_h = da_phi.hist(Q=q_bins)
    phi_bins = sc.linspace('phi', 0, pi * 2, 5, unit='rad') + phi_offset
    da_gr = sc.groupby(da_h, group='phi', bins=phi_bins).sum(
        (set(da_h.dims) - {'Q'}).pop())
    ref = da_gr['phi', 0]
    cost = ((da_gr['phi', 1] - ref)**2 + (da_gr['phi', 2] - ref)**2 +
            (da_gr['phi', 3] - ref)**2) / ref**2
    return cost.sum().value


def beam_center(data: sc.DataArray,
                q_bins,
                masking_radius,
                gravity: bool = False,
                minimizer='Nelder-Mead',
                tolerance=0.1,
                debug=False):
    logger = get_logger('sans')
    if gravity and ('gravity' not in data.coords):
        data = data.copy(deep=False)
        data.coords['gravity'] = gravity_vector()
    # Use center of mass to get initial guess for beam center
    xc, yc = _center_of_mass(data)
    logger.info('Initial guess for beam center: '
                f'x={xc.value}[{xc.unit}], y={yc.value}[{yc.unit}]')
    # Refine using Scipy optimize
    from scipy.optimize import minimize
    graph, _ = make_coordinate_transform_graphs(gravity=gravity, scatter=True)
    x = data.coords['position'].fields.x
    y = data.coords['position'].fields.y
    res = minimize(_refine,
                   x0=[xc.value, yc.value],
                   args=(data, graph, q_bins, masking_radius),
                   bounds=[(x.min().value, x.max().value),
                           (y.min().value, y.max().value)],
                   method=minimizer,
                   tol=tolerance)
    out = [sc.scalar(res.x[0], unit=x.unit), sc.scalar(res.x[1], unit=y.unit)]
    if debug:
        out.append({'initialguess': (xc, yc), 'result': res})
    return out
