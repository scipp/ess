# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc

from .common import gravity_vector
from ..logging import get_logger
from .i_of_q import make_coordinate_transform_graphs


def _center_of_mass(data, nx=100, ny=100):
    """
    Find the center of mass of the data counts.
    """
    da = data.copy(deep=False)
    da.coords['x'] = da.coords['position'].fields.x
    da.coords['y'] = da.coords['position'].fields.y
    da_hist = da.bin(y=ny, x=nx).hist()
    da_summed = da_hist.sum(list(set(da_hist.dims) - {'x', 'y'}))
    xb = sc.broadcast(sc.midpoints(da_summed.coords['x']), sizes=da_summed.sizes)
    yb = sc.broadcast(sc.midpoints(da_summed.coords['y']), sizes=da_summed.sizes)
    ncounts = da_summed.data.sum()
    xc = sc.sum(xb * da_summed.data) / ncounts
    yc = sc.sum(yb * da_summed.data) / ncounts
    return xc, yc


def _refinement(xy, data, graph, q_bins):
    """
    Compute the intensity as a function of Q inside 4 quadrants in Phi.
    Return the sum of the squares of the relative differences between the 4 quadrants.
    """
    print('xy=', xy)
    da_init = data.copy(deep=False)
    da_init.coords['position'] = data.coords['position'].copy(deep=True)
    u = da_init.coords['position'].unit
    da_init.coords['position'].fields.x -= sc.scalar(xy[0], unit=u)
    da_init.coords['position'].fields.y -= sc.scalar(xy[1], unit=u)
    da_q = da_init.transform_coords('Q', graph=graph)
    da_phi = da_q.transform_coords('phi', graph=graph)
    da_phi.coords['phi'] += sc.scalar(sc.constants.pi.value / 4, unit='rad')
    binned = da_phi.bin(phi=4)
    summed = binned.hist(Q=q_bins)
    # ref = summed['phi', 0]
    # cost = ((summed['phi', 1] - ref)**2 + (summed['phi', 2] - ref)**2 +
    #         (summed['phi', 3] - ref)**2) / ref**2
    cost = ((summed['phi', 2] - summed['phi', 0])**2 / summed['phi', 0]**2 +
            (summed['phi', 3] - summed['phi', 1])**2 / summed['phi', 1]**2)
    print(cost.sum().value)
    return cost.sum().value


def beam_center(data: sc.DataArray,
                q_bins,
                gravity: bool = False,
                minimizer='Nelder-Mead'):
    logger = get_logger('sans')
    if gravity and ('gravity' not in data.coords):
        data = data.copy(deep=False)
        data.coords['gravity'] = gravity_vector()
    # Use center of mass to get initial guess for beam center
    xc, yc = _center_of_mass(data)
    logger.info('Initial guess for beam center: '
                f'x={xc.value}[{xc.unit}], y={yc.value}[{yc.unit}]')
    print(xc, yc)
    # Refine using Scipy optimize
    from scipy.optimize import minimize
    graph, _ = make_coordinate_transform_graphs(gravity=gravity, scatter=True)
    x = data.coords['position'].fields.x
    y = data.coords['position'].fields.y
    res = minimize(_refinement,
                   x0=[xc.value, yc.value],
                   args=(data, graph, q_bins),
                   bounds=[(x.min().value, x.max().value),
                           (y.min().value, y.max().value)],
                   method=minimizer)
    print(res)
    return sc.scalar(res.x[0], unit=x.unit), sc.scalar(res.x[1], unit=y.unit)
