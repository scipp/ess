# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict

import numpy as np
import scipp as sc

from ..logging import get_logger
from . import i_of_q
from .common import gravity_vector
from .conversions import sans_elastic
from .normalization import normalize


def _center_of_mass(data):
    """
    Find the center of mass of the data counts.
    """
    summed = data.sum(list(set(data.dims) - set(data.meta['position'].dims)))
    v = sc.values(summed.data)
    com = sc.sum(summed.meta['position'] * v) / v.sum()
    return com.fields.x, com.fields.y


def _cost(xy, data):
    """
    Cost function for determining how close the I(Q) curves are in all four quadrants.
    """
    ref = data['right']
    c = ((data['top'] - ref)**2 + (data['left'] - ref)**2 +
         (data['bottom'] - ref)**2) / ref**2
    out = c.sum().value
    print(xy, out)
    return out


def _refine(xy, sample, denominator, graph, q_bins, masking_radius, gravity,
            wavelength_bands):
    """
    Compute the intensity as a function of Q inside 4 quadrants in Phi.
    Return the sum of the squares of the relative differences between the 4 quadrants.
    """
    # Make a copy of the original data
    data = sc.DataArray(data=sample.data)
    coord_list = ['position', 'sample_position', 'source_position']
    for c in coord_list:
        data.coords[c] = sample.meta[c].copy(deep=True)
    # Offset the position according to the initial guess from the center-of-mass
    u = data.coords['position'].unit
    data.coords['position'].fields.x -= sc.scalar(xy[0], unit=u)
    data.coords['position'].fields.y -= sc.scalar(xy[1], unit=u)
    # Add the circular mask
    r = sc.sqrt(data.coords['position'].fields.x**2 +
                data.coords['position'].fields.y**2)
    data.masks['circle'] = r > masking_radius

    # Insert a copy of coords and masks needed for conversion to Q
    for c in coord_list:
        denominator.coords[c] = data.coords[c]
    denominator.masks['circle'] = data.masks['circle']

    pi = sc.constants.pi.value
    phi_offset = sc.scalar(pi / 4, unit='rad')
    phi = (sc.atan2(y=data.coords['position'].fields.y,
                    x=data.coords['position'].fields.x) +
           phi_offset) % (2 * (pi * sc.units.rad))
    phi_bins = sc.linspace('phi', 0, pi * 2, 5, unit='rad')
    quadrants = ['right', 'top', 'left', 'bottom']

    out = {}
    for i, quad in enumerate(quadrants):
        # Select pixels based on phi
        sel = (phi >= phi_bins[i]) & (phi < phi_bins[i + 1])
        # Data counts into Q bins
        data_q = i_of_q.convert_to_q_and_merge_spectra(
            data=data[sel],
            graph=graph,
            q_bins=q_bins,
            gravity=gravity,
            wavelength_bands=wavelength_bands)
        # Denominator counts into Q bins
        denominator_q = i_of_q.convert_to_q_and_merge_spectra(
            data=denominator[sel],
            graph=graph,
            q_bins=q_bins,
            gravity=gravity,
            wavelength_bands=wavelength_bands)
        # Normalize
        out[quad] = normalize(numerator=data_q, denominator=denominator_q).hist()
    # Compute cost
    cost = _cost(xy=xy, data=out)
    if not np.isfinite(cost):
        raise ValueError('Non-finite value computed in cost. This is likely due to a '
                         'division by zero. Try increasing the size of your Q bins to '
                         'improve statistics in the denominator.')
    return cost


def beam_center(data: sc.DataArray,
                data_monitors: Dict[str, sc.DataArray],
                direct_monitors: Dict[str, sc.DataArray],
                direct_beam: sc.DataArray,
                wavelength_bins: sc.Variable,
                q_bins: sc.Variable,
                masking_radius: sc.Variable,
                gravity: bool = False,
                minimizer: str = 'Nelder-Mead',
                tolerance: float = 0.1):
    """
    Find the beam center of a SANS scattering pattern.
    Description of the procedure:

    #. obtain an initial guess by computing the center-of-mass of the pixels,
       weighted by the counts on each pixel
    #. from that initial guess, divide the panel into 4 quadrants
    #. compute :math:`I(Q)` inside each quadrant and compute the residual difference
       between all 4 quadrants
    #. iteratively move the centre position and repeat 2. and 3. until all 4
       :math:`I(Q)` curves lie on top of each other

    Parameters
    ----------
    data:
        The DataArray containing the detector data.
    data_monitors:
        The data arrays for the incident and transmission monitors for the measurement
        run.
    direct_monitors:
        The data arrays for the incident and transmission monitors for the direct
        run.
    direct_beam:
        The direct beam function of the instrument (histogrammed,
        depends on wavelength).
    wavelength_bins:
        The binning in the wavelength dimension to be used.
    q_bins:
        The binning in the Q dimension to be used.
    masking_radius:
        The radius of the circular mask to apply to the data while iterating.
    gravity:
        Include the effects of gravity when computing the scattering angle if ``True``.
    minimizer:
        The Scipy minimizer method to use (see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        for details).
    tolerance:
        Tolerance for termination (see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        for details).
    """  # noqa: E501
    logger = get_logger('sans')
    if gravity and ('gravity' not in data.meta):
        data = data.copy(deep=False)
        data.coords['gravity'] = gravity_vector()
    # Use center of mass to get initial guess for beam center
    xc, yc = _center_of_mass(data)
    logger.info('Initial guess for beam center: '
                f'x={xc.value}[{xc.unit}], y={yc.value}[{yc.unit}]')
    # Refine using Scipy optimize
    from scipy.optimize import minimize
    graph = sans_elastic(gravity=gravity)

    # Compute the denominator used for normalization
    denominator = i_of_q.normalization_denominator(data=data,
                                                   data_monitors=data_monitors,
                                                   direct_monitors=direct_monitors,
                                                   direct_beam=direct_beam,
                                                   wavelength_bins=wavelength_bins)

    wavelength_bands = sc.concat(
        [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength')

    x = data.meta['position'].fields.x
    y = data.meta['position'].fields.y
    res = minimize(_refine,
                   x0=[xc.value, yc.value],
                   args=(data, denominator, graph, q_bins, masking_radius, gravity,
                         wavelength_bands),
                   bounds=[(x.min().value, x.max().value),
                           (y.min().value, y.max().value)],
                   method=minimizer,
                   tol=tolerance)
    logger.info('Final beam center value: '
                f'x={res.x[0]}[{xc.unit}], y={res.x[1]}[{yc.unit}]')
    logger.info(f'Beam center finder minimizer info: {res}')
    return sc.scalar(res.x[0], unit=x.unit), sc.scalar(res.x[1], unit=y.unit)
