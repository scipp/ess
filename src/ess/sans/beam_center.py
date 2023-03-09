# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, List, Tuple

import numpy as np
import scipp as sc

from ..logging import get_logger
from . import i_of_q
from .common import gravity_vector
from .conversions import sans_elastic
from .normalization import normalize, solid_angle_of_rectangular_pixels


def _center_of_mass(data: sc.DataArray) -> Tuple[sc.Variable, sc.Variable]:
    """
    Find the center of mass of the data counts.
    """
    summed = data.sum(list(set(data.dims) - set(data.meta['position'].dims)))
    v = sc.values(summed.data)
    return sc.sum(summed.meta['position'] * v) / v.sum()


def _cost(data: Dict[str, sc.DataArray]) -> float:
    """
    Cost function for determining how close the I(Q) curves are in all four quadrants.
    """
    ref = data['north-east']
    c = ((data['north-west'] - ref)**2 + (data['south-west'] - ref)**2 +
         (data['south-east'] - ref)**2) / ref**2
    return c.sum().value


def _offsets_to_vector(data: sc.DataArray, xy: List[float], graph: dict) -> sc.Variable:
    """
    Convert x,y offsets inside the plane normal to the beam to a vector in absolute
    coordinates.
    """
    u = data.coords['position'].unit
    # Get two vectors that define the plane normal to the beam
    coords = data.transform_coords(['cyl_x_unit_vector', 'cyl_y_unit_vector'],
                                   graph=graph).coords
    center = xy[0] * coords['cyl_x_unit_vector'] + xy[1] * coords['cyl_y_unit_vector']
    center.unit = u
    return center


def _refine(xy: List[float], sample: sc.DataArray, denominator: sc.DataArray,
            graph: dict, q_bins: sc.Variable, masking_radius: sc.Variable,
            gravity: bool, wavelength_bands: sc.Variable) -> float:
    """
    Compute the intensity as a function of Q inside 4 quadrants in Phi.
    Return the sum of the squares of the relative differences between the 4 quadrants.
    """
    data = sample.copy(deep=False)
    data.coords['position'] = sample.coords['position'].copy(deep=True)

    # Offset the position according to the input shift
    center = _offsets_to_vector(data=data, xy=xy, graph=graph)
    data.coords['position'] -= center

    # Add the circular mask
    coords = data.transform_coords(['cylindrical_x', 'cylindrical_y'],
                                   graph=graph).coords
    r = sc.sqrt(coords['cylindrical_x']**2 + coords['cylindrical_y']**2)
    data.masks['circle'] = r > masking_radius
    # Insert a copy of coords and masks needed for conversion to Q
    for c in ['position', 'sample_position', 'source_position']:
        denominator.coords[c] = data.coords[c]
    denominator.masks['circle'] = data.masks['circle']

    pi = sc.constants.pi.value
    phi = data.transform_coords('phi',
                                graph=graph,
                                keep_intermediate=False,
                                keep_inputs=False).coords['phi']
    phi_bins = sc.linspace('phi', -pi, pi, 5, unit='rad')
    quadrants = ['south-west', 'south-east', 'north-east', 'north-west']

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
    cost = _cost(out)
    logger = get_logger('sans')
    logger.info(f'Beam center finder: x={xy[0]}, y={xy[1]}, cost={cost}')
    if not np.isfinite(cost):
        raise ValueError('Non-finite value computed in cost. This is likely due to a '
                         'division by zero. Try increasing the size of your Q bins to '
                         'improve statistics in the denominator.')
    return cost


def beam_center(data: sc.DataArray,
                data_monitors: Dict[str, sc.DataArray],
                direct_monitors: Dict[str, sc.DataArray],
                wavelength_bins: sc.Variable,
                q_bins: sc.Variable,
                masking_radius: sc.Variable,
                gravity: bool = False,
                minimizer: str = 'Nelder-Mead',
                tolerance: float = 0.1) -> Tuple[sc.Variable, sc.Variable]:
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
    if 'gravity' not in data.meta:
        data = data.copy(deep=False)
        data.coords['gravity'] = gravity_vector()
    # Use center of mass to get initial guess for beam center
    com = _center_of_mass(data)
    logger.info(f'Initial guess for beam center: {com}')
    graph = sans_elastic(gravity=gravity)

    # We compute the shift between the incident beam direction and the center-of-mass
    incident_beam = data.transform_coords('incident_beam',
                                          graph=graph).coords['incident_beam']
    n_beam = incident_beam / sc.norm(incident_beam)
    com_shift = com - sc.dot(com, n_beam) * n_beam

    # Compute the denominator used for normalization. The denominator is defined as:
    # pixel_solid_angles * Sample_T_monitor * Direct_I_monitor / Direct_T_monitor
    solid_angle = solid_angle_of_rectangular_pixels(
        data,
        pixel_width=data.coords['pixel_width'],
        pixel_height=data.coords['pixel_height'])

    denominator = (solid_angle * data_monitors['transmission'] *
                   direct_monitors['incident'] / direct_monitors['transmission'])

    # Convert wavelength coordinate to midpoints for future histogramming
    denominator.coords['wavelength'] = sc.midpoints(denominator.coords['wavelength'])

    wavelength_bands = sc.concat(
        [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength')

    coords = data.transform_coords(['cylindrical_x', 'cylindrical_y'],
                                   graph=graph).coords

    # Refine using Scipy optimize
    from scipy.optimize import minimize
    res = minimize(_refine,
                   x0=[com_shift.fields.x.value, com_shift.fields.y.value],
                   args=(data, denominator, graph, q_bins, masking_radius, gravity,
                         wavelength_bands),
                   bounds=[(coords['cylindrical_x'].min().value,
                            coords['cylindrical_x'].max().value),
                           (coords['cylindrical_y'].min().value,
                            coords['cylindrical_y'].max().value)],
                   method=minimizer,
                   tol=tolerance)
    center = _offsets_to_vector(data=data, xy=res.x, graph=graph)
    logger.info(f'Final beam center value: {center}')
    logger.info(f'Beam center finder minimizer info: {res}')
    return center
