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
    all_q = sc.concat(list(data.values()), dim='quadrant')
    ref = sc.values(all_q.mean('quadrant'))
    c = sc.abs(all_q - ref) / ref
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

    Notes
    -----
    We record here the thought process we went through during the writing of this
    algorithm. This information is important for understanding why the beam center
    finding is implemented the way it is, and should be considered carefully before
    making changes to the logic of the algorithm.

    **Use a + cut, not an X cut**

    The first idea for implementing the beam center finder was to cut the detector
    panel into 4 wedges using a cross (X) shape. This seemed natural, because the
    offsets when searching for the beam center would be applied along the horizontal
    and vertical directions.
    This worked well on square detector panels (like the SANS2D detector), but on
    rectangular detectors, the north and south wedges ended up holding many less pixels
    than the east and west panels.
    More pixels means more contributions to a particular :math:`Q` bin, and comparing
    the :math:`I(Q)` curves in the 4 wedges was thus not possible.
    We therefore divided the detector panel into 4 quadrants using a ``+`` cut instead.
    Note that since we are looking at an isotropic scattering pattern, the shape of the
    cut (and the number of quadrants) should not matter for the resulting shapes of the
    :math:`I(Q)` curves.

    **Normalization inside the 4 quadrants**

    The first attempt at implementing the beam center finder was to only compute the
    raw counts as a function of $Q$ for the sample run, and not compute any
    normalization term.
    The idea was that even though this would change the shape of the :math:`I(Q)` curve,
    because we were looking at isotropic scattering, it would change the shape of the
    curve isotropically, thus still allowing us to find the center when the curves in
    all 4 quadrants overlap.
    The motivation for this was to save computational cost.

    After discovering the issue that using a ``X`` shaped cut for dividing the detector
    panel would yield different contributions to :math:`I(Q)` in the different wedges,
    we concluded that some normalization was necessary.
    The first version was to simply sum the counts in each quadrant and use this to
    normalize the counts for each intensity curve.

    This was, however, not sufficient in cases where masks are applied to the detector
    pixels. It is indeed very common to mask broken pixels, as well as the region of
    the detector where the sample holder is casting a shadow.
    Such a sample holder will not appear in all 4 quadrants, and because it spans a
    range of scattering (:math:`2{\theta}`) angles, it spans a range of :math:`Q` bins.

    All this means that we in fact need to perform a reduction as close as possible to
    the full :math:`I(Q)` reduction in each of the 4 quadrants to achieve a reliable
    result.
    We write 'as close as possible' because In the full :math:`I(Q)` reduction, there
    is a term :math:`D({\\lambda})` in the normalization called the 'direct beam' which
    gives the efficiency of the detectors as a function of wavelength.
    Because finding the beam center is required to compute the direct beam in the first
    place, we do not include this term in the computation of :math:`I(Q)` for finding
    the beam center. This changes the shape of the :math:`I(Q)` curve, but since it
    changes it in the same manner for all :math:`{\\phi}` angles, this does not affect the
    results for finding the beam center.

    This is what is now implemented in this version of the algorithm.

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
        While iterating to find the beam center, the current center will not be in the
        center of the detector panel. This can introduce bias in the shape of the
        :math:`I(Q)` inside the 4 quadrants. To avoid this, we apply a circular mask
        around the current center, to ensure all directions contribute equally to
        :math:`Q` bins.
    gravity:
        Include the effects of gravity when computing the scattering angle if ``True``.
    minimizer:
        The Scipy minimizer method to use (see the
        `Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
        for details).
    tolerance:
        Tolerance for termination (see the
        `Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
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
