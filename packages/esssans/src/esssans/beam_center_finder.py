# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, List, Tuple, Union

import numpy as np
import scipp as sc

from . import i_of_q
from .common import gravity_vector
from .conversions import sans_elastic
from .logging import get_logger
from .normalization import iofq_denominator, normalize


def center_of_mass(data: sc.DataArray) -> sc.Variable:
    """
    Find the center-of-mass of the data counts.
    The center-of-mass is simply the weighted mean of the positions.

    Parameters
    ----------
    data:
        The data to find the center-of-mass of.

    Returns
    -------
    :
        The position of the center-of-mass, as a vector.
    """
    summed = data.sum(list(set(data.dims) - set(data.meta['position'].dims)))
    v = sc.values(summed.data)
    return sc.sum(summed.meta['position'] * v) / v.sum()


def _offsets_to_vector(data: sc.DataArray, xy: List[float], graph: dict) -> sc.Variable:
    """
    Convert x,y offsets inside the plane normal to the beam to a vector in absolute
    coordinates.
    """
    u = data.coords['position'].unit
    # Get two vectors that define the plane normal to the beam
    coords = data.transform_coords(
        ['cyl_x_unit_vector', 'cyl_y_unit_vector'], graph=graph
    ).coords
    center = xy[0] * coords['cyl_x_unit_vector'] + xy[1] * coords['cyl_y_unit_vector']
    center.unit = u
    return center


def iofq_in_quadrants(
    xy: List[float],
    sample: sc.DataArray,
    norm: sc.DataArray,
    graph: dict,
    q_bins: Union[int, sc.Variable],
    gravity: bool,
    wavelength_range: sc.Variable,
) -> Dict[str, sc.DataArray]:
    """
    Compute the intensity as a function of Q inside 4 quadrants in Phi.

    Parameters
    ----------
    xy:
        The x,y offsets in the plane normal to the beam.
    sample:
        The sample data.
    norm:
        The denominator data for normalization.
    graph:
        Coordinate transformation graph.
    q_bins:
        Bin edges for Q.
    gravity:
        If true, the gravity vector is used to compute the scattering angle.
    wavelength_range:
        The wavelength range to use for computing the intensity as a function of Q.

    Returns
    -------
    :
        A dictionary containing the intensity as a function of Q in each quadrant.
        The quadrants are named 'south-west', 'south-east', 'north-east', and
        'north-west'.
    """
    data = sample.copy(deep=False)
    data.coords['position'] = sample.coords['position'].copy(deep=True)

    # Offset the position according to the input shift
    center = _offsets_to_vector(data=data, xy=xy, graph=graph)
    data.coords['position'] -= center

    # Insert a copy of coords needed for conversion to Q
    for c in ['position', 'sample_position', 'source_position']:
        norm.coords[c] = data.coords[c]

    pi = sc.constants.pi.value
    phi = data.transform_coords(
        'phi', graph=graph, keep_intermediate=False, keep_inputs=False
    ).coords['phi']
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
            wavelength_bands=wavelength_range,
        )
        # Denominator counts into Q bins
        norm_q = i_of_q.convert_to_q_and_merge_spectra(
            data=norm[sel],
            graph=graph,
            q_bins=q_bins,
            gravity=gravity,
            wavelength_bands=wavelength_range,
        )
        # Normalize
        out[quad] = normalize(numerator=data_q, denominator=norm_q).hist()
    return out


def cost(xy: List[float], *args) -> float:
    """
    Cost function for determining how close the :math:`I(Q)` curves are in all four
    quadrants. The cost is defined as

    .. math::

       \\text{cost} = \\frac{\\sum_{Q}\\sum_{i=1}^{i=4} \\overline{I}(Q)\\left(I(Q)_{i} - \\overline{I}(Q)\\right)^2}{\\sum_{Q}\\overline{I}(Q)} ~,

    where :math:`i` represents the 4 quadrants and :math:`\\overline{I}(Q)` is the mean
    intensity of the 4 quadrants as a function of :math:`Q`. This is basically a
    weighted mean of the square of the differences between the :math:`I(Q)` curves in
    the 4 quadrants with respect to the mean, and where the weights are
    :math:`\\overline{I}(Q)`.
    We use a weighted mean, as opposed to relative (percentage) differences to give
    less importance to regions with low statistics which are potentially noisy and
    would contribute significantly to the computed cost.

    Parameters
    ----------
    xy:
        The x,y offsets in the plane normal to the beam.
    *args:
        Arguments passed to :func:`iofq_in_quadrants`.

    Returns
    -------
    :
        The sum of the residuals for :math:`I(Q)` in the 4 quadrants, with respect to
        the mean :math:`I(Q)` in all quadrants.

    Notes
    -----
    Mantid uses a different cost function. They compute the horizontal (Left - Right)
    and the vertical (Top - Bottom) costs, and require both to be below the tolerance.
    The costs are defined as

    .. math::

       \\text{cost} = \\sum_{Q} \\left(I(Q)_{\\text{L,T}} - I(Q)_{\\text{R,B}}\\right)^2 ~.

    Using absolute differences instead of a weighted mean is similar to our cost
    function in the way that it would give a lot of weight to even a small difference
    in a high-intensity region. However, it also means that an absolute difference of
    e.g. 2 in a high-intensity region would be weighted the same as a difference of 2
    in a low-intensity region.
    It is also not documented why two separate costs are computed, instead of a single
    one. The Mantid implementation is available
    `here <https://github.com/mantidproject/mantid/blob/main/Framework/PythonInterface/plugins/algorithms/WorkflowAlgorithms/SANS/SANSBeamCentreFinder.py`_.
    """  # noqa: E501
    iofq = iofq_in_quadrants(xy, *args)
    all_q = sc.concat([sc.values(da) for da in iofq.values()], dim='quadrant')
    ref = all_q.mean('quadrant')
    c = (all_q - ref) ** 2
    out = (sc.sum(ref * c) / sc.sum(ref)).value
    logger = get_logger('sans')
    logger.info(f'Beam center finder: x={xy[0]}, y={xy[1]}, cost={out}')
    if not np.isfinite(out):
        raise ValueError(
            'Non-finite value computed in cost. This is likely due to a division by '
            'zero. Try restricting your Q range, or increasing the size of your Q bins '
            'to improve statistics in the denominator.'
        )
    return out


def minimize(
    fun, x0, args=(), bounds=None, method: str = 'Nelder-Mead', tol: float = 0.1
):
    """
    Minimize the supplied cost function using Scipy's optimize.minimize. See the
    `Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
    for more details.

    Parameters
    ----------
    fun:
        The cost function to minimize.
    x0:
        Initial guess.
    args:
        Additional arguments passed to the cost function.
    bounds:
        Bounds on the variables.
    method:
        The minimization method to use.
    tol:
        The tolerance for termination.

    Returns
    -------
    :
        The result of the minimization.
    """  # noqa: E501
    from scipy.optimize import minimize as scipy_minimize

    return scipy_minimize(fun, x0=x0, args=args, bounds=bounds, method=method, tol=tol)


def beam_center(
    data: sc.DataArray,
    data_monitors: Dict[str, sc.DataArray],
    direct_monitors: Dict[str, sc.DataArray],
    wavelength_bins: sc.Variable,
    q_bins: Union[int, sc.Variable],
    gravity: bool = False,
    minimizer: str = 'Nelder-Mead',
    tolerance: float = 0.1,
) -> Tuple[sc.Variable, sc.Variable]:
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

    Returns
    -------
    :
        The beam center position as a vector.

    Notes
    -----
    We record here the thought process we went through during the writing of this
    algorithm. This information is important for understanding why the beam center
    finding is implemented the way it is, and should be considered carefully before
    making changes to the logic of the algorithm.

    **Use a + cut, not an X cut**

    The first idea for implementing the beam center finder was to cut the detector
    panel into 4 wedges using a cross (X) shape. This is what Mantid does, and seemed
    natural, because the offsets when searching for the beam center would be applied
    along the horizontal and vertical directions.
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
    range of scattering (:math:`2{\\theta}`) angles, it spans a range of :math:`Q` bins.

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
    """  # noqa: E501
    logger = get_logger('sans')
    if 'gravity' not in data.meta:
        data = data.copy(deep=False)
        data.coords['gravity'] = gravity_vector()
    # Use center of mass to get initial guess for beam center
    com = center_of_mass(data)
    logger.info(f'Initial guess for beam center: {com}')
    graph = sans_elastic(gravity=gravity)

    # We compute the shift between the incident beam direction and the center-of-mass
    incident_beam = data.transform_coords('incident_beam', graph=graph).coords[
        'incident_beam'
    ]
    n_beam = incident_beam / sc.norm(incident_beam)
    com_shift = com - sc.dot(com, n_beam) * n_beam

    # Compute the denominator used for normalization.
    norm = iofq_denominator(
        data=data,
        data_transmission_monitor=sc.values(data_monitors['transmission']),
        direct_incident_monitor=sc.values(direct_monitors['incident']),
        direct_transmission_monitor=sc.values(direct_monitors['transmission']),
    )

    wavelength_range = sc.concat(
        [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength'
    )

    coords = data.transform_coords(
        ['cylindrical_x', 'cylindrical_y'], graph=graph
    ).coords
    bounds = [
        (coords['cylindrical_x'].min().value, coords['cylindrical_x'].max().value),
        (coords['cylindrical_y'].min().value, coords['cylindrical_y'].max().value),
    ]

    # Refine using Scipy optimize
    res = minimize(
        cost,
        x0=[com_shift.fields.x.value, com_shift.fields.y.value],
        args=(data, norm, graph, q_bins, gravity, wavelength_range),
        bounds=bounds,
        method=minimizer,
        tol=tolerance,
    )

    center = _offsets_to_vector(data=data, xy=res.x, graph=graph)
    logger.info(f'Final beam center value: {center}')
    logger.info(f'Beam center finder minimizer info: {res}')
    return center
