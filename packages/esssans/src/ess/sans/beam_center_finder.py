# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from dataclasses import dataclass

import numpy as np
import sciline
import scipp as sc
import scippnexus as snx
from scipp.core import concepts

from ess.reduce.uncertainty import UncertaintyBroadcastMode

from .conversions import ElasticCoordTransformGraph
from .logging import get_logger
from .types import (
    BeamCenter,
    CleanDirectBeam,
    CorrectedDetector,
    DetectorBankSizes,
    DimsToKeep,
    IntensityQ,
    NeXusComponent,
    Numerator,
    QBins,
    ReturnEvents,
    SampleRun,
    WavelengthBands,
    WavelengthMask,
)


def _xy_extrema(pos: sc.Variable) -> sc.Variable:
    x_min = pos.fields.x.min()
    x_max = pos.fields.x.max()
    y_min = pos.fields.y.min()
    y_max = pos.fields.y.max()
    return sc.concat([x_min, x_max, y_min, y_max], dim='extremes')


def _find_beam_center(
    data,
    beam_stop_radius,
    beam_stop_arm_width,
):
    '''
    Each iteration the center of mass of the remaining intensity is computed
    and assigned to be the current beam center guess.
    Then three symmetrical masks are created to make sure that the remaining intensity
    distribution does not extend outside of the detector and that the beam stop
    does not make the remaining intensity asymmetrical.

    The three masks are:

      - one "outer" circular mask with radius less than the minimal distance
        from the current beam center guess to the border of the detector
      - one "inner" circular mask with radius larger than the beam stop
      - one "arm" rectangular mask with width wider than the beam stop arm

    The "outer" mask radius is found from the detector size.
    The "inner" mask radius is supplied by the caller.
    The "arm" mask slope is determined by the direction of minimum intensity
    around the current beam center guess, the "arm" mask width is an argument
    supplied by the caller.
    '''
    events = data.copy()
    events.masks.clear()
    intensity = events.bins.sum()

    for i in range(20):
        # Average position, weighted by intensity.
        center = (
            intensity.coords['position'] * sc.values(intensity)
        ).sum() / sc.values(intensity).sum()
        distance_to_center = intensity.coords['position'] - center.data

        # Outer radius of annulus around center.
        # Defined so that the annulus does not go outside of the bounds of the detector.
        outer = 0.9 * min(
            sc.abs(distance_to_center.fields.x.min()),
            sc.abs(distance_to_center.fields.x.max()),
            sc.abs(distance_to_center.fields.y.min()),
            sc.abs(distance_to_center.fields.y.max()),
        )
        intensity.masks['outer'] = (
            distance_to_center.fields.x**2 + distance_to_center.fields.y**2 > outer**2
        )
        # Inner radius defined by size of beam stop.
        intensity.masks['inner'] = (
            distance_to_center.fields.x**2 + distance_to_center.fields.y**2
            < beam_stop_radius**2
        )

        # Iterate without the arm mask a few times to settle near the center
        # before introducing the arm mask.
        if i > 10:
            # Angle between +x and distance_to_center.
            intensity.coords['theta'] = sc.where(
                distance_to_center.fields.x > sc.scalar(0.0, unit='m'),
                sc.atan2(y=distance_to_center.fields.y, x=distance_to_center.fields.x),
                sc.scalar(sc.constants.pi.value, unit='rad')
                - sc.atan2(
                    y=distance_to_center.fields.y, x=-distance_to_center.fields.x
                ),
            )
            intensity_over_angle = intensity.drop_masks(
                ['arm'] if 'arm' in intensity.masks else []
            ).hist(theta=100)
            # Assume angle of arm coincides with intensity minimum
            arm_angle = intensity_over_angle.coords['theta'][
                np.argmin(intensity_over_angle.values)
            ]

            slope = sc.tan(arm_angle)
            intensity.masks['arm'] = (
                distance_to_center.fields.y
                < slope * distance_to_center.fields.x + beam_stop_arm_width
            ) & (
                distance_to_center.fields.y
                > slope * distance_to_center.fields.x - beam_stop_arm_width
            )

    return center.data


def beam_center_from_center_of_mass_alternative(
    workflow,
    beam_stop_radius=None,
    beam_stop_arm_width=None,
) -> BeamCenter:
    """
    Estimate the beam center via the center-of-mass of the data counts.

    We are assuming the intensity distribution is symmetric around the beam center.
    Even if the intensity distribution is symmetric around the beam center
    the intensity distribution in the detector might not be, because

        - the detector has a finite extent,
        - and there is a beam stop covering part of the detector.

    To deal with the limited size of the detector a mask can be applied that is small
    enough so that the the remaining intensity is entirely inside the detector.
    To deal with the beam stop we can mask the region of the detector that the
    beam stop covers.

    But to preserve the symmetry of the intensity around the beam center the masks
    also need to be symmetical around the beam center.
    The problem is, the beam center is unknown.
    However, if the beam center was known to us, and we applied symmetrical masks
    that covered the regions of the detector where the intensity distribution is
    asymmetrical,
    then the center of mass of the remaining intensity would equal the beam center.
    Conversely, if we apply symmetrical masks around a point that is not the beam center
    the center of mass of the remaining intensity will (likely) not equal the original
    point.
    This suggests the beam center can be found using a fixed point iteration where each
    iteration we

    1. Compute the center of mass of the remaining intensity and assign it to be our
       current estimate of the beam center.
    2. Create symmetrical masks around the current estimate of the beam center.
    3. Repeat from 1. until convergence.

    Parameters
    ----------
    workflow:
        The reduction workflow to compute CorrectedDetector[SampleRun, Numerator].

    Returns
    -------
    :
        The beam center position as a vector.
    """

    if beam_stop_radius is None:
        beam_stop_radius = sc.scalar(0.05, unit='m')
    if beam_stop_arm_width is None:
        beam_stop_arm_width = sc.scalar(0.02, unit='m')

    workflow = _with_default_beam_center(workflow)
    data = workflow.compute(CorrectedDetector[SampleRun, Numerator])
    graph = workflow.compute(ElasticCoordTransformGraph[SampleRun])
    com = _find_beam_center(data, beam_stop_radius, beam_stop_arm_width)
    return BeamCenter(com - _sample_position(data, graph))


def _sample_position(data: sc.DataArray, graph: dict) -> sc.Variable:
    """Sample position as defined by the coordinate transformation graph."""
    return data.transform_coords('sample_position', graph=graph).coords[
        'sample_position'
    ]


def _with_default_beam_center(workflow: sciline.Pipeline) -> sciline.Pipeline:
    """
    Return a workflow that can compute detector data even without a beam center set.

    Computing the detector data may require a beam center, but the beam center found by
    the functions below does not depend on it: it is derived from the pixel positions,
    which the beam center does not modify.
    """
    try:
        workflow.compute(BeamCenter)
    except sciline.UnsatisfiedRequirement:
        workflow = workflow.copy()
        workflow[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    return workflow


def beam_center_from_center_of_mass(workflow: sciline.Pipeline) -> BeamCenter:
    """
    Estimate the beam center via the center-of-mass of the data counts.

    The center-of-mass is simply the weighted mean of the positions.
    Areas with low counts are excluded from the center of mass calculation, as they
    typically fall into asymmetric regions of the detector panel and would thus lead
    to a biased result.

    The result is the center-of-mass relative to the sample, i.e., the transverse offset
    of the beam together with the distance from the sample at which that offset was
    determined. See :py:class:`ess.sans.types.BeamCenter`.

    On a bank with depth, the reported distance is the intensity-weighted mean depth,
    not the geometric center of the bank: layers closer to the sample see more counts
    and pull it forward. That is consistent rather than wrong. The transverse position
    of the beam is linear in the distance from the sample, so any intensity-weighted
    average of points on the beam is again a point on the beam. The weighting moves the
    result along the beam, not away from it, and the reported distance says where it
    ended up.

    This holds as long as the intensity is symmetric around the beam at every depth. It
    is not exactly true: masking, the shadow of the beam stop and, with gravity, the
    wavelength-dependent drop all bias the center-of-mass, and they do so differently at
    different depths. The resulting uncertainty on the distance is a few centimetres for
    a Loki bank, which is of the order of the depth of the bank itself. The distance is
    used to extrapolate the transverse offset to other banks, where an error of a few
    centimetres out of several metres is negligible, but it means the variation of the
    correction *within* one bank is not resolved any better than the noise on it.

    The result is anchored at the sample position, so an error in the sample position at
    the time of determination tilts the inferred beam direction by that error divided by
    the distance. One millimetre of transverse error at five metres is 0.2 mrad, which
    is 0.24 mm of transverse error on a bank one metre from the sample. Errors along the
    beam are second order and negligible. Note that this is an error on the beam
    direction and does not require the sample to be in the same place for the run the
    beam center is applied to: the direction is re-anchored at that run's own sample
    position.

    Parameters
    ----------
    workflow:
        The reduction workflow to compute CorrectedDetector[SampleRun, Numerator].

    Returns
    -------
    :
        The beam center position as a vector.
    """
    workflow = _with_default_beam_center(workflow)
    data = workflow.compute(CorrectedDetector[SampleRun, Numerator])
    graph = workflow.compute(ElasticCoordTransformGraph[SampleRun])

    dims_to_sum = set(data.dims) - set(data.coords['position'].dims)
    if dims_to_sum:
        summed = data.sum(dims_to_sum)
    else:
        summed = data.bins.sum()
    if summed.ndim > 1:
        summed = summed.flatten(to=uuid.uuid4().hex)

    pos = summed.coords['position']
    v = sc.values(summed)
    mask = concepts.irreducible_mask(summed, dim=None)
    if mask is None:
        mask = sc.zeros(sizes=pos.sizes, dtype='bool')
    extrema = _xy_extrema(pos[~mask])
    # Mean including existing masks
    cutoff = 0.1 * v.mean().data
    low_counts = v.data < cutoff
    # Increase cutoff until we no longer include pixels at the X/Y min/max.
    # This would be simpler if the logical panel shape was reflected in the
    # dims of the input data, instead of having a flat list of pixels.
    while sc.any(_xy_extrema(pos[~(mask | low_counts)]) == extrema):
        cutoff *= 2.0
        low_counts = v.data < cutoff
    # See scipp/scipp#3271, the following lines are a workaround
    select = ~(low_counts | mask)
    v = v.data[select]
    pos = pos[select]
    com = sc.sum(pos * v) / v.sum()

    # The center-of-mass is a point on the beam, so relative to the sample it is the
    # beam center, including the distance at which it was determined.
    return BeamCenter(com - _sample_position(summed, graph))


@dataclass(frozen=True)
class _BeamPlane:
    """
    Plane normal to the incident beam in which beam-center offsets are expressed.

    ``axis`` points from the sample to the plane, i.e., it carries the distance at which
    the beam center is determined. Offsets alone do not define a beam direction, so
    searching for a beam center means searching within such a plane.
    """

    unit_x: sc.Variable
    unit_y: sc.Variable
    axis: sc.Variable

    @staticmethod
    def from_beam_center(
        data: sc.DataArray, graph: dict, beam_center: sc.Variable
    ) -> '_BeamPlane':
        """Plane containing ``beam_center``, normal to the incident beam."""
        coords = data.transform_coords(
            ['cyl_x_unit_vector', 'cyl_y_unit_vector', 'incident_beam'], graph=graph
        ).coords
        incident_beam = coords['incident_beam']
        direction = incident_beam / sc.norm(incident_beam)
        return _BeamPlane(
            unit_x=coords['cyl_x_unit_vector'],
            unit_y=coords['cyl_y_unit_vector'],
            axis=sc.dot(beam_center, direction) * direction,
        )

    def offsets(self, beam_center: sc.Variable) -> list[float]:
        """Components of ``beam_center`` within the plane."""
        return [sc.dot(beam_center, e).value for e in (self.unit_x, self.unit_y)]

    def beam_center(self, xy: list[float]) -> sc.Variable:
        """Beam center given by offsets within the plane."""
        center = xy[0] * self.unit_x + xy[1] * self.unit_y
        center.unit = self.axis.unit
        return center + self.axis


def _iofq_in_quadrants(
    xy: list[float],
    workflow: sciline.Pipeline,
    detector: sc.DataArray,
    norm: sc.DataArray,
    plane: _BeamPlane,
) -> dict[str, sc.DataArray]:
    """
    Compute the intensity as a function of Q inside 4 quadrants in Phi.

    Parameters
    ----------
    xy:
        The x,y offsets in the plane normal to the beam.
    detector:
        The raw detector.
    norm:
        The denominator data for normalization.
    plane:
        The plane normal to the beam in which ``xy`` are given.

    Returns
    -------
    :
        A dictionary containing the intensity as a function of Q in each quadrant.
        The quadrants are named 'south-west', 'south-east', 'north-east', and
        'north-west'.
    """
    pi = sc.constants.pi.value
    phi_bins = sc.linspace('phi', -pi, pi, 5, unit='rad')
    quadrants = ['south-west', 'south-east', 'north-east', 'north-west']

    workflow = workflow.copy()
    workflow[BeamCenter] = plane.beam_center(xy)
    graph = workflow.compute(ElasticCoordTransformGraph[SampleRun])
    calibrated = workflow.compute(CorrectedDetector[SampleRun, Numerator])
    with_phi = calibrated.transform_coords(
        'phi', graph=graph, keep_intermediate=False, keep_inputs=False
    )
    # If gravity-correction is enabled, phi depends on wavelength (and event).
    # We cannot handle this below, so we approximate phi by the mean value.
    if ('phi' not in with_phi.coords) and ('phi' in with_phi.bins.coords):
        # This is the case where we have a phi event coord but no coord at the top level
        phi = with_phi.bins.coords['phi'].bins.mean()
    else:
        phi = with_phi.coords['phi']
        if phi.bins is not None or 'wavelength' in phi.dims:
            phi = phi.mean('wavelength')

    out = {}
    for i, quad in enumerate(quadrants):
        # Select pixels based on phi
        sel = (phi >= phi_bins[i]) & (phi < phi_bins[i + 1])
        # Restrict the raw detector to the quadrant, so the denominator (solid angle)
        # covers the same pixels as the numerator.
        workflow[NeXusComponent[snx.NXdetector, SampleRun]] = sc.DataGroup(
            data=detector[sel]
        )
        # MaskedData would be computed automatically, but we did it above already
        workflow[CorrectedDetector[SampleRun, Numerator]] = calibrated[sel]
        workflow[CleanDirectBeam] = norm if norm.dims == ('wavelength',) else norm[sel]
        out[quad] = workflow.compute(IntensityQ[SampleRun])
    return out


def _cost(xy: list[float], *args) -> float:
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
    iofq = _iofq_in_quadrants(xy, *args)
    all_q = sc.concat([sc.values(da) for da in iofq.values()], dim='quadrant')
    ref = all_q.mean('quadrant')
    c = (all_q - ref) ** 2
    out = (sc.sum(ref * c) / sc.sum(ref)).value
    logger = get_logger('sans')
    if not np.isfinite(out):
        out = np.inf
        logger.info(
            'Non-finite value computed in cost. This is likely due to a division by '
            'zero. If the final results for the beam center are not satisfactory, '
            'try restricting your Q range, or increasing the size of your Q bins to '
            'improve statistics in the denominator.'
        )
    logger.info('Beam center finder: x=%s, y=%s, cost=%s', xy[0], xy[1], out)
    return out


def beam_center_from_iofq(
    *,
    workflow: sciline.Pipeline,
    q_bins: int | sc.Variable,
    minimizer: str | None = None,
    tolerance: float | None = None,
) -> BeamCenter:
    """
    Find the beam center of a SANS scattering pattern using an I(Q) calculation.

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
    workflow:
        The reduction workflow to compute I(Q).
    q_bins:
        The binning in the Q dimension to be used.
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
    the detector where the beam stop is casting a shadow.
    Such a beam stop will not appear in all 4 quadrants, and because it spans a
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
    changes it in the same manner for all :math:`{\\phi}` angles, this does not affect
    the results for finding the beam center.

    This is what is now implemented in this version of the algorithm.
    """
    from scipy.optimize import minimize

    logger = get_logger('sans')

    logger.info('Requested minimizer: %s', minimizer)
    logger.info('Requested tolerance: %s', tolerance)
    minimizer = minimizer or 'Nelder-Mead'
    tolerance = tolerance or 0.1
    logger.info('Using minimizer: %s', minimizer)
    logger.info('Using tolerance: %s', tolerance)

    keys = (
        NeXusComponent[snx.NXdetector, SampleRun],
        CorrectedDetector[SampleRun, Numerator],
        CleanDirectBeam,
        ElasticCoordTransformGraph[SampleRun],
    )
    workflow = workflow.copy()
    # Avoid reshape of detector, which would break boolean-indexing by cost function
    workflow[DetectorBankSizes] = {}
    results = workflow.compute(keys)
    detector = results[NeXusComponent[snx.NXdetector, SampleRun]]['data']
    data = results[CorrectedDetector[SampleRun, Numerator]]
    norm = results[CleanDirectBeam]
    graph = results[ElasticCoordTransformGraph[SampleRun]]

    # Avoid reloading the detector
    workflow[NeXusComponent[snx.NXdetector, SampleRun]] = sc.DataGroup(data=detector)
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    workflow[ReturnEvents] = False
    workflow[DimsToKeep] = ()
    workflow[WavelengthMask] = None
    workflow[WavelengthBands] = None
    workflow[QBins] = q_bins

    # Use center of mass to get initial guess for beam center
    com = beam_center_from_center_of_mass(workflow)
    logger.info('Initial guess for beam center: %s', com)

    # The refinement below only varies the offsets within the plane normal to the beam.
    # The distance at which the beam center is determined is taken from the initial
    # guess and kept fixed.
    plane = _BeamPlane.from_beam_center(data=data, graph=graph, beam_center=com)

    coords = data.transform_coords(
        ['cylindrical_x', 'cylindrical_y'], graph=graph
    ).coords
    bounds = [
        (coords['cylindrical_x'].min().value, coords['cylindrical_x'].max().value),
        (coords['cylindrical_y'].min().value, coords['cylindrical_y'].max().value),
    ]

    # Refine using Scipy optimize
    res = minimize(
        _cost,
        x0=plane.offsets(com),
        args=(workflow, detector, norm, plane),
        bounds=bounds,
        method=minimizer,
        tol=tolerance,
    )

    center = plane.beam_center(res.x)
    logger.info('Final beam center value: %s', center)
    logger.info('Beam center finder minimizer info: %s', res)
    return center
