# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable

import sciline
import scipp as sc
import scippnexus as snx
from scippneutron.conversion.beamline import (
    beam_aligned_unit_vectors,
    scattering_angles_with_gravity,
)
from scippneutron.conversion.graph import beamline, tof

from ess.reduce.uncertainty import broadcast_uncertainties

from .common import mask_range
from .types import (
    BeamCenter,
    BinnedQ,
    BinnedQxQy,
    CorrectForGravity,
    Denominator,
    DetectorTerm,
    GravityVector,
    IofQPart,
    MonitorTerm,
    NormalizedQ,
    NormalizedQxQy,
    Numerator,
    Position,
    QDetector,
    QxyDetector,
    RunType,
    UncertaintyBroadcastMode,
    WavelengthMask,
)


def _incident_beam_from_beam_center(
    beam_center: sc.Variable, nominal_incident_beam: sc.Variable
) -> Callable[[], sc.Variable]:
    """
    Return a provider for ``incident_beam`` that points along the actual beam.

    Scattering angles must be measured relative to the actual beam. The beam passes
    through the sample, but its direction, set by the collimation, in general deviates
    from the nominal beam axis defined by source and sample. The beam center is a point
    on the actual beam, measured from the sample, so it defines that direction.

    Only the direction is taken from the beam center. The length is that of the nominal
    incident beam, since ``L1`` is a property of the beamline and not of the beam
    center. Scaling ``beam_center`` therefore leaves the result unchanged.
    """
    axis = nominal_incident_beam / sc.norm(nominal_incident_beam)
    if sc.dot(beam_center, axis).value <= 0.0:
        raise ValueError(
            f'Invalid beam center {beam_center}. The beam center is the position of '
            'the beam relative to the sample, so its component along the beam must be '
            'the (positive) distance from the sample to the plane in which the beam '
            'center was determined. A beam center given as a transverse offset alone, '
            'without that distance, cannot define a beam direction.'
        )
    incident_beam = sc.norm(nominal_incident_beam) * beam_center / sc.norm(beam_center)
    return lambda: incident_beam


def cyl_unit_vectors(incident_beam: sc.Variable, gravity: sc.Variable):
    vectors = beam_aligned_unit_vectors(incident_beam=incident_beam, gravity=gravity)
    return {
        'cyl_x_unit_vector': vectors['beam_aligned_unit_x'],
        'cyl_y_unit_vector': vectors['beam_aligned_unit_y'],
    }


def cylindrical_x(
    cyl_x_unit_vector: sc.Variable, scattered_beam: sc.Variable
) -> sc.Variable:
    """
    Compute the horizontal x coordinate perpendicular to the incident beam direction.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    return sc.dot(scattered_beam, cyl_x_unit_vector)


def cylindrical_y(
    cyl_y_unit_vector: sc.Variable, scattered_beam: sc.Variable
) -> sc.Variable:
    """
    Compute the vertical y coordinate perpendicular to the incident beam direction.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    return sc.dot(scattered_beam, cyl_y_unit_vector)


def phi_no_gravity(
    cylindrical_x: sc.Variable, cylindrical_y: sc.Variable
) -> sc.Variable:
    """
    Compute the cylindrical phi angle around the incident beam. Note that it is assumed
    here that the incident beam is perpendicular to the gravity vector.
    """
    return sc.atan2(y=cylindrical_y, x=cylindrical_x)


def Qxy(Q: sc.Variable, phi: sc.Variable) -> dict[str, sc.Variable]:
    """
    Compute the Qx and Qy components of the scattering vector from the scattering angle,
    wavelength, and phi angle.
    """
    Qx = sc.cos(phi)
    Qy = sc.sin(phi)
    if Q.bins is not None and phi.bins is not None:
        Qx *= Q
        Qy *= Q
    else:
        Qx = Qx * Q
        Qy = Qy * Q
    return {'Qx': Qx, 'Qy': Qy}


class ElasticCoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """
    Coordinate transformation graph for SANS elastic scattering (which possibly
    includes the effects of gravitationaly pull on the neutrons).
    See :func:`sans_elastic` for more details.
    """


def sans_elastic(
    correct_for_gravity: CorrectForGravity,
    *,
    sample_position: Position[snx.NXsample, RunType],
    source_position: Position[snx.NXsource, RunType],
    beam_center: BeamCenter,
    gravity: GravityVector,
) -> ElasticCoordTransformGraph[RunType]:
    """
    Generate a coordinate transformation graph for SANS elastic scattering.

    It is based on classical conversions from ``tof`` and pixel ``position`` to
    :math:`\\lambda` (``wavelength``), :math:`\\theta` (``theta``) and
    :math:`Q` (``Q``), but can take into account the Earth's gravitational field,
    which bends the flight path of the neutrons, to compute the scattering angle
    :math:`\\theta`.

    The angle can be found using the following expression
    (`Seeger & Hjelm 1991 <https://doi.org/10.1107/S0021889891004764>`_):

    .. math::

       \\theta = \\frac{1}{2}\\sin^{-1}\\left(\\frac{\\sqrt{ x^{2} + \\left( y + \\frac{g m_{\\rm n}}{2 h^{2}} \\lambda^{2} L_{2}^{2} \\right)^{2} } }{L_{2}}\\right)

    where :math:`x` and :math:`y` are the spatial coordinates of the pixels in the
    horizontal and vertical directions, respectively,
    :math:`m_{\\rm n}` is the neutron mass,
    :math:`L_{2}` is the distance between the sample and a detector pixel,
    :math:`g` is the acceleration due to gravity,
    and :math:`h` is Planck's constant.

    By default, the effects of gravity on the neutron flight paths are not included
    (equivalent to :math:`g = 0` in the expression above).

    Parameters
    ----------
    correct_for_gravity:
        Take into account the bending of the neutron flight paths from the
        Earth's gravitational field if ``True``.
    gravity:
        A vector indicating the strength and direction of gravity.
        Required even if ``correct_for_gravity`` is ``False``.
    sample_position:
        Position of the sample as a vector.
    source_position:
        Position of the source as a vector.
    beam_center:
        Position of the beam center relative to the sample, i.e., the transverse offset
        of the beam together with the distance from the sample at which that offset was
        determined. Set to a zero vector to apply no correction.
    """  # noqa: E501
    graph = {
        **beamline.beamline(scatter=True),
        **tof.elastic_Q('wavelength'),
        'sample_position': lambda: sample_position,
        'source_position': lambda: source_position,
        'gravity': lambda: gravity,
    }
    # A zero beam center means no correction, leaving the plain beamline graph.
    if sc.norm(beam_center).value != 0.0:
        graph['incident_beam'] = _incident_beam_from_beam_center(
            beam_center, nominal_incident_beam=sample_position - source_position
        )
    if correct_for_gravity:
        del graph['two_theta']
        graph[('two_theta', 'phi')] = scattering_angles_with_gravity
    else:
        graph['phi'] = phi_no_gravity
    graph[('cyl_x_unit_vector', 'cyl_y_unit_vector')] = cyl_unit_vectors
    graph['cylindrical_x'] = cylindrical_x
    graph['cylindrical_y'] = cylindrical_y
    graph[('Qx', 'Qy')] = Qxy
    return ElasticCoordTransformGraph[RunType](graph)


def mask_wavelength_q(
    da: BinnedQ[RunType, Numerator], mask: WavelengthMask
) -> NormalizedQ[RunType, Numerator]:
    if mask is not None:
        da = mask_range(da, mask=mask)
    return NormalizedQ[RunType, Numerator](da)


def mask_wavelength_qxy(
    da: BinnedQxQy[RunType, Numerator], mask: WavelengthMask
) -> NormalizedQxQy[RunType, Numerator]:
    if mask is not None:
        da = mask_range(da, mask=mask)
    return NormalizedQxQy[RunType, Numerator](da)


def mask_and_scale_wavelength_q(
    da: BinnedQ[RunType, Denominator],
    mask: WavelengthMask,
    wavelength_term: MonitorTerm[RunType],
    uncertainties: UncertaintyBroadcastMode,
) -> NormalizedQ[RunType, Denominator]:
    da = da * broadcast_uncertainties(wavelength_term, prototype=da, mode=uncertainties)
    if mask is not None:
        da = mask_range(da, mask=mask)
    return NormalizedQ[RunType, Denominator](da)


def mask_and_scale_wavelength_qxy(
    da: BinnedQxQy[RunType, Denominator],
    mask: WavelengthMask,
    wavelength_term: MonitorTerm[RunType],
    uncertainties: UncertaintyBroadcastMode,
) -> NormalizedQxQy[RunType, Denominator]:
    da = da * broadcast_uncertainties(wavelength_term, prototype=da, mode=uncertainties)
    if mask is not None:
        da = mask_range(da, mask=mask)
    return NormalizedQxQy[RunType, Denominator](da)


def _compute_Q(
    data: sc.DataArray, graph: ElasticCoordTransformGraph, target: tuple[str, ...]
) -> sc.DataArray:
    # Keep naming of wavelength dim, subsequent steps use a (Q[xy], wavelength) binning.
    return QDetector[RunType, IofQPart](
        data.transform_coords(
            target,
            graph=graph,
            keep_intermediate=False,
            rename_dims=False,
        )
    )


def compute_Q(
    data: DetectorTerm[RunType, IofQPart],
    graph: ElasticCoordTransformGraph[RunType],
) -> QDetector[RunType, IofQPart]:
    """
    Convert a data array from wavelength to Q.
    """
    return QDetector[RunType, IofQPart](
        _compute_Q(data=data, graph=graph, target=('Q',))
    )


def compute_Qxy(
    data: DetectorTerm[RunType, IofQPart],
    graph: ElasticCoordTransformGraph[RunType],
) -> QxyDetector[RunType, IofQPart]:
    """
    Convert a data array from wavelength to Qx and Qy.
    """
    return QxyDetector[RunType, IofQPart](
        _compute_Q(data=data, graph=graph, target=('Qx', 'Qy'))
    )


providers = (
    sans_elastic,
    mask_wavelength_q,
    mask_wavelength_qxy,
    mask_and_scale_wavelength_q,
    mask_and_scale_wavelength_qxy,
    compute_Q,
    compute_Qxy,
)
