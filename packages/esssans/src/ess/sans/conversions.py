# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import scipp as sc
from scipp.constants import h, m_n
from scippneutron._utils import elem_unit
from scippneutron.conversion.graph import beamline, tof

from .common import mask_range
from .types import (
    BeamCenter,
    CalibratedMaskedData,
    CleanQ,
    CleanWavelength,
    CleanWavelengthMasked,
    CorrectForGravity,
    IofQPart,
    MaskedData,
    MonitorType,
    Numerator,
    QxyBins,
    RunType,
    ScatteringRunType,
    TofMonitor,
    WavelengthMask,
    WavelengthMonitor,
)


def cyl_x_unit_vector(gravity: sc.Variable, incident_beam: sc.Variable) -> sc.Variable:
    """
    Compute the horizontal unit vector in the plane normal to the incident beam
    direction. Note that it is assumed here that the incident beam is perpendicular to
    the gravity vector.
    """
    v_x = sc.cross(incident_beam, gravity)
    return v_x / sc.norm(v_x)


def cyl_y_unit_vector(gravity: sc.Variable) -> sc.Variable:
    """
    Compute the vertical unit vector in the plane normal to the incident beam
    direction. Note that it is assumed here that the incident beam is perpendicular to
    the gravity vector.
    """
    v_y = -gravity
    return v_y / sc.norm(v_y)


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


def two_theta(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
) -> dict[str, sc.Variable]:
    """
    Compute the scattering angle from the incident and scattered beam vectors, taking
    into account the effects of gravity.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)

    x_term = cylindrical_x(cyl_x_unit_vector(gravity, incident_beam), scattered_beam)

    y_term = sc.to_unit(wavelength, elem_unit(L2), copy=True)
    y_term *= y_term
    drop = L2**2
    drop *= grav * (m_n**2 / (2 * h**2))
    # Optimization when handling either the dense or the event coord of binned data:
    # - For the event coord, both operands have same dims, and we can multiply in place
    # - For the dense coord, we need to broadcast using non in-place operation
    if set(drop.dims).issubset(set(y_term.dims)):
        y_term *= drop
    else:
        y_term = drop * y_term
    y_term += cylindrical_y(cyl_y_unit_vector(gravity), scattered_beam)
    phi = sc.atan2(y=y_term, x=x_term)

    x_term *= x_term
    y_term *= y_term

    if set(x_term.dims).issubset(set(y_term.dims)):
        y_term += x_term
    else:
        y_term = y_term + x_term
    out = sc.sqrt(y_term, out=y_term)
    out /= L2
    out = sc.asin(out, out=out)
    return {'two_theta': out, 'phi': phi}


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


ElasticCoordTransformGraph = NewType('ElasticCoordTransformGraph', dict)
MonitorCoordTransformGraph = NewType('MonitorCoordTransformGraph', dict)


def sans_elastic(gravity: Optional[CorrectForGravity]) -> ElasticCoordTransformGraph:
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
    gravity:
        Take into account the bending of the neutron flight paths from the
        Earth's gravitational field if ``True``.
    """  # noqa: E501
    graph = {**beamline.beamline(scatter=True), **tof.elastic_Q('tof')}
    if gravity:
        del graph['two_theta']
        graph[('two_theta', 'phi')] = two_theta
    else:
        graph['phi'] = phi_no_gravity
    graph['cyl_x_unit_vector'] = cyl_x_unit_vector
    graph['cyl_y_unit_vector'] = cyl_y_unit_vector
    graph['cylindrical_x'] = cylindrical_x
    graph['cylindrical_y'] = cylindrical_y
    graph[('Qx', 'Qy')] = Qxy
    return ElasticCoordTransformGraph(graph)


def sans_monitor() -> MonitorCoordTransformGraph:
    """
    Generate a coordinate transformation graph for SANS monitor (no scattering).
    """
    return MonitorCoordTransformGraph(
        {**beamline.beamline(scatter=False), **tof.elastic_wavelength('tof')}
    )


def monitor_to_wavelength(
    monitor: TofMonitor[RunType, MonitorType], graph: MonitorCoordTransformGraph
) -> WavelengthMonitor[RunType, MonitorType]:
    return WavelengthMonitor[RunType, MonitorType](
        monitor.transform_coords('wavelength', graph=graph, keep_inputs=False)
    )


def calibrate_positions(
    detector: MaskedData[ScatteringRunType], beam_center: BeamCenter
) -> CalibratedMaskedData[ScatteringRunType]:
    """
    Calibrate pixel positions.

    Currently the only applied calibration is the beam-center offset.
    """
    detector = detector.copy(deep=False)
    detector.coords['position'] = detector.coords['position'] - beam_center
    return detector


# TODO This demonstrates a problem: Transforming to wavelength should be possible
# for RawData, MaskedData, ... no reason to restrict necessarily.
# Would we be fine with just choosing on option, or will this get in the way for users?
def detector_to_wavelength(
    detector: CalibratedMaskedData[ScatteringRunType],
    graph: ElasticCoordTransformGraph,
) -> CleanWavelength[ScatteringRunType, Numerator]:
    return CleanWavelength[ScatteringRunType, Numerator](
        detector.transform_coords('wavelength', graph=graph, keep_inputs=False)
    )


def mask_wavelength(
    da: CleanWavelength[ScatteringRunType, IofQPart], mask: Optional[WavelengthMask]
) -> CleanWavelengthMasked[ScatteringRunType, IofQPart]:
    if mask is not None:
        da = mask_range(da, mask=mask)
    return CleanWavelengthMasked[ScatteringRunType, IofQPart](da)


def compute_Q(
    data: CleanWavelengthMasked[ScatteringRunType, IofQPart],
    graph: ElasticCoordTransformGraph,
    compute_Qxy: Optional[QxyBins],
) -> CleanQ[ScatteringRunType, IofQPart]:
    """
    Convert a data array from wavelength to Q.
    """
    # Keep naming of wavelength dim, subsequent steps use a (Q[xy], wavelength) binning.
    return CleanQ[ScatteringRunType, IofQPart](
        data.transform_coords(
            ('Qx', 'Qy') if compute_Qxy else 'Q',
            graph=graph,
            keep_intermediate=False,
            rename_dims=False,
        )
    )


providers = (
    sans_elastic,
    sans_monitor,
    calibrate_positions,
    monitor_to_wavelength,
    detector_to_wavelength,
    mask_wavelength,
    compute_Q,
)
