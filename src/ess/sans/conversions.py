# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scipp.constants import h, m_n
from scippneutron._utils import elem_unit
from scippneutron.conversion.graph import beamline, tof


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


def cylindrical_x(cyl_x_unit_vector: sc.Variable,
                  scattered_beam: sc.Variable) -> sc.Variable:
    """
    Compute the horizontal x coordinate perpendicular to the incident beam direction.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    return sc.dot(scattered_beam, cyl_x_unit_vector)


def cylindrical_y(cyl_y_unit_vector: sc.Variable,
                  scattered_beam: sc.Variable) -> sc.Variable:
    """
    Compute the vertical y coordinate perpendicular to the incident beam direction.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    return sc.dot(scattered_beam, cyl_y_unit_vector)


def two_theta(incident_beam: sc.Variable, scattered_beam: sc.Variable,
              wavelength: sc.Variable, gravity: sc.Variable) -> sc.Variable:
    """
    Compute the scattering angle from the incident and scattered beam vectors, taking
    into account the effects of gravity.
    Note that it is assumed here that the incident beam is perpendicular to the gravity
    vector.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)

    x_term = cylindrical_x(cyl_x_unit_vector(gravity, incident_beam), scattered_beam)
    x_term *= x_term

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
    y_term *= y_term

    if set(x_term.dims).issubset(set(y_term.dims)):
        y_term += x_term
    else:
        y_term = y_term + x_term
    out = sc.sqrt(y_term, out=y_term)
    out /= L2
    out = sc.asin(out, out=out)
    return out


def phi(cylindrical_x: sc.Variable, cylindrical_y: sc.Variable) -> sc.Variable:
    """
    Compute the cylindrial phi angle around the incident beam. Note that it is assumed
    here that the incident beam is perpendicular to the gravity vector.
    """
    return sc.atan2(y=cylindrical_y, x=cylindrical_x)


def sans_elastic(gravity: bool = False, scatter: bool = True) -> dict:
    """
    Generate a coordinate transformation graph for SANS elastic scattering.
    By default, the effects of gravity on the neutron flight paths are not included.

    :param gravity: Take into account the bending of the neutron flight paths from the
        Earth's gravitational field if ``True``.
    """
    graph = {**beamline.beamline(scatter=scatter), **tof.elastic_Q('tof')}
    if gravity:
        graph['two_theta'] = two_theta
    graph['cyl_x_unit_vector'] = cyl_x_unit_vector
    graph['cyl_y_unit_vector'] = cyl_y_unit_vector
    graph['cylindrical_x'] = cylindrical_x
    graph['cylindrical_y'] = cylindrical_y
    graph['phi'] = phi
    return graph


def sans_monitor() -> dict:
    """
    Generate a coordinate transformation graph for SANS monitor (no scattering).
    """
    return {**beamline.beamline(scatter=False), **tof.elastic_wavelength('tof')}
