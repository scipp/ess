import scipp as sc
import numpy as np
from scipp.constants import neutron_mass, h, g


def to_velocity(wavelength):
    return sc.to_unit(h / (wavelength * neutron_mass),
                      sc.units.m / sc.units.s,
                      copy=False)


def to_y_dash(wavelength, scattered_beam, vertical_unit_vector,
              forward_beam_unit_vector):
    velocity_sq = to_velocity(wavelength)
    velocity_sq *= velocity_sq
    g_v = sc.norm(vertical_unit_vector * g)
    # dy due to gravity = -0.5gt^2 = -0.5g(dz/dv)^2
    # therefore y'(z) = dy/dz - 0.5g.dz/dv^2 / dz
    forward = sc.dot(scattered_beam, forward_beam_unit_vector)
    vertical = sc.dot(scattered_beam, vertical_unit_vector)
    return (-0.5 * g_v * forward / velocity_sq) + (vertical / forward)


def incident_beam(sample_position, source_position):
    return sample_position - source_position


def scattered_beam(position, sample_position):
    return position - sample_position


def L1(incident_beam):
    return sc.norm(incident_beam)


def L2(scattered_beam):
    return sc.norm(scattered_beam)


def _angle(a, b):
    return sc.acos(sc.dot(a, b) / (sc.norm(a) * sc.norm(b)))


def to_scattering_angle(w_norm, wavelength, detector_id, sample_position, incident_beam,
                        scattered_beam):
    w_norm = w_norm / sc.norm(w_norm)
    incident_beam_norm = incident_beam / sc.norm(incident_beam)
    scattered_beam_norm = scattered_beam / sc.norm(scattered_beam)
    # vector pointing along surface in forward beam direction
    surface = sc.cross(w_norm, sc.cross(incident_beam_norm, scattered_beam_norm))
    # Assume specular reflection. And reflect incident beam through surface
    reflection = incident_beam - 2.0 * sc.dot(incident_beam, surface) * surface
    forward_beam_direction = incident_beam - reflection
    # For a specular reflection, this would be basis aligned
    forward_beam_direction /= sc.norm(forward_beam_direction)
    # Account for non-specular scattering
    forward_beam_direction = sc.vector(value=np.round(forward_beam_direction.value),
                                       unit=forward_beam_direction.unit)
    # Vertical direction
    vertical_direction = sc.vector(value=np.round(w_norm.value), unit=w_norm.unit)
    y_dash = to_y_dash(wavelength, scattered_beam, vertical_direction,
                       forward_beam_direction)
    start = sc.dot(vertical_direction, sample_position)
    height = y_dash * sc.dot(forward_beam_direction, scattered_beam) + start

    w = _angle(w_norm, surface) - sc.scalar(value=np.pi / 2, unit=sc.units.rad)
    return sc.atan2(y=height, x=sc.dot(forward_beam_direction, incident_beam)) - w
