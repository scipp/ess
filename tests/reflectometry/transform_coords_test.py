import scipp as sc
from scipp.constants import g, h, neutron_mass
from ess.reflectometry import transform_coords
import numpy as np


def test_y_dash_for_gravitational_effect():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 0.5, 1], unit=sc.units.m)
    scattered_beam = detector_position - sample_position
    vertical_unit_vector = sc.vector(value=[0, 1, 0])
    forward_beam_unit_vector = sc.vector(value=[0, 0, 1])

    # Approximate cold-neutron velocities
    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass), unit=sc.units.angstrom)

    grad = transform_coords.to_y_dash(wavelength=wav,
                                      scattered_beam=scattered_beam,
                                      vertical_unit_vector=vertical_unit_vector,
                                      forward_beam_unit_vector=forward_beam_unit_vector)

    scattered_beam = detector_position - sample_position
    no_gravity_grad = scattered_beam.fields.y / scattered_beam.fields.z
    gravity_effect_grad = (-0.5 * g * scattered_beam.fields.z / (vel * vel))
    assert sc.isclose(grad, no_gravity_grad + gravity_effect_grad).value


def test_y_dash_with_different_velocities():
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)
    scattered_beam = detector_position - sample_position
    vertical_unit_vector = sc.vector(value=[0, 1, 0])
    fwd_beam_unit_vector = sc.vector(value=[0, 0, 1])

    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass), unit=sc.units.angstrom)

    # In this setup the faster the neutrons the closer d'y(z) tends to 1.0
    transform_args = {
        "wavelength": wav,
        "scattered_beam" : scattered_beam,
        "vertical_unit_vector" : vertical_unit_vector,
        "forward_beam_unit_vector" : fwd_beam_unit_vector
    }
    grad = transform_coords.to_y_dash(**transform_args)
    assert sc.less(grad, 1 * sc.units.one).value

    vel *= 2
    transform_args["wavelength"]\
        = sc.to_unit(h / (vel * neutron_mass), unit=sc.units.angstrom)
    grad_fast = transform_coords.to_y_dash(**transform_args)
    # Testing that gravity has greater influence on slow neutrons.
    assert sc.less(grad, grad_fast).value


def _angle(a, b):
    return sc.acos(sc.dot(a, b) / (sc.norm(a) * sc.norm(b)))


def test_scattering_angle():
    source_position = sc.vector(value=[0, 1, -1], unit=sc.units.m)
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)
    incident_beam = sample_position - source_position
    scattered_beam = detector_position - sample_position
    no_gravity_angle = _angle(scattered_beam, incident_beam)

    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass), unit=sc.units.angstrom)

    angle = transform_coords.to_scattering_angle(w_norm=sc.vector(value=[0, 1, 0]),
                                                 wavelength=wav,
                                                 detector_id=None,
                                                 sample_position=sample_position,
                                                 incident_beam=incident_beam,
                                                 scattered_beam=scattered_beam)
    assert sc.less(angle, no_gravity_angle).value

    gravity_shift_y = -0.5 * g * (scattered_beam.fields.z ** 2 / vel ** 2)
    expected = _angle(scattered_beam + gravity_shift_y
                      * sc.vector(value=[0, 1, 0]), incident_beam) / 2.0
    assert sc.isclose(angle, expected).value


def test_scattering_angle_xzy():
    # Same as previous but we define forward beam direction to be +x
    # up direction to be z (gravity therefore acts in -z)
    # perpendicular direction to be y, as in w is rotation around y

    source_position = sc.vector(value=[-1, 0, 1], unit=sc.units.m)
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    detector_position = sc.vector(value=[1, 0, 1], unit=sc.units.m)
    incident_beam = sample_position - source_position
    scattered_beam = detector_position - sample_position

    vel = 1000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass), unit=sc.units.angstrom)

    angle = transform_coords.to_scattering_angle(w_norm=sc.vector(value=[0, 0, 1]),
                                                 wavelength=wav,
                                                 detector_id=None,
                                                 sample_position=sample_position,
                                                 incident_beam=incident_beam,
                                                 scattered_beam=scattered_beam)

    gravity_shift_y = -0.5 * g * (scattered_beam.fields.z ** 2 / vel ** 2)
    expected = _angle(scattered_beam + gravity_shift_y
                      * sc.vector(value=[0, 1, 0]), incident_beam) / 2.0
    assert sc.isclose(angle, expected).value


def test_det_wavelength_to_wavelength_scattering_angle():
    # comparible with cold-neutrons from moderator
    vel = 2000 * (sc.units.m / sc.units.s)
    wav = sc.to_unit(h / (vel * neutron_mass), unit=sc.units.angstrom)
    sample_position = sc.vector(value=[0, 0, 0], unit=sc.units.m)
    source_position = sc.vector(value=[0, 1, -1], unit=sc.units.m)
    detector_position = sc.vector(value=[0, 1, 1], unit=sc.units.m)

    coords = {}
    coords["sample_position"] = sample_position
    coords["source_position"] = source_position
    coords["position"] = detector_position
    coords["wavelength"] = wav
    coords["w_norm"] = sc.vector(value=[0, 1, 0], unit=sc.units.rad)
    coords["detector_id"] = 0.0 * sc.units.one
    measurement = sc.DataArray(data=1.0 * sc.units.one, coords=coords)

    settings = {"scattering_angle": transform_coords.to_scattering_angle,
                "incident_beam": transform_coords.incident_beam,
                "scattered_beam": transform_coords.scattered_beam}
    transformed = sc.transform_coords(x=measurement,
                                      coords=['wavelength', 'scattering_angle'],
                                      graph=settings)
    assert sc.isclose(transformed.coords['scattering_angle'],
                      (np.pi / 4) * sc.units.rad,
                      atol=1e-4 * sc.units.rad).value
