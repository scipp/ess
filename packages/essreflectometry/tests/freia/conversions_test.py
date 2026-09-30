# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scipp.testing import assert_allclose

from ess.freia.conversions import (
    hypothetical_incident_direction,
    incident_angle,
    incident_beam_directions,
    incident_direction,
    outgoing_direction,
    scattering_angle,
    theta,
)


def test_scattering_and_reflection_angles():
    normal = sc.vector([0.0, np.sqrt(3) / 2, -0.5])
    # The first ray is 15 degrees above the tilted surface. The second lies in
    # the surface but outside the vertical scattering plane.
    direction = outgoing_direction(
        scattered_beam=sc.vectors(
            dims=['event'], values=[[0, 1, 1], [1, 0.5, np.sqrt(3) / 2]], unit='m'
        ),
        wavelength=sc.scalar(4.0, unit='angstrom'),
        gravity=sc.vector([0.0, 0.0, 0.0], unit='m/s^2'),
    )

    assert_allclose(
        scattering_angle(direction)['event', 0],
        sc.scalar(45.0, unit='deg').to(unit='rad'),
    )
    assert_allclose(
        theta(direction, normal),
        sc.array(dims=['event'], values=[15.0, 0.0], unit='deg').to(unit='rad'),
        atol=sc.scalar(1e-15, unit='rad'),
    )


def test_scattering_angle_with_gravity():
    # Horizontal rays at 1000 and 500 m/s fall 0.49 and 1.96 mm over 10 m.
    wavelength = (
        sc.constants.h
        / sc.constants.m_n
        / sc.array(dims=['event'], values=[1000.0, 500.0], unit='m/s')
    )
    direction = outgoing_direction(
        scattered_beam=sc.vectors(
            dims=['event'],
            values=[[0, -0.0004903325, 10], [0, -0.00196133, 10]],
            unit='m',
        ),
        wavelength=wavelength.to(unit='angstrom'),
        gravity=sc.vector([0.0, -9.80665, 0.0], unit='m/s^2'),
    )

    assert_allclose(
        scattering_angle(direction),
        sc.zeros(dims=['event'], shape=[2], unit='rad'),
        atol=sc.scalar(1e-10, unit='rad'),
    )


def test_incident_directions_are_computed_from_corresponding_slit_centers():
    upstream = sc.vectors(
        dims=['incident_beam'],
        values=[[0.0, 0.0, -2.0], [0.0, 0.1, -2.0], [0.0, 0.2, -2.0]],
        unit='m',
    )
    downstream = sc.vectors(
        dims=['incident_beam'],
        values=[[0.0, -0.1, -1.0], [0.0, 0.1, 0.0], [0.0, 0.5, 1.0]],
        unit='m',
    )

    result = incident_beam_directions(upstream, downstream)

    expected = downstream - upstream
    expected /= sc.norm(expected)
    assert_allclose(result, expected)


def test_assigns_each_neutron_to_closest_specular_incident_direction():
    incident_angles = sc.array(
        dims=['incident_beam'], values=[0.3, 1.0, 3.5], unit='deg'
    ).to(unit='rad')
    directions = sc.vectors(
        dims=['incident_beam'],
        values=[
            [0.0, -np.sin(angle), np.cos(angle)] for angle in incident_angles.values
        ],
    )
    reflection_angles = sc.array(dims=['event'], values=[0.4, 1.2, 3.0], unit='deg').to(
        unit='rad'
    )
    outgoing = sc.vectors(
        dims=['event'],
        values=[
            [0.0, np.sin(angle), np.cos(angle)] for angle in reflection_angles.values
        ],
    )
    normal = sc.vector([0.0, 1.0, 0.0])

    hypothetical = hypothetical_incident_direction(outgoing, normal)
    assigned = incident_direction(hypothetical, directions)

    assert_allclose(
        incident_angle(assigned, normal),
        incident_angles.rename_dims({'incident_beam': 'event'}),
    )
