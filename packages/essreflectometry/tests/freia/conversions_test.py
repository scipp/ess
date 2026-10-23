# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose, assert_identical

from ess.freia.conversions import (
    assign_incident_angle,
    assign_incident_angle_from_slit_logs,
    incident_beam_directions,
    outgoing_direction,
    scattering_angle,
    theta,
    time_at_slit,
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
    upstream = {
        'top': sc.vector([0.0, 0.0, -2.0], unit='m'),
        'middle': sc.vector([0.0, 0.1, -2.0], unit='m'),
        'bottom': sc.vector([0.0, 0.2, -2.0], unit='m'),
    }
    # Deliberately use a different order to ensure channels are paired by name.
    downstream = {
        'bottom': sc.vector([0.0, 0.5, 1.0], unit='m'),
        'top': sc.vector([0.0, -0.1, -1.0], unit='m'),
        'middle': sc.vector([0.0, 0.1, 0.0], unit='m'),
    }

    result = incident_beam_directions(upstream, downstream)

    assert result.keys() == upstream.keys()
    for key, upstream_center in upstream.items():
        expected = downstream[key] - upstream_center
        expected /= sc.norm(expected)
        assert_allclose(result[key], expected)


def test_incident_directions_require_matching_slit_names():
    upstream = {
        key: sc.vector([0.0, 0.0, -2.0], unit='m')
        for key in ('top', 'middle', 'bottom')
    }
    downstream = upstream.copy()
    downstream['other'] = downstream.pop('bottom')

    with pytest.raises(ValueError, match='must have the same keys'):
        incident_beam_directions(upstream, downstream)


@pytest.mark.parametrize('channels', [('top',), ('top', 'middle')])
def test_incident_directions_support_one_or_two_slit_channels(channels):
    upstream = {key: sc.vector([0.0, 0.0, -2.0], unit='m') for key in channels}
    downstream = {key: sc.vector([0.0, 0.0, -1.0], unit='m') for key in channels}

    result = incident_beam_directions(upstream, downstream)

    assert result.keys() == upstream.keys()


def test_assigns_each_neutron_to_closest_expected_reflection_angle():
    incident_angles = {
        'top': sc.scalar(0.3, unit='deg').to(unit='rad'),
        'middle': sc.scalar(1.0, unit='deg').to(unit='rad'),
        'bottom': sc.scalar(3.5, unit='deg').to(unit='rad'),
    }
    reflection_angles = sc.array(dims=['event'], values=[0.4, 1.2, 3.0], unit='deg').to(
        unit='rad'
    )

    assigned = assign_incident_angle(reflection_angles, incident_angles)

    assert_allclose(
        assigned,
        sc.array(dims=['event'], values=[0.3, 1.0, 3.5], unit='deg').to(unit='rad'),
    )


def test_single_incident_beam_returns_scalar():
    angle = sc.scalar(1.0, unit='rad')
    assigned = assign_incident_angle(
        sc.array(dims=['event'], values=[0.0, 2.0], unit='rad'), {'middle': angle}
    )

    assert_identical(assigned, angle)


def _events(offsets=(30, 40, 50, 60), speeds=(1000.0, 1000.0, 1000.0, 1000.0)):
    size = len(offsets)
    return {
        'event_time_zero': sc.datetimes(dims=['event'], values=[0] * size, unit='us'),
        'event_time_offset': sc.array(dims=['event'], values=offsets, unit='ms'),
        'wavelength': (
            sc.constants.h
            / sc.constants.m_n
            / sc.array(dims=['event'], values=speeds, unit='m/s')
        ).to(unit='nm'),
        'Ltotal': sc.scalar(30.0, unit='m'),
        'source_position': sc.vector([0.0, 0.0, 0.0], unit='m'),
    }


def _log(position, times, states):
    return sc.DataArray(
        sc.array(dims=['time'], values=states, dtype='bool'),
        coords={
            'time': sc.datetimes(dims=['time'], values=times, unit='ms'),
            'position': sc.vector([0.0, 0.0, position], unit='m'),
        },
    )


def test_time_at_slit_includes_emission_delay_and_uses_relative_source_distance():
    inputs = _events(offsets=[30, 32, 42], speeds=[1000.0] * 3)
    inputs['source_position'] = sc.vector([0.0, 0.0, -10.0], unit='m')
    result = time_at_slit(
        **inputs, slit_position=sc.vector([0.0, 0.0, 1000.0], unit='cm')
    )
    assert_identical(
        result,
        sc.datetimes(
            dims=['event'], values=[20_000_000, 22_000_000, 32_000_000], unit='ns'
        ),
    )


def test_time_at_slit_depends_on_wavelength_and_pixel_flight_path():
    inputs = _events(offsets=[32, 82], speeds=[1000.0, 500.0])
    inputs['Ltotal'] = sc.array(dims=['event'], values=[30.0, 40.0], unit='m')
    result = time_at_slit(**inputs, slit_position=sc.vector([0.0, 0.0, 20.0], unit='m'))
    assert_identical(
        result, sc.datetimes(dims=['event'], values=[22_000_000, 42_000_000], unit='ns')
    )


def test_each_slit_is_checked_at_its_own_distance_and_ambiguous_events_are_unassigned():
    angles = {'a': sc.scalar(0.3, unit='deg'), 'b': sc.scalar(1.0, unit='deg')}
    # At 20 m, crossing times are 20, 30, 40, 50 ms.
    # At 25 m, they are 25, 35, 45, 55 ms.
    # Reversed key order makes sure beams are paired by name.
    logs = {
        'b': _log(25.0, [0, 30, 50], [False, True, False]),
        'a': _log(20.0, [0, 25, 35, 45], [True, False, True, False]),
    }
    result = assign_incident_angle_from_slit_logs(
        **_events(), incident_angles=angles, slit_status_logs=logs
    )
    assert_allclose(
        result['event', :2],
        sc.array(dims=['event'], values=[0.3, 1.0], unit='deg').to(unit='rad'),
    )
    assert sc.all(sc.isnan(result['event', 2:])).value
    # An angular assignment would choose b for all four of these neutrons.
    nearest = assign_incident_angle(
        sc.full(dims=['event'], shape=[4], value=1.0, unit='deg'), angles
    )
    assert_allclose(nearest, sc.full(dims=['event'], shape=[4], value=1.0, unit='deg'))


def test_slit_assignment_uses_neutron_crossing_time_instead_of_pulse_or_detector_time():
    inputs = _events(offsets=[30, 32, 42], speeds=[1000.0] * 3)
    # Crossings at 20, 22, 32 ms. All pulse and detector times are outside
    # the opening, but the middle neutron crossed while the slit was open.
    result = assign_incident_angle_from_slit_logs(
        **inputs,
        incident_angles={'a': sc.scalar(1.0, unit='deg')},
        slit_status_logs={'a': _log(20.0, [0, 21, 23], [False, True, False])},
    )
    assert np.isnan(result.values[[0, 2]]).all()
    assert_allclose(result['event', 1], sc.scalar(1.0, unit='deg').to(unit='rad'))


def test_status_changes_apply_at_the_slit_crossing_timestamp():
    result = assign_incident_angle_from_slit_logs(
        **_events(offsets=[30, 31, 32, 33, 34], speeds=[1000.0] * 5),
        incident_angles={'a': sc.scalar(1.0, unit='deg')},
        slit_status_logs={'a': _log(20.0, [0, 21, 23], [False, True, False])},
    )
    assert_allclose(
        result,
        sc.array(
            dims=['event'], values=[np.nan, 1.0, 1.0, np.nan, np.nan], unit='deg'
        ).to(unit='rad'),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ('unknown_times', 'unknown_states'), [([], []), ([100], [False])]
)
def test_unknown_slit_status_does_not_silently_identify_another_beam(
    unknown_times, unknown_states
):
    result = assign_incident_angle_from_slit_logs(
        **_events(),
        incident_angles={
            'a': sc.scalar(0.3, unit='deg'),
            'b': sc.scalar(1.0, unit='deg'),
        },
        slit_status_logs={
            'a': _log(20.0, [0], [True]),
            'b': _log(25.0, unknown_times, unknown_states),
        },
    )
    assert sc.all(sc.isnan(result)).value
