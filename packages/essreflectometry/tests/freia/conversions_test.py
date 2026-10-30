# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

from ess.freia.conversions import (
    _shutter_state_at_time,
    assign_incident_angle,
    assign_incident_angle_from_shutter_logs,
    fixed_slits_incident_angle_graph,
    incident_beam_directions,
    outgoing_direction,
    scattering_angle,
    shutter_log_incident_angle_graph,
    theta,
    time_at_detector,
    time_at_shutter,
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
        'middle': sc.vector([0.0, 1.0, -2.0], unit='m'),
        'bottom': sc.vector([0.0, 2.0, -2.0], unit='m'),
    }
    # Deliberately use a different order to ensure channels are paired by name.
    downstream = {
        'bottom': sc.vector([0.0, 5.0, 2.0], unit='m'),
        'top': sc.vector([0.0, -3.0, 2.0], unit='m'),
        'middle': sc.vector([0.0, 1.0, 0.0], unit='m'),
    }

    result = incident_beam_directions(upstream, downstream)

    assert result.keys() == upstream.keys()
    assert_allclose(result['top'], sc.vector([0.0, -0.6, 0.8]))
    assert_allclose(result['middle'], sc.vector([0.0, 0.0, 1.0]))
    assert_allclose(result['bottom'], sc.vector([0.0, 0.6, 0.8]))


def test_incident_directions_require_matching_slit_names():
    upstream = {'top': sc.vector([0.0, 0.0, -2.0], unit='m')}
    downstream = {'bottom': sc.vector([0.0, 0.0, -1.0], unit='m')}

    with pytest.raises(ValueError, match='must have the same keys'):
        incident_beam_directions(upstream, downstream)


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


def test_fixed_slits_incidence_needs_no_event_coordinates():
    graph = fixed_slits_incident_angle_graph(
        upstream_slit_centers={'beam': sc.vector([0.0, 0.0, -2.0], unit='m')},
        downstream_slit_centers={
            'beam': sc.vector([0.0, -1.0, -2.0 + np.sqrt(3)], unit='m')
        },
        sample_surface_normal=sc.vector(
            [0.0, np.cos(np.deg2rad(10.0)), -np.sin(np.deg2rad(10.0))]
        ),
    )
    data = sc.DataArray(sc.ones(dims=['event'], shape=[2], unit='counts'))

    result = data.transform_coords('incident_angle', graph=graph)

    # The beam is 30 degrees below horizontal and the sample is tilted by 10 degrees.
    assert_allclose(
        result.coords['incident_angle'], sc.scalar(40.0, unit='deg').to(unit='rad')
    )


def _events(offsets=(30, 40, 50, 60), speeds=None):
    size = len(offsets)
    if speeds is None:
        speeds = [1000.0] * size
    return {
        'event_time_zero': sc.datetimes(dims=['event'], values=[0] * size, unit='us'),
        'event_time_offset': sc.array(dims=['event'], values=offsets, unit='ms'),
        'wavelength': (
            sc.constants.h
            / sc.constants.m_n
            / sc.array(dims=['event'], values=speeds, unit='m/s')
        ).to(unit='nm'),
        'L2': sc.scalar(5.0, unit='m'),
        'sample_position': sc.vector([0.0, 0.0, 25.0], unit='m'),
    }


def _log(times, states):
    return sc.DataArray(
        sc.array(dims=['time'], values=states, dtype='bool'),
        coords={
            'time': sc.datetimes(dims=['time'], values=times, unit='ms'),
        },
    )


def test_time_at_shutter_uses_path_via_sample_and_retains_emission_delay():
    inputs = _events(offsets=[30, 32, 42])
    # The shutter-to-sample segment is 5 m (a 3-4-5 triangle), followed by L2=5 m.
    inputs['sample_position'] = sc.vector([3.0, 0.0, 14.0], unit='m')
    inputs['L2'] = sc.scalar(500.0, unit='cm')
    result = time_at_shutter(
        time_at_detector(inputs['event_time_zero'], inputs['event_time_offset']),
        inputs['wavelength'],
        inputs['L2'],
        inputs['sample_position'],
        shutter_position=sc.vector([0.0, 0.0, 1000.0], unit='cm'),
    )
    assert_allclose(
        result,
        sc.array(dims=['event'], values=[0.020, 0.022, 0.032], unit='s'),
    )


def test_time_at_shutter_depends_on_wavelength_and_pixel_flight_path():
    inputs = _events(offsets=[32, 82], speeds=[1000.0, 500.0])
    inputs['L2'] = sc.array(dims=['event'], values=[5.0, 15.0], unit='m')
    result = time_at_shutter(
        time_at_detector(inputs['event_time_zero'], inputs['event_time_offset']),
        inputs['wavelength'],
        inputs['L2'],
        inputs['sample_position'],
        shutter_position=sc.vector([0.0, 0.0, 20.0], unit='m'),
    )
    assert_allclose(result, sc.array(dims=['event'], values=[0.022, 0.042], unit='s'))


def test_shutter_assignment_requires_exactly_one_open_shutter():
    angles = {'a': sc.scalar(0.3, unit='deg'), 'b': sc.scalar(1.0, unit='deg')}
    # Reversed key order makes sure beams are paired by name.
    logs = {
        'b': _log([0, 25, 45], [False, True, False]),
        'a': _log([0, 25, 35, 45], [True, False, True, False]),
    }
    # At crossings of 20, 30, 40, 50 ms: only a, only b, both, neither are open.
    result = assign_incident_angle_from_shutter_logs(
        **_events(),
        incident_angles=angles,
        shutter_positions={
            name: sc.vector([0.0, 0.0, 20.0], unit='m') for name in angles
        },
        shutter_status_logs=logs,
    )
    assert_allclose(
        result,
        sc.array(dims=['event'], values=[0.3, 1.0, np.nan, np.nan], unit='deg').to(
            unit='rad'
        ),
        equal_nan=True,
    )


def test_shutter_assignment_uses_crossing_time_instead_of_pulse_or_detector_time():
    inputs = _events(offsets=[30, 32, 42])
    # Crossings at 20, 22, 32 ms. All pulse and detector times are outside
    # the opening, but the middle neutron crossed while the shutter was open.
    result = assign_incident_angle_from_shutter_logs(
        **inputs,
        incident_angles={'a': sc.scalar(1.0, unit='deg')},
        shutter_positions={'a': sc.vector([0.0, 0.0, 20.0], unit='m')},
        shutter_status_logs={'a': _log([0, 21, 23], [False, True, False])},
    )
    assert_allclose(
        result,
        sc.array(dims=['event'], values=[np.nan, 1.0, np.nan], unit='deg').to(
            unit='rad'
        ),
        equal_nan=True,
    )


def test_status_changes_apply_at_the_shutter_crossing_timestamp():
    result = _shutter_state_at_time(
        sc.array(
            dims=['event'], values=[0.9998, 1.0, 1.0002, 2.9998, 3.0, 3.0002], unit='s'
        ),
        _log([0, 1000, 3000], [False, True, False]),
    )
    np.testing.assert_array_equal(result.values, [0, 1, 1, 1, 0, 0])


def test_shutter_assignment_resolves_hundreds_of_microseconds_at_modern_dates():
    inputs = _events(offsets=[30.8, 31.2, 32.8, 33.2])
    pulse = sc.datetime('2026-10-30T12:34:56.123456789', unit='ns')
    inputs['event_time_zero'] = sc.concat([pulse] * 4, dim='event')
    log = _log([0, 21, 23], [False, True, False])
    log.coords['time'] = log.coords['time'].to(unit='ns') + (
        pulse - sc.epoch(unit='ns')
    )
    result = assign_incident_angle_from_shutter_logs(
        **inputs,
        incident_angles={'a': sc.scalar(1.0, unit='deg')},
        shutter_positions={'a': sc.vector([0.0, 0.0, 20.0], unit='m')},
        shutter_status_logs={'a': log},
    )
    assert_allclose(
        result,
        sc.array(dims=['event'], values=[np.nan, 1.0, 1.0, np.nan], unit='deg').to(
            unit='rad'
        ),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ('unknown_times', 'unknown_states'),
    [([], []), ([100], [False])],
    ids=['empty_log', 'before_first_entry'],
)
def test_unknown_shutter_status_does_not_silently_identify_another_beam(
    unknown_times, unknown_states
):
    result = assign_incident_angle_from_shutter_logs(
        **_events(),
        incident_angles={
            'a': sc.scalar(0.3, unit='deg'),
            'b': sc.scalar(1.0, unit='deg'),
        },
        shutter_positions={
            name: sc.vector([0.0, 0.0, 20.0], unit='m') for name in ('a', 'b')
        },
        shutter_status_logs={
            'a': _log([0], [True]),
            'b': _log(unknown_times, unknown_states),
        },
    )
    assert sc.all(sc.isnan(result)).value


def test_shutter_logs_are_sampled_at_each_shutters_crossing_time():
    inputs = _events(offsets=[30, 40, 50])
    positions = {
        'a': sc.vector([0.0, 0.0, 10.0], unit='m'),
        'b': sc.vector([0.0, 0.0, 20.0], unit='m'),
    }
    # Identical logs, but the neutron reaches a at 10, 20, 30 ms and b at
    # 20, 30, 40 ms. A shared timestamp would never identify a unique beam.
    result = assign_incident_angle_from_shutter_logs(
        **inputs,
        shutter_positions=positions,
        incident_angles={
            'a': sc.scalar(0.3, unit='deg'),
            'b': sc.scalar(1.0, unit='deg'),
        },
        shutter_status_logs={
            name: _log([0, 15, 25], [False, True, False]) for name in positions
        },
    )
    assert_allclose(
        result,
        sc.array(dims=['event'], values=[1.0, 0.3, np.nan], unit='deg').to(unit='rad'),
        equal_nan=True,
    )


def test_final_shutter_state_persists():
    result = assign_incident_angle_from_shutter_logs(
        **_events(offsets=[30.2, 60]),
        incident_angles={'a': sc.scalar(1.0, unit='deg')},
        shutter_positions={'a': sc.vector([0.0, 0.0, 20.0], unit='m')},
        shutter_status_logs={'a': _log([0, 20], [False, True])},
    )
    assert_allclose(
        result, sc.array(dims=['event'], values=[1.0, 1.0], unit='deg').to(unit='rad')
    )


@pytest.mark.parametrize('input_name', ['shutter_positions', 'shutter_status_logs'])
def test_shutter_graph_requires_matching_beam_names(input_name):
    inputs = {
        'upstream_slit_centers': {'a': sc.vector([0.0, 0.1, -2.0], unit='m')},
        'downstream_slit_centers': {'a': sc.vector([0.0, 0.0, -1.0], unit='m')},
        'sample_surface_normal': sc.vector([0.0, 1.0, 0.0]),
        'shutter_positions': {'a': sc.vector([0.0, 0.0, -3.0], unit='m')},
        'shutter_status_logs': {'a': _log([0], [True])},
    }
    inputs[input_name]['b'] = inputs[input_name].pop('a')
    with pytest.raises(ValueError, match='must have the same keys'):
        shutter_log_incident_angle_graph(**inputs)


def test_shutter_graph_requires_at_least_one_beam():
    with pytest.raises(ValueError, match='At least one incident beam'):
        shutter_log_incident_angle_graph({}, {}, sc.vector([0.0, 1.0, 0.0]), {}, {})
