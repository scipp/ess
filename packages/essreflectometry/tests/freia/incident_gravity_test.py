# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from scipp.constants import h, m_n
from scipp.testing import assert_allclose, assert_identical
from scippnexus import NXsample, NXsource
from scipy.optimize import root_scalar
from scipy.spatial.transform import Rotation

from ess.freia import FreiaOffspecWorkflow
from ess.freia.conversions import (
    closest_beam_incident_angle_graph,
    fixed_slits_incident_angle_graph,
    shutter_log_incident_angle_graph,
)
from ess.freia.mirror import mirror_geometry
from ess.freia.types import (
    DownstreamSlitCenters,
    MirrorNormal,
    MirrorPosition,
    SampleSurfaceNormal,
    ShutterPositions,
    ShutterStatusLogs,
    UpstreamSlitCenters,
)
from ess.reflectometry.types import CorrectedDetector, SampleRun, WavelengthDetector


def _geometry(mirror_tilt=0.0, *, a=1.0, b=1.0, c=0.375):
    """A nominal 0.3-degree ray through two slits, with the mirror at the origin."""
    beta = np.deg2rad(-0.3 if mirror_tilt is None else 0.3)
    alpha = beta if mirror_tilt is None else 2 * np.deg2rad(mirror_tilt) - beta
    direction = sc.vector([0.0, np.sin(beta), np.cos(beta)])
    return {
        'upstream_slit_centers': {
            'beam': sc.vector([0.0, -a * np.sin(alpha), -a * np.cos(alpha)], unit='m')
        },
        'downstream_slit_centers': {'beam': sc.scalar(b, unit='m') * direction},
        'sample_position': sc.scalar(b + c, unit='m') * direction,
        'sample_surface_normal': sc.vector(
            [0.0, 1.0 if mirror_tilt is None else -1.0, 0.0]
        ),
        'gravity': sc.vector([0.0, -9.80665, 0.0], unit='m/s^2'),
        'mirror': None
        if mirror_tilt is None
        else mirror_geometry(
            sc.vector([0.0, 0.0, 0.0], unit='m'),
            sc.vector(
                [0.0, np.cos(np.deg2rad(mirror_tilt)), -np.sin(np.deg2rad(mirror_tilt))]
            ),
        ),
    }


def _angles(geometry, wavelength):
    graph = fixed_slits_incident_angle_graph(**geometry)
    coords = {'wavelength': wavelength, 'gravity': geometry['gravity']}
    data = sc.DataArray(sc.zeros(sizes=wavelength.sizes), coords=coords)
    return data.transform_coords('incident_angle', graph=graph).coords['incident_angle']


def _ballistic_angle(geometry, wavelength, mirror_tilt):
    """Independent 2-D shooting solution with exact parabolic free flights.

    Find the launch angle at fixed speed that passes through both slit centres.
    Intersect the actual trajectories with the mirror and sample planes, rather
    than using nominal flight times or the first-order correction coefficient.
    """

    def yz(vector):
        return vector.value[[2, 1]]  # longitudinal, vertical

    gravity = yz(geometry['gravity'])
    upstream = yz(geometry['upstream_slit_centers']['beam'])
    downstream = yz(geometry['downstream_slit_centers']['beam'])
    sample = yz(geometry['sample_position'])
    sample_normal = yz(geometry['sample_surface_normal'])
    speed = (h / m_n / sc.scalar(wavelength, unit='angstrom')).to(unit='m/s').value

    def intersect(point, velocity, plane_point, normal):
        coefficients = [
            0.5 * np.dot(gravity, normal),
            np.dot(velocity, normal),
            np.dot(point - plane_point, normal),
        ]
        roots = np.roots(np.trim_zeros(coefficients, 'f'))
        time = min(t.real for t in roots if abs(t.imag) < 1e-12 and t.real > 1e-10)
        return (
            point + velocity * time + 0.5 * gravity * time**2,
            velocity + gravity * time,
        )

    def at_downstream(launch_angle):
        point = upstream
        velocity = speed * np.array([np.cos(launch_angle), np.sin(launch_angle)])
        if mirror_tilt is not None:
            tilt = np.deg2rad(mirror_tilt)
            normal = np.array([-np.sin(tilt), np.cos(tilt)])
            point, velocity = intersect(point, velocity, np.zeros(2), normal)
            velocity = velocity - 2 * np.dot(velocity, normal) * normal
        return intersect(point, velocity, downstream, np.array([1.0, 0.0]))

    nominal = downstream - upstream if mirror_tilt is None else -upstream
    guess = np.arctan2(nominal[1], nominal[0])
    root = root_scalar(
        lambda angle: at_downstream(angle)[0][1] - downstream[1],
        x0=guess,
        x1=guess + 1e-6,
        xtol=1e-13,
    )
    assert root.converged
    point, velocity = at_downstream(root.root)
    _, velocity = intersect(point, velocity, sample, sample_normal)
    return np.arcsin(-np.dot(sample_normal, velocity) / np.linalg.norm(velocity))


@pytest.mark.parametrize(
    ('mirror_tilt', 'a', 'b'),
    [
        (None, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (0.0, 1.8, 0.2),
    ],
)
@pytest.mark.parametrize(
    ('wavelength', 'tolerance'), [(2.0, 1e-8), (10.0, 1e-6), (18.0, 1e-5)]
)
def test_incidence_matches_exact_ballistic_trajectories(
    mirror_tilt, a, b, wavelength, tolerance
):
    geometry = _geometry(mirror_tilt, a=a, b=b)
    result = _angles(geometry, sc.scalar(wavelength, unit='angstrom'))
    exact = _ballistic_angle(geometry, wavelength, mirror_tilt)
    # Tolerance allows the neglected second-order terms, largest at long lambda.
    assert_allclose(
        result, sc.scalar(exact, unit='rad'), atol=sc.scalar(tolerance, unit='rad')
    )


def test_coefficient_is_invariant_under_rotation_translation_and_length_units():
    geometry = _geometry(-1.0)
    wavelengths = sc.array(dims=['event'], values=[0.2, 0.5, 1.0], unit='nm')
    expected = _angles(geometry, wavelengths)
    rotation = sc.spatial.linear_transform(
        value=Rotation.from_rotvec([0.3, -0.4, 0.1]).as_matrix()
    )
    shift = sc.vector([0.2, -0.1, 4.0], unit='m')
    # Moving the reference point within the sample plane must not change c.
    geometry['sample_position'] += sc.vector([0.3, 0.0, 0.7], unit='m')
    for key, unit in [
        ('upstream_slit_centers', 'mm'),
        ('downstream_slit_centers', 'cm'),
    ]:
        geometry[key] = {
            'beam': (rotation * geometry[key]['beam'] + shift).to(unit=unit)
        }
    geometry['sample_position'] = (rotation * geometry['sample_position'] + shift).to(
        unit='mm'
    )
    geometry['sample_surface_normal'] = rotation * geometry['sample_surface_normal']
    geometry['gravity'] = rotation * geometry['gravity']
    tilt = np.deg2rad(-1.0)
    geometry['mirror'] = mirror_geometry(
        shift, rotation * sc.vector([0.0, np.cos(tilt), -np.sin(tilt)])
    )
    assert_allclose(_angles(geometry, wavelengths), expected)


@pytest.mark.parametrize(('detector_angle', 'selected'), [(0.303, 0), (0.307, 1)])
def test_closest_selection_uses_nominal_geometry_before_correcting_incidence(
    detector_angle, selected
):
    angles = {'low': 0.3, 'high': 0.31}
    graph = closest_beam_incident_angle_graph(
        upstream_slit_centers={
            name: sc.vector([0.0, 2.5 * np.tan(np.deg2rad(angle)), -2.5], unit='m')
            for name, angle in angles.items()
        },
        downstream_slit_centers={
            name: sc.vector([0.0, 0.5 * np.tan(np.deg2rad(angle)), -0.5], unit='m')
            for name, angle in angles.items()
        },
        sample_surface_normal=sc.vector([0.0, 1.0, 0.0]),
        sample_position=sc.vector([0.0, 0.0, 0.0], unit='m'),
        gravity=sc.vector([0.0, -9.80665, 0.0], unit='m/s^2'),
    )
    wavelengths = sc.array(dims=['event'], values=[2.0, 10.0], unit='angstrom')
    result = sc.DataArray(
        sc.ones(dims=['event'], shape=[2]),
        coords={
            'wavelength': wavelengths,
            'sample_position': sc.vector([0.0, 0.0, 0.0], unit='m'),
            'gravity': sc.vector([0.0, -9.80665, 0.0], unit='m/s^2'),
            'scattered_beam': sc.vector(
                [
                    0.0,
                    3 * np.sin(np.deg2rad(detector_angle)),
                    3 * np.cos(np.deg2rad(detector_angle)),
                ],
                unit='m',
            ),
            # A corrected outgoing angle must not influence beam identification.
            'reflection_angle': sc.array(
                dims=['event'], values=[0.32, 0.29], unit='deg'
            ).to(unit='rad'),
        },
    ).transform_coords(
        ['nominal_incident_angle', 'incident_gravity_length', 'incident_angle'],
        graph=graph,
    )
    # Both wavelengths select the same beam; the correction is applied afterwards.
    assert_allclose(
        result.coords['nominal_incident_angle'],
        sc.scalar([0.3, 0.31][selected], unit='deg').to(unit='rad'),
    )
    assert_allclose(result.coords['incident_gravity_length'], sc.scalar(1.5, unit='m'))
    delta = (
        sc.scalar(9.80665 * 1.5, unit='m^2/s^2') * (m_n / h * wavelengths) ** 2
    ).to(unit='dimensionless') * sc.scalar(1.0, unit='rad')
    expected = sc.scalar([0.3, 0.31][selected], unit='deg').to(unit='rad') + delta
    assert_allclose(result.coords['incident_angle'], expected)


@pytest.mark.parametrize('method', ['fixed_slits', 'closest', 'shutter_logs'])
@pytest.mark.parametrize('inversion', [False, True])
def test_workflow_corrects_binned_incidence_and_preserves_counts(method, inversion):
    geometry = _geometry(0.0 if inversion else None)
    wf = FreiaOffspecWorkflow(incident_beam_from=method, inversion=inversion)
    wf[GravityVector] = geometry['gravity']
    if inversion:
        wf[MirrorPosition[SampleRun]] = geometry['mirror'].position
        wf[MirrorNormal[SampleRun]] = geometry['mirror'].normal
    wf[Position[NXsample, SampleRun]] = geometry['sample_position']
    wf[Position[NXsource, SampleRun]] = 5 * geometry['upstream_slit_centers']['beam']
    wf[SampleSurfaceNormal[SampleRun]] = geometry['sample_surface_normal']
    wf[UpstreamSlitCenters[SampleRun]] = geometry['upstream_slit_centers']
    wf[DownstreamSlitCenters[SampleRun]] = geometry['downstream_slit_centers']
    wavelengths = sc.array(dims=['event'], values=[2.0, 10.0, 18.0], unit='angstrom')
    data = sc.DataArray(
        sc.ones(dims=['event'], shape=[3], with_variances=True, unit='counts'),
        coords={
            'wavelength': wavelengths,
            'pixel': sc.array(dims=['event'], values=[0, 0, 1], unit=None),
            'event_time_zero': sc.datetimes(
                dims=['event'], values=[0, 0, 0], unit='ns'
            ),
            'event_time_offset': sc.full(
                dims=['event'], shape=[3], value=0.05, unit='s'
            ),
        },
    ).group('pixel')
    beta = np.deg2rad(-0.3 if inversion else 0.3)
    pixel_position = geometry['sample_position'] + sc.vector(
        [0.0, 3 * np.sin(beta), 3 * np.cos(beta)], unit='m'
    )
    data.coords['position'] = sc.concat([pixel_position, pixel_position], dim='pixel')
    wf[WavelengthDetector[SampleRun]] = data
    if method == 'shutter_logs':
        wf[ShutterPositions[SampleRun]] = geometry['upstream_slit_centers']
        wf[ShutterStatusLogs[SampleRun]] = {
            'beam': sc.DataArray(
                sc.array(dims=['time'], values=[True]),
                coords={'time': sc.datetimes(dims=['time'], values=[0], unit='ns')},
            )
        }
    result = wf.compute(CorrectedDetector[SampleRun])
    # Analytic horizontal-mirror / ordinary-slit limits for a=b=1, c=.375.
    effective_length = -0.875 if inversion else 1.375
    delta = (
        sc.scalar(9.80665 * effective_length * np.cos(beta), unit='m^2/s^2')
        * (m_n / h * wavelengths) ** 2
    ).to(unit='dimensionless') * sc.scalar(1.0, unit='rad')
    expected = sc.scalar(0.3, unit='deg').to(unit='rad') + delta
    assert_allclose(
        result.bins.coords['incident_angle'].bins.constituents['data'], expected
    )
    assert_identical(result.bins.data, data.bins.data)
    assert result.sum().value == 3.0
    assert sc.all(sc.isfinite(result.bins.coords['Qx'])).value
    assert sc.all(sc.isfinite(result.bins.coords['Qz'])).value


@pytest.mark.parametrize('dtype', ['int64', 'float32', 'float64'])
def test_incident_correction_accepts_numeric_wavelengths_without_modifying_them(dtype):
    wavelengths = sc.array(dims=['event'], values=[2, 10], unit='angstrom', dtype=dtype)
    original = wavelengths.copy()
    geometry = _geometry(0.0)
    expected = _angles(geometry, wavelengths.to(dtype='float64'))
    assert_allclose(_angles(geometry, wavelengths), expected)
    assert_identical(wavelengths, original)


def test_shutter_selection_assigns_distinct_lengths_with_equal_nominal_angles():
    angle = sc.scalar(0.3, unit='deg').to(unit='rad')
    direction = sc.vector([0.0, -np.sin(angle.value), np.cos(angle.value)])
    upstream = {
        name: sc.scalar(-distance, unit='m') * direction
        for name, distance in [('a', 3.0), ('b', 5.0)]
    }
    graph = shutter_log_incident_angle_graph(
        upstream_slit_centers=upstream,
        downstream_slit_centers={
            name: sc.scalar(-1.0, unit='m') * direction for name in upstream
        },
        sample_surface_normal=sc.vector([0.0, 1.0, 0.0]),
        sample_position=sc.vector([0.0, 0.0, 0.0], unit='m'),
        gravity=sc.vector([0.0, -9.80665, 0.0], unit='m/s^2'),
        shutter_positions=upstream,
        shutter_status_logs={
            name: sc.DataArray(
                sc.array(dims=['time'], values=states),
                coords={'time': sc.datetimes(dims=['time'], values=[0, 40], unit='ms')},
            )
            for name, states in [('b', [False, True]), ('a', [True, False])]
        },
    )
    wavelength = sc.scalar(10.0, unit='angstrom')
    result = sc.DataArray(
        sc.ones(sizes={'event': 2}),
        coords={
            'wavelength': wavelength,
            'gravity': sc.vector([0.0, -9.80665, 0.0], unit='m/s^2'),
            'sample_position': sc.vector([0.0, 0.0, 0.0], unit='m'),
            'L2': sc.scalar(1.0, unit='m'),
            'event_time_zero': sc.datetime(0, unit='ns'),
            'event_time_offset': sc.array(dims=['event'], values=[30, 60], unit='ms'),
        },
    ).transform_coords(
        ['nominal_incident_angle', 'incident_gravity_length', 'incident_angle'],
        graph=graph,
    )
    assert_allclose(
        result.coords['nominal_incident_angle'], sc.concat([angle, angle], dim='event')
    )
    assert_allclose(
        result.coords['incident_gravity_length'],
        sc.array(dims=['event'], values=[2.0, 3.0], unit='m') * np.cos(angle.value),
    )
    effective_lengths = sc.array(dims=['event'], values=[2.0, 3.0], unit='m')
    delta = (
        sc.scalar(9.80665 * np.cos(angle.value), unit='m/s^2')
        * effective_lengths
        * (m_n / h * wavelength) ** 2
    ).to(unit='dimensionless') * sc.scalar(1.0, unit='rad')
    assert_allclose(result.coords['incident_angle'], angle + delta)
