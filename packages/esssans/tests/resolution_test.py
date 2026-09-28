# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.sans.conversions import sans_elastic
from ess.sans.resolution import (
    detector_q_variance,
    pixel_scattering_angle_variance,
    resolution_first_moment,
    resolution_second_moment,
)
from ess.sans.types import (
    CollimationLength,
    CorrectForGravity,
    DetectorPixelShape,
    NeXusTransformation,
    PixelScatteringAngleVariance,
    SampleApertureRadius,
    SampleRun,
    SourceApertureRadius,
)
from scipp.testing import assert_allclose

POSITIONS = np.array([[0.5, 0.0, 5.0], [0.0, 2.0, 4.0]])
WAVELENGTH_EDGES = np.array([2.0, 3.0, 5.0])


@pytest.fixture
def graph() -> dict:
    return sans_elastic(
        CorrectForGravity(False),
        sample_position=sc.vector([0.0, 0.0, 0.0], unit='m'),
        source_position=sc.vector([0.0, 0.0, -20.0], unit='m'),
        gravity=sc.vector([0.0, -9.81, 0.0], unit='m/s^2'),
    )


@pytest.fixture
def wavelength_bins() -> sc.Variable:
    return sc.array(dims=['wavelength'], values=WAVELENGTH_EDGES, unit='angstrom')


@pytest.fixture
def detector_term(graph: dict, wavelength_bins: sc.Variable) -> sc.DataArray:
    da = sc.DataArray(
        sc.array(
            dims=['pixel', 'wavelength'],
            values=[[1.0, 2.0], [3.0, 4.0]],
            variances=[[0.1, 0.2], [0.3, 0.4]],
        ),
        coords={
            'position': sc.vectors(dims=['pixel'], values=POSITIONS, unit='m'),
            'wavelength': sc.midpoints(wavelength_bins),
        },
    )
    return da.transform_coords(
        'Q', graph=graph, keep_intermediate=False, rename_dims=False
    )


def test_detector_q_variance_matches_mildner_carpenter(
    detector_term: sc.DataArray, graph: dict
) -> None:
    r1, r2, l1, spread = 0.015, 0.005, 5.0, 0.1
    pixel_variance = np.array([1e-6, 4e-6])
    variance = detector_q_variance(
        detector_term,
        graph=graph,
        source_spread=sc.scalar(spread, unit='angstrom'),
        pixel_variance=PixelScatteringAngleVariance[SampleRun](
            sc.array(dims=['pixel'], values=pixel_variance)
        ),
        source_aperture=SourceApertureRadius(sc.scalar(1000 * r1, unit='mm')),
        sample_aperture=SampleApertureRadius(sc.scalar(1000 * r2, unit='mm')),
        collimation_length=CollimationLength(sc.scalar(l1, unit='m')),
    )

    l2 = np.linalg.norm(POSITIONS, axis=1)[:, np.newaxis]
    two_theta = np.arccos(POSITIONS[:, 2:] / l2)
    wavelength = 0.5 * (WAVELENGTH_EDGES[1:] + WAVELENGTH_EDGES[:-1])
    q = 4 * np.pi * np.sin(two_theta / 2) / wavelength
    angular = (
        (r1 / l1) ** 2 / 4
        + (r2 * (1 / l1 + np.cos(two_theta) / l2)) ** 2 / 4
        + pixel_variance[:, np.newaxis]
    )
    expected = (2 * np.pi * np.cos(two_theta / 2) / wavelength) ** 2 * angular
    expected += q**2 * spread**2 / wavelength**2

    assert variance.dims == ('pixel', 'wavelength')
    assert variance.unit == '1/angstrom**2'
    np.testing.assert_allclose(variance.values, expected, rtol=1e-12)


def _points_in_cylinder(axis: np.ndarray, radius: float, n: int = 100) -> np.ndarray:
    """Midpoint grid of points, with equal volume per point, in a cylinder centered
    on the origin."""
    length = np.linalg.norm(axis)
    a = axis / length
    b = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    u = (np.arange(n) + 0.5) / n - 0.5
    # Radii with equal area per ring
    r = radius * np.sqrt((np.arange(n) + 0.5) / n)
    phi = 2 * np.pi * (np.arange(n) + 0.5) / n
    u, r, phi = (x.ravel() for x in np.meshgrid(u, r, phi, indexing='ij'))
    return (
        length * u[:, None] * a
        + (r * np.cos(phi))[:, None] * b
        + (r * np.sin(phi))[:, None] * c
    )


@pytest.mark.parametrize(
    'axis',
    [
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.012, -0.01, 0.013],
    ],
)
def test_pixel_scattering_angle_variance_matches_brute_force(
    graph: dict, axis: list[float]
) -> None:
    radius = 0.004
    positions = np.array([[0.5, 0.0, 5.0], [0.3, -0.8, 3.0], [-1.5, 0.2, 4.0]])
    detector = sc.DataArray(
        sc.ones(dims=['pixel'], shape=[len(positions)]),
        coords={'position': sc.vectors(dims=['pixel'], values=positions, unit='m')},
    )
    # The transformation is applied to the pixel shape, so we use a rotation to check
    # that it is taken into account.
    rotation = sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])
    shape = DetectorPixelShape[SampleRun](
        {
            'vertices': sc.vectors(
                dims=['vertex'],
                values=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, radius],
                    (sc.spatial.inv(rotation) * sc.vector(axis)).value,
                ],
                unit='m',
            )
        }
    )
    variance = pixel_scattering_angle_variance(
        detector,
        pixel_shape=shape,
        transform=NeXusTransformation[snx.NXdetector, SampleRun](rotation),
        graph=graph,
    )

    points = _points_in_cylinder(np.array(axis), radius)
    expected = []
    for position in positions:
        p = position + points
        two_theta = np.arccos(p[:, 2] / np.linalg.norm(p, axis=1))
        expected.append(np.var(two_theta))
    assert variance.unit == sc.units.one
    np.testing.assert_allclose(variance.values, expected, rtol=1e-3)


def test_resolution_moments_are_weighted_by_denominator_values(
    detector_term: sc.DataArray,
) -> None:
    variance = sc.array(
        dims=['pixel', 'wavelength'],
        values=[[1e-6, 2e-6], [3e-6, 4e-6]],
        unit='1/angstrom**2',
    )
    q = detector_term.coords['Q']
    n = sc.values(detector_term.data)

    first = resolution_first_moment(detector_term)
    second = resolution_second_moment(detector_term, variance=variance)

    assert_allclose(first.data, n * q)
    assert_allclose(second.data, n * (variance + q**2))
    assert sc.identical(first.coords['Q'], q)
    assert sc.identical(second.coords['Q'], q)
