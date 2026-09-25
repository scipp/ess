# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from ess.sans.conversions import sans_elastic
from ess.sans.resolution import (
    detector_q_variance,
    resolution_first_moment,
    resolution_second_moment,
)
from ess.sans.types import (
    CollimationLength,
    CorrectForGravity,
    DetectorPixelSize,
    SampleApertureRadius,
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
    detector_term: sc.DataArray, graph: dict, wavelength_bins: sc.Variable
) -> None:
    r1, r2, l1, d_r, spread = 0.015, 0.005, 5.0, 0.008, 0.1
    variance = detector_q_variance(
        detector_term,
        graph=graph,
        source_spread=sc.scalar(spread, unit='angstrom'),
        wavelength_bins=wavelength_bins,
        source_aperture=SourceApertureRadius(sc.scalar(1000 * r1, unit='mm')),
        sample_aperture=SampleApertureRadius(sc.scalar(1000 * r2, unit='mm')),
        collimation_length=CollimationLength(sc.scalar(l1, unit='m')),
        pixel_size=DetectorPixelSize(sc.scalar(1000 * d_r, unit='mm')),
    )

    l2 = np.linalg.norm(POSITIONS, axis=1)[:, np.newaxis]
    two_theta = np.arccos(POSITIONS[:, 2:] / l2)
    wavelength = 0.5 * (WAVELENGTH_EDGES[1:] + WAVELENGTH_EDGES[:-1])
    d_wavelength = np.diff(WAVELENGTH_EDGES)
    q = 4 * np.pi * np.sin(two_theta / 2) / wavelength
    angular = (
        3 * (r1 / l1) ** 2
        + 3 * (r2 * (1 / l1 + 1 / l2)) ** 2
        + (d_r * np.cos(two_theta) / l2) ** 2
    ) / 12
    expected = (2 * np.pi * np.cos(two_theta / 2) / wavelength) ** 2 * angular
    expected += q**2 * (spread**2 + d_wavelength**2 / 12) / wavelength**2

    assert variance.dims == ('pixel', 'wavelength')
    assert variance.unit == '1/angstrom**2'
    np.testing.assert_allclose(variance.values, expected, rtol=1e-12)


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
