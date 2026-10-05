# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose, assert_identical

from ess.reflectometry.tools import smooth_with_resolution


@pytest.fixture
def ideal() -> sc.DataArray:
    q = sc.linspace('Q', 0.0, 1.0, 10001, unit='1/angstrom')
    return sc.DataArray(sc.ones(sizes=q.sizes), coords={'Q': q})


@pytest.fixture
def measured() -> sc.DataArray:
    return sc.DataArray(
        sc.ones(sizes={'Q': 3}, with_variances=True),
        coords={
            'Q': sc.array(dims=['Q'], values=[0.3, 0.5, 0.7], unit='1/angstrom'),
            'Q_resolution': sc.array(
                dims=['Q'], values=[0.01, 0.02, 0.04], unit='1/angstrom'
            ),
        },
    )


def test_preserves_constant_curve_and_inputs(ideal, measured):
    ideal *= 0.25
    ideal.unit = 'counts'
    original_ideal = ideal.copy()
    original_measured = measured.copy()

    result = smooth_with_resolution(ideal, measured)

    assert_allclose(
        result.data, sc.full(sizes=measured.sizes, value=0.25, unit='counts')
    )
    assert_identical(result.coords['Q'], measured.coords['Q'])
    assert_identical(ideal, original_ideal)
    assert_identical(measured, original_measured)


@pytest.mark.parametrize('plateau_fraction', [0.0, 0.6])
def test_kernel_variance_matches_q_resolution(ideal, measured, plateau_fraction):
    ideal.values[:] = ideal.coords['Q'].values ** 2

    result = smooth_with_resolution(ideal, measured, plateau_fraction=plateau_fraction)

    # A normalized symmetric kernel maps x**2 to x**2 + sigma**2.
    # Linear interpolation of x**2 on this grid has error at most (1e-4)**2 / 4.
    expected = (
        measured.coords['Q'].values ** 2 + measured.coords['Q_resolution'].values ** 2
    )
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=3e-9)


def test_converts_units_and_uses_bin_centres(ideal, measured):
    ideal.values[:] = ideal.coords['Q'].values ** 2
    measured.coords['Q'] = sc.array(
        dims=['Q'], values=[2.0, 4.0, 6.0, 8.0], unit='1/nm'
    )
    measured.coords['Q_resolution'] = measured.coords['Q_resolution'].to(unit='1/m')

    result = smooth_with_resolution(ideal, measured)

    assert_identical(result.coords['Q'], sc.midpoints(measured.coords['Q']))
    np.testing.assert_allclose(
        result.values,
        [0.3**2 + 0.01**2, 0.5**2 + 0.02**2, 0.7**2 + 0.04**2],
        rtol=0.0,
        atol=3e-9,
    )


def test_zero_resolution_interpolates_at_measured_points(measured):
    ideal = sc.DataArray(
        sc.array(dims=['Q'], values=[1.0, 0.0, 1.0]),
        coords={'Q': sc.array(dims=['Q'], values=[0.0, 0.5, 1.0], unit='1/angstrom')},
    )
    measured.coords['Q_resolution'] *= 0.0

    result = smooth_with_resolution(ideal, measured)

    np.testing.assert_allclose(result.values, [0.4, 0.0, 0.4])


def test_invalid_resolution_and_incomplete_kernel_coverage_give_nan(ideal):
    measured = sc.DataArray(
        sc.ones(sizes={'Q': 7}),
        coords={
            'Q': sc.array(
                dims=['Q'],
                values=[0.01, 0.3, 0.4, 0.5, 0.6, 0.7, 0.99],
                unit='1/angstrom',
            ),
            'Q_resolution': sc.array(
                dims=['Q'],
                values=[0.01, 0.01, np.nan, np.inf, -0.01, 0.04, 0.04],
                unit='1/angstrom',
            ),
        },
    )

    result = smooth_with_resolution(ideal, measured)

    np.testing.assert_allclose(
        result.values, [np.nan, 1.0, np.nan, np.nan, np.nan, 1.0, np.nan]
    )
