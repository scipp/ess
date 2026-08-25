# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest

from ess.reduce.unwrap.interpolator_numba import (
    Interpolator as InterpolatorNumba,
)
from ess.reduce.unwrap.interpolator_scipy import (
    Interpolator as InterpolatorScipy,
)


def _f(x, y):
    """
    Function to interpolate, copied from Scipy docs. See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    """

    return 2 * x**3 + 3 * y**2


def _make_interpolators():
    time_edges = np.linspace(0, 71, 101)
    distance_edges = np.linspace(40, 70, 201)
    time_g, distance_g = np.meshgrid(
        time_edges, distance_edges, indexing='ij', sparse=True
    )
    values = _f(time_g, distance_g).T

    numba_interp = InterpolatorNumba(
        time_edges=time_edges, distance_edges=distance_edges, values=values
    )

    scipy_interp = InterpolatorScipy(
        time_edges=time_edges, distance_edges=distance_edges, values=values
    )
    return numba_interp, scipy_interp


def test_numba_and_scipy_interpolators_yield_same_results():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(0, 71, npoints)
    distances = rng.uniform(40, 70, npoints)

    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)

    # Do not use equal_nan because there should be no NaNs here
    assert np.allclose(numba_result, scipy_result)


def test_numba_and_scipy_interpolators_yield_same_results_with_pulse_offset():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(0, 71, npoints)
    distances = rng.uniform(40, 70, npoints)
    offsets = rng.uniform(0, 2, npoints)
    period = 1.0

    numba_result = numba_interp(times, distances, period, offsets)
    scipy_result = scipy_interp(times, distances, period, offsets)

    assert np.allclose(numba_result, scipy_result, equal_nan=True)


def test_numba_and_scipy_interpolators_yield_same_results_with_out_of_bounds():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(-1, 72, npoints)
    distances = rng.uniform(39, 71, npoints)

    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)

    assert np.allclose(numba_result, scipy_result, equal_nan=True)


def test_numba_and_scipy_interpolators_yield_same_results_with_values_on_edges():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 2

    times = np.array([0.0, 71.0])
    distances = rng.uniform(39, 71, npoints)
    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)
    assert np.allclose(numba_result, scipy_result, equal_nan=True)

    times = rng.uniform(0, 71, npoints)
    distances = np.array([40.0, 70.0])
    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)
    assert np.allclose(numba_result, scipy_result, equal_nan=True)


def _make_rectilinear_interpolators():
    """Interpolators on a distance axis that is dense in two places only.

    The shape a lookup table has when it samples the beamline where components
    sit and not at all in between.
    """
    time_edges = np.linspace(0, 71, 101)
    distance_edges = np.concatenate(
        [np.linspace(6.4, 6.8, 5), np.linspace(72.0, 72.4, 5)]
    )
    time_g, distance_g = np.meshgrid(
        time_edges, distance_edges, indexing='ij', sparse=True
    )
    values = _f(time_g, distance_g).T

    numba_interp = InterpolatorNumba(
        time_edges=time_edges, distance_edges=distance_edges, values=values
    )
    scipy_interp = InterpolatorScipy(
        time_edges=time_edges, distance_edges=distance_edges, values=values
    )
    return numba_interp, scipy_interp


def test_numba_and_scipy_interpolators_yield_same_results_on_rectilinear_grid():
    numba_interp, scipy_interp = _make_rectilinear_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(0, 71, npoints)
    distances = np.concatenate(
        [
            rng.uniform(6.4, 6.8, npoints // 2),
            rng.uniform(72.0, 72.4, npoints - npoints // 2),
        ]
    )

    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)

    assert np.allclose(numba_result, scipy_result)


def test_numba_and_scipy_interpolators_agree_across_a_gap_in_the_grid():
    # Between the dense regions the grid has one wide cell, which both
    # implementations interpolate across rather than treating as out of bounds.
    numba_interp, scipy_interp = _make_rectilinear_interpolators()

    times = np.array([10.0, 35.0, 60.0])
    distances = np.array([20.0, 40.0, 60.0])

    assert np.allclose(numba_interp(times, distances), scipy_interp(times, distances))


def test_uniform_grid_is_detected():
    # The uniform case takes a division rather than a search to locate a cell,
    # so it must still be recognized as uniform.
    numba_interp, _ = _make_interpolators()

    assert numba_interp.time_uniform
    assert numba_interp.distance_uniform


def test_rectilinear_grid_is_not_reported_as_uniform():
    numba_interp, _ = _make_rectilinear_interpolators()

    assert numba_interp.time_uniform
    assert not numba_interp.distance_uniform


@pytest.mark.parametrize('npoints', [0, 1])
def test_grid_too_short_to_bracket_a_value_raises(npoints: int):
    # A single grid point cannot bracket anything: it used to be read past the
    # end of the array.
    with pytest.raises(ValueError, match='at least two'):
        InterpolatorNumba(
            time_edges=np.linspace(0, 71, 101),
            distance_edges=np.linspace(40, 70, npoints),
            values=np.zeros((npoints, 101)),
        )
