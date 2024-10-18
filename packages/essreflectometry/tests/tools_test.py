# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from numpy.testing import assert_allclose as np_assert_allclose
from scipp.testing import assert_allclose

from ess.reflectometry.tools import combine_curves, scale_reflectivity_curves_to_overlap


def curve(d, qmin, qmax):
    return sc.DataArray(data=d, coords={'Q': sc.linspace('Q', qmin, qmax, len(d) + 1)})


def test_reflectivity_curve_scaling():
    data = sc.concat(
        (
            sc.ones(dims=['Q'], shape=[10], with_variances=True),
            0.5 * sc.ones(dims=['Q'], shape=[15], with_variances=True),
        ),
        dim='Q',
    )
    data.variances[:] = 0.1

    curves, factors = scale_reflectivity_curves_to_overlap(
        (curve(data, 0, 0.3), curve(0.8 * data, 0.2, 0.7), curve(0.1 * data, 0.6, 1.0)),
    )

    assert_allclose(curves[0].data, data, rtol=sc.scalar(1e-5))
    assert_allclose(curves[1].data, 0.5 * data, rtol=sc.scalar(1e-5))
    assert_allclose(curves[2].data, 0.25 * data, rtol=sc.scalar(1e-5))
    np_assert_allclose((1, 0.5 / 0.8, 0.25 / 0.1), factors, 1e-4)


def test_reflectivity_curve_scaling_with_critical_edge():
    data = sc.concat(
        (
            sc.ones(dims=['Q'], shape=[10], with_variances=True),
            0.5 * sc.ones(dims=['Q'], shape=[15], with_variances=True),
        ),
        dim='Q',
    )
    data.variances[:] = 0.1

    curves, factors = scale_reflectivity_curves_to_overlap(
        (
            2 * curve(data, 0, 0.3),
            curve(0.8 * data, 0.2, 0.7),
            curve(0.1 * data, 0.6, 1.0),
        ),
        critical_edge_interval=(sc.scalar(0.01), sc.scalar(0.05)),
    )

    assert_allclose(curves[0].data, data, rtol=sc.scalar(1e-5))
    assert_allclose(curves[1].data, 0.5 * data, rtol=sc.scalar(1e-5))
    assert_allclose(curves[2].data, 0.25 * data, rtol=sc.scalar(1e-5))
    np_assert_allclose((0.5, 0.5 / 0.8, 0.25 / 0.1), factors, 1e-4)


def test_combined_curves():
    qgrid = sc.linspace('Q', 0, 1, 26)
    data = sc.concat(
        (
            sc.ones(dims=['Q'], shape=[10], with_variances=True),
            0.5 * sc.ones(dims=['Q'], shape=[15], with_variances=True),
        ),
        dim='Q',
    )
    data.variances[:] = 0.1
    curves = (
        curve(data, 0, 0.3),
        curve(0.5 * data, 0.2, 0.7),
        curve(0.25 * data, 0.6, 1.0),
    )

    combined = combine_curves(curves, qgrid)
    assert_allclose(
        combined.data,
        sc.array(
            dims='Q',
            values=[
                1.0,
                1,
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0.125,
            ],
            variances=[
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.02,
                0.02,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.025,
                0.005,
                0.005,
                0.00625,
                0.00625,
                0.00625,
                0.00625,
                0.00625,
                0.00625,
                0.00625,
                0.00625,
            ],
        ),
    )
