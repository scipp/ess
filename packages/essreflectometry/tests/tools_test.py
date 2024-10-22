# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import sciline as sl
import scipp as sc
from numpy.testing import assert_allclose as np_assert_allclose
from orsopy.fileio import Orso, OrsoDataset
from scipp.testing import assert_allclose

from ess.reflectometry.orso import OrsoIofQDataset
from ess.reflectometry.tools import (
    combine_curves,
    linlogspace,
    orso_datasets_from_measurements,
    scale_reflectivity_curves_to_overlap,
)
from ess.reflectometry.types import Filename, ReflectivityOverQ, SampleRun


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


def test_linlogspace_linear():
    q_lin = linlogspace(
        dim='qz', edges=[0.008, 0.08], scale='linear', num=50, unit='1/angstrom'
    )
    expected = sc.linspace(dim='qz', start=0.008, stop=0.08, num=50, unit='1/angstrom')
    assert sc.allclose(q_lin, expected)


def test_linlogspace_linear_list_input():
    q_lin = linlogspace(
        dim='qz', edges=[0.008, 0.08], unit='1/angstrom', scale=['linear'], num=[50]
    )
    expected = sc.linspace(dim='qz', start=0.008, stop=0.08, num=50, unit='1/angstrom')
    assert sc.allclose(q_lin, expected)


def test_linlogspace_log():
    q_log = linlogspace(
        dim='qz', edges=[0.008, 0.08], unit='1/angstrom', scale='log', num=50
    )
    expected = sc.geomspace(dim='qz', start=0.008, stop=0.08, num=50, unit='1/angstrom')
    assert sc.allclose(q_log, expected)


def test_linlogspace_linear_log():
    q_linlog = linlogspace(
        dim='qz',
        edges=[0.008, 0.03, 0.08],
        unit='1/angstrom',
        scale=['linear', 'log'],
        num=[16, 20],
    )
    exp_lin = sc.linspace(dim='qz', start=0.008, stop=0.03, num=16, unit='1/angstrom')
    exp_log = sc.geomspace(dim='qz', start=0.03, stop=0.08, num=21, unit='1/angstrom')
    expected = sc.concat([exp_lin, exp_log['qz', 1:]], 'qz')
    assert sc.allclose(q_linlog, expected)


def test_linlogspace_log_linear():
    q_loglin = linlogspace(
        dim='qz',
        edges=[0.008, 0.03, 0.08],
        unit='1/angstrom',
        scale=['log', 'linear'],
        num=[16, 20],
    )
    exp_log = sc.geomspace(dim='qz', start=0.008, stop=0.03, num=16, unit='1/angstrom')
    exp_lin = sc.linspace(dim='qz', start=0.03, stop=0.08, num=21, unit='1/angstrom')
    expected = sc.concat([exp_log, exp_lin['qz', 1:]], 'qz')
    assert sc.allclose(q_loglin, expected)


def test_linlogspace_linear_log_linear():
    q_linloglin = linlogspace(
        dim='qz',
        edges=[0.008, 0.03, 0.08, 0.12],
        unit='1/angstrom',
        scale=['linear', 'log', 'linear'],
        num=[16, 20, 10],
    )
    exp_lin = sc.linspace(dim='qz', start=0.008, stop=0.03, num=16, unit='1/angstrom')
    exp_log = sc.geomspace(dim='qz', start=0.03, stop=0.08, num=21, unit='1/angstrom')
    exp_lin2 = sc.linspace(dim='qz', start=0.08, stop=0.12, num=11, unit='1/angstrom')
    expected = sc.concat([exp_lin, exp_log['qz', 1:], exp_lin2['qz', 1:]], 'qz')
    assert sc.allclose(q_linloglin, expected)


def test_linlogspace_bad_input():
    with pytest.raises(ValueError, match="Sizes do not match"):
        _ = linlogspace(
            dim='qz',
            edges=[0.008, 0.03, 0.08, 0.12],
            unit='1/angstrom',
            scale=['linear', 'log'],
            num=[16, 20],
        )


@pytest.mark.filterwarnings("ignore:No suitable")
def test_orso_datasets_tool():
    def normalized_ioq(filename: Filename[SampleRun]) -> ReflectivityOverQ:
        return filename

    def orso_dataset(filename: Filename[SampleRun]) -> OrsoIofQDataset:
        class Reduction:
            corrections = []  # noqa: RUF012

        return OrsoDataset(
            Orso({}, Reduction, [], name=f'{filename}.orso'), np.ones((0, 0))
        )

    workflow = sl.Pipeline(
        [normalized_ioq, orso_dataset], params={Filename[SampleRun]: 'default'}
    )
    datasets = orso_datasets_from_measurements(
        workflow,
        [{}, {Filename[SampleRun]: 'special'}],
        scale_to_overlap=False,
    )
    assert len(datasets) == 2
    assert tuple(d.info.name for d in datasets) == ('default.orso', 'special.orso')
