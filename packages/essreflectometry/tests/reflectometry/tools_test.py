# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import sciline as sl
import scipp as sc
from numpy.testing import assert_almost_equal
from orsopy.fileio import Orso, OrsoDataset
from scipp.testing import assert_allclose

from ess.reflectometry.tools import (
    BatchProcessor,
    batch_compute,
    batch_processor,
    combine_curves,
    linlogspace,
    scale_for_reflectivity_overlap,
)
from ess.reflectometry.types import (
    Filename,
    QBins,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRun,
)


def make_sample_events(scale, qmin, qmax):
    n1 = 10
    n2 = 15
    qbins = sc.linspace('Q', qmin, qmax, n1 + n2 + 1)
    data = sc.DataArray(
        data=sc.concat(
            (
                sc.ones(dims=['Q'], shape=[10], with_variances=True),
                0.5 * sc.ones(dims=['Q'], shape=[15], with_variances=True),
            ),
            dim='Q',
        )
        * scale,
        coords={'Q': sc.midpoints(qbins, 'Q')},
    )
    data.variances[:] = 0.1
    data.unit = 'counts'
    return data.bin(Q=qbins)


def make_reference_events(qmin, qmax):
    n = 25
    qbins = sc.linspace('Q', qmin, qmax, n + 1)
    data = sc.DataArray(
        data=sc.ones(dims=['Q'], shape=[n], with_variances=True),
        coords={'Q': sc.midpoints(qbins, 'Q')},
    )
    data.variances[:] = 0.1
    data.unit = 'counts'
    return data.bin(Q=qbins)


def make_workflow():
    def sample_data_from_filename(
        filename: Filename[SampleRun],
    ) -> ReducibleData[SampleRun]:
        return ReducibleData[SampleRun](
            make_sample_events(*(float(x) for x in filename.split('_')))
        )

    def reference_data_from_filename(
        filename: Filename[ReferenceRun],
    ) -> ReducibleData[ReferenceRun]:
        return ReducibleData[ReferenceRun](
            make_reference_events(*(float(x) for x in filename.split('_')))
        )

    def reflectivity(
        sample: ReducibleData[SampleRun],
        reference: ReducibleData[ReferenceRun],
        qbins: QBins,
    ) -> ReflectivityOverQ:
        return ReflectivityOverQ(sample.hist(Q=qbins) / reference.hist(Q=qbins))

    return sl.Pipeline(
        [sample_data_from_filename, reference_data_from_filename, reflectivity]
    )


def test_reflectivity_curve_scaling():
    wf = make_workflow()
    params = {'a': (1.0, 0, 0.3), 'b': (0.8, 0.2, 0.7), 'c': (0.1, 0.6, 1.0)}
    workflows = {}
    for k, v in params.items():
        workflows[k] = wf.copy()
        workflows[k][Filename[SampleRun]] = "_".join(map(str, v))
        workflows[k][Filename[ReferenceRun]] = "_".join(map(str, v[1:]))
        workflows[k][QBins] = make_reference_events(*v[1:]).coords['Q']

    batch = BatchProcessor(workflows)

    scaling_factors = scale_for_reflectivity_overlap(batch.compute(ReflectivityOverQ))

    assert np.isclose(scaling_factors['a'], 1.0)
    assert np.isclose(scaling_factors['b'], 0.5 / 0.8)
    assert np.isclose(scaling_factors['c'], 0.25 / 0.1)


def test_reflectivity_curve_scaling_with_critical_edge():
    wf = make_workflow()
    params = {'a': (2, 0, 0.3), 'b': (0.8, 0.2, 0.7), 'c': (0.1, 0.6, 1.0)}
    workflows = {}
    for k, v in params.items():
        workflows[k] = wf.copy()
        workflows[k][Filename[SampleRun]] = "_".join(map(str, v))
        workflows[k][Filename[ReferenceRun]] = "_".join(map(str, v[1:]))
        workflows[k][QBins] = make_reference_events(*v[1:]).coords['Q']

    batch = BatchProcessor(workflows)

    scaling_factors = scale_for_reflectivity_overlap(
        batch.compute(ReflectivityOverQ),
        critical_edge_interval=(sc.scalar(0.01), sc.scalar(0.05)),
    )

    assert np.isclose(scaling_factors['a'], 0.5)
    assert np.isclose(scaling_factors['b'], 0.5 / 0.8)
    assert np.isclose(scaling_factors['c'], 0.25 / 0.1)


def test_reflectivity_curve_scaling_works_with_single_curve_and_critical_edge():
    wf = make_workflow()
    wf[Filename[SampleRun]] = '2.5_0.4_0.8'
    wf[Filename[ReferenceRun]] = '0.4_0.8'
    wf[QBins] = make_reference_events(0.4, 0.8).coords['Q']

    scaling_factor = scale_for_reflectivity_overlap(
        wf.compute(ReflectivityOverQ),
        critical_edge_interval=(sc.scalar(0.0), sc.scalar(0.5)),
    )

    assert np.isclose(scaling_factor, 0.4)


def test_combined_curves():
    qgrid = sc.linspace('Q', 0, 1, 26)
    curves = (
        make_sample_events(1.0, 0, 0.3).hist(),
        0.5 * make_sample_events(1.0, 0.2, 0.7).hist(),
        0.25 * make_sample_events(1.0, 0.6, 1.0).hist(),
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
            unit='counts',
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
def test_batch_processor_uses_expected_parameters_from_each_run():
    def normalized_ioq(filename: Filename[SampleRun]) -> ReflectivityOverQ:
        return filename

    def orso_dataset(filename: Filename[SampleRun]) -> OrsoDataset:
        class Reduction:
            corrections = []  # noqa: RUF012

        return OrsoDataset(
            Orso({}, Reduction, [], name=f'{filename}.orso'), np.ones((0, 0))
        )

    workflow = sl.Pipeline(
        [normalized_ioq, orso_dataset], params={Filename[SampleRun]: 'default'}
    )

    batch = batch_processor(workflow, {'a': {}, 'b': {Filename[SampleRun]: 'special'}})

    results = batch.compute(OrsoDataset)
    assert len(results) == 2
    assert results['a'].info.name == 'default.orso'
    assert results['b'].info.name == 'special.orso'


def test_batch_processor_merges_event_lists():
    wf = make_workflow()

    runs = {
        'a': {Filename[SampleRun]: ('1.0_0.0_0.3', '1.5_0.0_0.3')},
        'b': {Filename[SampleRun]: '0.8_0.2_0.7'},
        'c': {Filename[SampleRun]: ('0.1_0.6_1.0', '0.2_0.6_1.0')},
    }
    batch = batch_processor(wf, runs)

    results = batch.compute(ReducibleData[SampleRun])

    assert_almost_equal(results['a'].sum().value, 10 + 15 * 0.5 + (10 + 15 * 0.5) * 1.5)
    assert_almost_equal(results['b'].sum().value, 10 * 0.8 + 15 * 0.5 * 0.8)
    assert_almost_equal(
        results['c'].sum().value, (10 + 15 * 0.5) * 0.1 + (10 + 15 * 0.5) * 0.2
    )


def test_batch_compute_single_target():
    def A(x: str) -> int:
        return int(x)

    params = {'a': {str: '1'}, 'b': {str: '2'}}
    workflow = sl.Pipeline([A])
    results = batch_compute(workflow, params, target=int)
    assert results == {'a': 1, 'b': 2}


def test_batch_compute_multiple_targets():
    def A(x: str) -> float:
        return float(x)

    def B(x: str) -> int:
        return int(x)

    params = {'a': {str: '1'}, 'b': {str: '2'}}
    workflow = sl.Pipeline([A, B])
    results = batch_compute(workflow, params, target=(float, int))
    assert results[float] == {'a': 1.0, 'b': 2.0}
    assert results[int] == {'a': 1, 'b': 2}


def test_batch_compute_does_not_recompute_reflectivity():
    R = sc.DataArray(
        sc.ones(dims=['Q'], shape=(50,), with_variances=True),
        coords={'Q': sc.linspace('Q', 0.1, 1, 50)},
    ).bin(Q=10)

    times_evaluated = 0

    def reflectivity() -> ReflectivityOverQ:
        nonlocal times_evaluated
        times_evaluated += 1
        return ReflectivityOverQ(R)

    def reducible_data() -> ReducibleData[SampleRun]:
        return ReducibleData[SampleRun](1.5)

    pl = sl.Pipeline([reflectivity, reducible_data])

    batch_compute(
        pl, {'a': {}, 'b': {}}, target=(ReflectivityOverQ,), scale_to_overlap=True
    )
    assert times_evaluated == 2


def test_batch_compute_applies_scaling_to_reflectivityoverq():
    R1 = sc.DataArray(
        sc.ones(dims=['Q'], shape=(50,), with_variances=True),
        coords={'Q': sc.linspace('Q', 0.1, 1, 50)},
    ).bin(Q=10)
    R2 = 0.5 * R1

    def reducible_data() -> ReducibleData[SampleRun]:
        return 1.5

    pl = sl.Pipeline([reducible_data])

    results = batch_compute(
        pl,
        {'a': {ReflectivityOverQ: R1}, 'b': {ReflectivityOverQ: R2}},
        target=ReflectivityOverQ,
        scale_to_overlap=(sc.scalar(0.0), sc.scalar(1.0)),
    )
    assert_allclose(results['a'], results['b'])


def test_batch_compute_applies_scaling_to_reducibledata():
    R1 = sc.DataArray(
        sc.ones(dims=['Q'], shape=(50,), with_variances=True),
        coords={'Q': sc.linspace('Q', 0.1, 1, 50)},
    ).bin(Q=10)
    R2 = 0.5 * R1

    def reducible_data() -> ReducibleData[SampleRun]:
        return sc.scalar(1)

    pl = sl.Pipeline([reducible_data])

    results = batch_compute(
        pl,
        {'a': {ReflectivityOverQ: R1}, 'b': {ReflectivityOverQ: R2}},
        target=ReducibleData[SampleRun],
        scale_to_overlap=(sc.scalar(0.0), sc.scalar(1.0)),
    )
    assert_allclose(results['a'], 0.5 * results['b'])
