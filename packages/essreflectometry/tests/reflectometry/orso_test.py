# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from datetime import datetime
from math import isnan

import numpy as np
import pytest
import sciline
import scipp as sc
from orsopy import fileio

from ess import amor, reflectometry
from ess.amor import data  # noqa: F401
from ess.reflectometry import orso
from ess.reflectometry.types import (
    Filename,
    ReducibleData,
    ReferenceRun,
    RunType,
    SampleRun,
)


def test_build_orso_data_source():
    pipeline = sciline.Pipeline(
        (*amor.load.providers, *orso.providers),
        params={
            Filename[SampleRun]: amor.data.amor_old_sample_run(),
            Filename[ReferenceRun]: amor.data.amor_old_reference_run(),
        },
        constraints={RunType: [SampleRun, ReferenceRun]},
    )
    pipeline[orso.OrsoInstrument] = None
    data_source = pipeline.compute(orso.OrsoDataSource)
    expected = fileio.data_source.DataSource(
        owner=fileio.base.Person(
            name="J. Stahn", contact="jochen.stahn@psi.ch", affiliation=None
        ),
        sample=fileio.data_source.Sample.empty(),
        experiment=fileio.data_source.Experiment(
            title="commissioning",
            instrument="AMOR",
            start_date=datetime(2020, 11, 25, 16, 3, 10),  # noqa: DTZ001
            probe="neutron",
            facility="SINQ",
        ),
        measurement=fileio.data_source.Measurement(
            data_files=[fileio.base.File(file="sample.nxs")],
            # We would need the full pipeline to determine this:
            additional_files=[fileio.File("reference.nxs", comment="supermirror")],
            instrument_settings=None,
        ),
    )
    assert data_source == expected


def test_build_orso_reduction_with_creator():
    creator = fileio.base.Person(
        name="Erika Mustermann", affiliation="ESS", contact="erika.mustermann@ess.eu"
    )
    pipeline = sciline.Pipeline(
        orso.providers, params={orso.OrsoCreator: orso.OrsoCreator(creator)}
    )
    reduction = pipeline.compute(orso.OrsoReduction)
    assert reduction.software.name == "ess.reflectometry"
    assert reduction.software.version == str(reflectometry.__version__)
    assert reduction.creator == creator


def test_build_orso_aggregates_are_not_nan():
    events = sc.DataArray(
        sc.array(dims='x', values=[1, 2, 3, 4]),
        coords={
            'theta': sc.array(dims='x', values=[0, 0.5, 1, float('nan')]),
            'wavelength': sc.array(dims='x', values=[0, 0.5, 1, float('nan')]),
        },
    )
    pipeline = sciline.Pipeline(
        orso.providers, params={ReducibleData[SampleRun]: events}
    )
    instrument = pipeline.compute(orso.OrsoInstrument)
    assert not any(
        isnan(getattr(instrument.incident_angle, attr)) for attr in ('min', 'max')
    )
    assert not any(
        isnan(getattr(instrument.wavelength, attr)) for attr in ('min', 'max')
    )


def test_orso_dataset_excludes_masked_and_nonfinite_rows():
    curve = sc.DataArray(
        sc.array(
            dims=['Q'],
            values=[1.0, 2.0, np.nan, np.inf, 5.0, 6.0],
            variances=[1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
        ),
        coords={'Q': sc.arange('Q', 6.0, unit='1/angstrom')},
        masks={
            'first': sc.array(
                dims=['Q'], values=[True, False, False, False, False, False]
            ),
            'second': sc.array(
                dims=['Q'], values=[False, False, False, False, False, True]
            ),
            'scalar': sc.scalar(False),
        },
    )
    result = _orso_dataset(curve)
    np.testing.assert_allclose(result.data[:, :3], [[1.0, 2.0, 1.0]])

    curve.masks['scalar'] = sc.scalar(True)
    assert _orso_dataset(curve).data.shape == (0, 4)


def test_orso_dataset_excludes_nonfinite_supplied_resolution():
    curve = sc.DataArray(
        sc.array(dims=['Q'], values=[1.0, 2.0, 3.0], variances=[1.0, 1.0, 1.0]),
        coords={
            'Q': sc.arange('Q', 3.0, unit='1/angstrom'),
            'Q_resolution': sc.array(
                dims=['Q'], values=[0.1, np.nan, np.inf], unit='1/angstrom'
            ),
        },
    )
    np.testing.assert_allclose(_orso_dataset(curve).data, [[0.0, 1.0, 1.0, 0.1]])


@pytest.mark.parametrize('with_resolution', [False, True])
def test_orso_dataset_converts_integer_coordinates_to_float(with_resolution):
    curve = sc.DataArray(
        sc.array(dims=['Q'], values=[1.0], variances=[1.0]),
        coords={'Q': sc.array(dims=['Q'], values=[1], unit='1/nm')},
    )
    if with_resolution:
        curve.coords['Q_resolution'] = sc.array(dims=['Q'], values=[2], unit='1/nm')
    np.testing.assert_allclose(
        _orso_dataset(curve).data, [[0.1, 1.0, 1.0, 0.2 if with_resolution else np.nan]]
    )


def test_orso_dataset_does_not_mutate_shared_reduction_metadata():
    curve = sc.DataArray(
        sc.array(dims=['Q'], values=[1.0], variances=[1.0]),
        coords={'Q': sc.array(dims=['Q'], values=[0.1], unit='1/angstrom')},
    )
    reduction = fileio.reduction.Reduction.empty()
    reduction.corrections = ['original']
    corrections = ['first']
    first = _orso_dataset(curve, reduction=reduction, corrections=corrections)
    second = _orso_dataset(curve, reduction=reduction, corrections=['second'])
    corrections.append('later')

    assert reduction.corrections == ['original']
    assert first.info.reduction.corrections == ['first']
    assert second.info.reduction.corrections == ['second']


def _orso_dataset(curve, *, reduction=None, corrections=None):
    return orso.build_orso_iofq_dataset(
        curve,
        data_source=fileio.data_source.DataSource.empty(),
        reduction=fileio.reduction.Reduction.empty()
        if reduction is None
        else reduction,
        corrections=[] if corrections is None else corrections,
    )
