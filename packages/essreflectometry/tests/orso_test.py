# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from datetime import datetime

import sciline
from orsopy import fileio

import essreflectometry
from essreflectometry import orso
from essreflectometry.amor.load import providers as amor_load_providers
from essreflectometry.types import Filename, Sample


def test_build_orso_data_source():
    pipeline = sciline.Pipeline(
        (
            *amor_load_providers,
            *orso.providers,
        ),
        params={Filename[Sample]: 'sample.nxs'},
    )
    data_source = pipeline.compute(orso.OrsoDataSource)
    expected = fileio.data_source.DataSource(
        owner=fileio.base.Person(
            name='J. Stahn', contact='jochen.stahn@psi.ch', affiliation=None
        ),
        sample=fileio.data_source.Sample.empty(),
        experiment=fileio.data_source.Experiment(
            title='commissioning',
            instrument='AMOR',
            start_date=datetime(2020, 11, 25, 16, 3, 10),
            probe='neutron',
            facility='SINQ',
        ),
        measurement=fileio.data_source.Measurement(
            data_files=[fileio.base.File(file='sample.nxs')],
            # We would need the full pipeline to determine this:
            additional_files=[],
            instrument_settings=None,
        ),
    )
    assert data_source == expected


def test_build_orso_reduction_without_creator():
    pipeline = sciline.Pipeline(orso.providers)
    reduction = pipeline.compute(orso.OrsoReduction)
    assert reduction.software.name == 'ess.reflectometry'
    assert reduction.software.version == str(essreflectometry.__version__)
    assert reduction.creator is None


def test_build_orso_reduction_with_creator():
    creator = fileio.base.Person(
        name='Erika Mustermann', affiliation='ESS', contact='erika.mustermann@ess.eu'
    )
    pipeline = sciline.Pipeline(
        orso.providers, params={orso.OrsoCreator: orso.OrsoCreator(creator)}
    )
    reduction = pipeline.compute(orso.OrsoReduction)
    assert reduction.software.name == 'ess.reflectometry'
    assert reduction.software.version == str(essreflectometry.__version__)
    assert reduction.creator == creator
