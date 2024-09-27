# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from datetime import datetime

import sciline
from orsopy import fileio

from ess import amor, reflectometry
from ess.amor import data  # noqa: F401
from ess.reflectometry import orso
from ess.reflectometry.types import Filename, ReferenceRun, SampleRun


def test_build_orso_data_source():
    pipeline = sciline.Pipeline(
        (*amor.load.providers, *orso.providers),
        params={
            Filename[SampleRun]: amor.data.amor_old_sample_run(),
            Filename[ReferenceRun]: amor.data.amor_old_reference_run(),
        },
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
