# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F403, F405

import numpy as np
import pytest
import sciline
import scipp as sc
from orsopy import fileio

from essreflectometry import orso
from essreflectometry.amor import default_parameters
from essreflectometry.amor import orso as amor_orso
from essreflectometry.amor import providers
from essreflectometry.types import *


@pytest.fixture()
def amor_pipeline() -> sciline.Pipeline:
    params = {
        **default_parameters,
        QBins: sc.geomspace(
            dim='Q', start=0.008, stop=0.075, num=200, unit='1/angstrom'
        ),
        SampleRotation[Sample]: sc.scalar(0.7989, unit='deg'),
        Filename[Sample]: "sample.nxs",
        SampleRotation[Reference]: sc.scalar(0.8389, unit='deg'),
        Filename[Reference]: "reference.nxs",
        WavelengthEdges: sc.array(
            dims=['wavelength'], values=[2.4, 16.0], unit='angstrom'
        ),
        orso.OrsoCreator: orso.OrsoCreator(
            fileio.base.Person(
                name='Max Mustermann',
                affiliation='European Spallation Source ERIC',
                contact='max.mustermann@ess.eu',
            )
        ),
    }
    return sciline.Pipeline(
        (*providers, *orso.providers, *amor_orso.providers), params=params
    )


def test_run_pipeline(amor_pipeline: sciline.Pipeline):
    res = amor_pipeline.compute(orso.OrsoIofQDataset)
    assert res.info.data_source.experiment.instrument == 'AMOR'
    assert res.info.reduction.software.name == 'ess.reflectometry'
    assert res.data.ndim == 2
    assert res.data.shape[1] == 4
    assert np.all(res.data[:, 1] > 0)


def test_find_corrections(amor_pipeline: sciline.Pipeline):
    graph = amor_pipeline.get(orso.OrsoIofQDataset)
    # In topological order
    assert orso.find_corrections(graph) == [
        'supermirror calibration',
        'chopper ToF correction',
        'footprint correction',
        'total counts',
    ]
