# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F403, F405

import numpy as np
import pytest
import sciline
import scipp as sc

from ess.estia import EstiaWorkflow
from ess.estia.data import estia_mcstas_reference_run, estia_mcstas_sample_run
from ess.estia.load import load_mcstas_events
from ess.reflectometry.types import (
    BeamDivergenceLimits,
    Filename,
    ProtonCurrent,
    QBins,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRun,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


@pytest.fixture
def estia_mcstas_pipeline() -> sciline.Pipeline:
    wf = EstiaWorkflow()
    wf.insert(load_mcstas_events)
    wf[Filename[ReferenceRun]] = estia_mcstas_reference_run()

    wf[YIndexLimits] = sc.scalar(35), sc.scalar(64)
    wf[ZIndexLimits] = sc.scalar(0), sc.scalar(14 * 32)
    wf[BeamDivergenceLimits] = sc.scalar(-1.0, unit='deg'), sc.scalar(1.0, unit='deg')
    wf[WavelengthBins] = sc.geomspace('wavelength', 3.5, 12, 2001, unit='angstrom')
    wf[ProtonCurrent[SampleRun]] = sc.DataArray(
        sc.array(dims=('time',), values=[]),
        coords={'time': sc.array(dims=('time',), values=[], unit='s')},
    )
    wf[ProtonCurrent[ReferenceRun]] = sc.DataArray(
        sc.array(dims=('time',), values=[]),
        coords={'time': sc.array(dims=('time',), values=[], unit='s')},
    )

    return wf


def test_mcstas_compute_reducible_data(estia_mcstas_pipeline: sciline.Pipeline):
    estia_mcstas_pipeline[Filename[SampleRun]] = estia_mcstas_sample_run(11)
    da = estia_mcstas_pipeline.compute(ReducibleData[SampleRun])
    assert da.dims == ('blade', 'wire', 'stripe')
    assert da.shape == (14, 32, 64)
    assert 'position' in da.coords
    assert 'sample_rotation' in da.coords
    assert 'detector_rotation' in da.coords
    assert 'theta' in da.coords
    assert 'wavelength' in da.bins.coords
    assert 'Q' in da.bins.coords


def test_can_compute_reflectivity_curve(estia_mcstas_pipeline: sciline.Pipeline):
    estia_mcstas_pipeline[Filename[SampleRun]] = estia_mcstas_sample_run(11)
    estia_mcstas_pipeline[QBins] = sc.geomspace('Q', 0.005, 0.1, 200, unit='1/angstrom')
    r = estia_mcstas_pipeline.compute(ReflectivityOverQ)
    assert "Q" in r.coords
    assert "Q_resolution" in r.coords
    r = r.hist()
    min_q = sc.where(
        r.data > sc.scalar(0),
        sc.midpoints(r.coords['Q']),
        sc.scalar(np.nan, unit='1/angstrom'),
    ).nanmin()
    max_q = sc.where(
        r.data > sc.scalar(0),
        sc.midpoints(r.coords['Q']),
        sc.scalar(np.nan, unit='1/angstrom'),
    ).nanmax()

    assert max_q > sc.scalar(0.075, unit='1/angstrom')
    assert min_q < sc.scalar(0.007, unit='1/angstrom')
