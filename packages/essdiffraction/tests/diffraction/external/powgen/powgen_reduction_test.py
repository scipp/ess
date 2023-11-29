# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import sciline
import scipp as sc

from ess.diffraction.types import (
    CalibrationFilename,
    DspacingBins,
    DspacingHistogram,
    Filename,
    NormalizedByProtonCharge,
    SampleRun,
    TwoThetaBins,
    ValidTofRange,
    VanadiumRun,
)


@pytest.fixture()
def providers():
    from ess import diffraction
    from ess.diffraction import powder
    from ess.diffraction.external import powgen

    return [*diffraction.providers, *powder.providers, *powgen.providers]


@pytest.fixture()
def params():
    return {
        Filename[SampleRun]: 'PG3_4844_event.zip',
        Filename[VanadiumRun]: 'PG3_4866_event.zip',
        CalibrationFilename: 'PG3_FERNS_d4832_2011_08_24.zip',
        ValidTofRange: sc.array(dims=['tof'], values=[0.0, 16666.67], unit='us'),
        DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 200, unit='angstrom'),
    }


def test_can_create_pipeline(providers, params):
    sciline.Pipeline(providers, params=params)


def test_pipeline_can_compute_dspacing_histogram(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    result = pipeline.compute(DspacingHistogram)
    assert result.sizes == {
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_workflow_is_deterministic(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = pipeline.get(DspacingHistogram, scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_can_compute_intermediate_results(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    result = pipeline.compute(NormalizedByProtonCharge[SampleRun])
    assert set(result.dims) == {'spectrum', 'tof'}


def test_pipeline_group_by_two_theta(providers, params):
    from ess.diffraction.grouping import group_by_two_theta, merge_all_pixels

    providers.remove(merge_all_pixels)
    providers.append(group_by_two_theta)
    params[TwoThetaBins] = sc.linspace(
        dim='two_theta', unit='deg', start=25.0, stop=90.0, num=16
    )
    pipeline = sciline.Pipeline(providers, params=params)
    result = pipeline.compute(DspacingHistogram)
    assert result.sizes == {
        'two_theta': 15,
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])
    assert sc.allclose(result.coords['two_theta'].to(unit='deg'), params[TwoThetaBins])
