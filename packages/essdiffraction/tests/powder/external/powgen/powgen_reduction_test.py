# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import sciline
import scipp as sc
from ess.powder.types import (
    CalibrationFilename,
    DspacingBins,
    DspacingHistogram,
    Filename,
    NeXusDetectorName,
    NormalizedByProtonCharge,
    SampleRun,
    TofMask,
    TwoThetaBins,
    UncertaintyBroadcastMode,
    VanadiumRun,
)


@pytest.fixture()
def providers():
    from ess import powder
    from ess.powder.external import powgen

    return [*powder.providers, *powgen.providers]


@pytest.fixture()
def params():
    from ess.powder.external import powgen

    return {
        NeXusDetectorName: "powgen_detector",
        Filename[SampleRun]: powgen.data.powgen_tutorial_sample_file(),
        Filename[VanadiumRun]: powgen.data.powgen_tutorial_vanadium_file(),
        CalibrationFilename: powgen.data.powgen_tutorial_calibration_file(),
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
        DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 200, unit='angstrom'),
        TofMask: lambda x: (x < sc.scalar(0.0, unit="us"))
        | (x > sc.scalar(16666.67, unit="us")),
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
    assert set(result.dims) == {'bank', 'column', 'row'}


def test_pipeline_group_by_two_theta(providers, params):
    params[TwoThetaBins] = sc.linspace(
        dim='two_theta', unit='deg', start=25.0, stop=90.0, num=16
    ).to(unit='rad')
    pipeline = sciline.Pipeline(providers, params=params)
    result = pipeline.compute(DspacingHistogram)
    assert result.sizes == {
        'two_theta': 15,
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])
    assert sc.allclose(result.coords['two_theta'], params[TwoThetaBins])
