# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import sciline
import scipp as sc
from ess import dream, powder

import ess.dream.data  # noqa: F401
from ess.powder.types import (
    AccumulatedProtonCharge,
    BackgroundRun,
    CalibrationFilename,
    DspacingBins,
    Filename,
    IofDspacing,
    IofDspacingTwoTheta,
    MaskedData,
    NeXusDetectorName,
    NeXusSample,
    NeXusSource,
    NormalizedByProtonCharge,
    SampleRun,
    TofMask,
    TwoThetaBins,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMask,
)


@pytest.fixture()
def providers():
    from ess.dream.io.geant4 import providers as geant4_providers

    return [*powder.providers, *geant4_providers]


@pytest.fixture(params=["mantle", "endcap_backward", "endcap_forward"])
def params(request):
    # Not available in simulated data
    sample = sc.DataGroup(position=sc.vector([0.0, 0.0, 0.0], unit='mm'))
    source = sc.DataGroup(position=sc.vector([-3.478, 0.0, -76550], unit='mm'))
    charge = sc.scalar(1.0, unit='µAh')

    return {
        NeXusDetectorName: request.param,
        Filename[SampleRun]: dream.data.simulated_diamond_sample(),
        Filename[VanadiumRun]: dream.data.simulated_vanadium_sample(),
        Filename[BackgroundRun]: dream.data.simulated_empty_can(),
        CalibrationFilename: None,
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
        DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 201, unit='angstrom'),
        TofMask: lambda x: (x < sc.scalar(0.0, unit='ns'))
        | (x > sc.scalar(86e6, unit='ns')),
        NeXusSample[SampleRun]: sample,
        NeXusSample[VanadiumRun]: sample,
        NeXusSource[SampleRun]: source,
        NeXusSource[VanadiumRun]: source,
        AccumulatedProtonCharge[SampleRun]: charge,
        AccumulatedProtonCharge[VanadiumRun]: charge,
        TwoThetaMask: None,
        WavelengthMask: None,
    }


@pytest.fixture()
def workflow(params):
    wf = dream.DreamGeant4Workflow()
    for key, value in params.items():
        wf[key] = value
    return wf


def test_pipeline_can_compute_dspacing_result(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {
        'dspacing': len(workflow.compute(DspacingBins)) - 1,
    }
    assert sc.identical(result.coords['dspacing'], workflow.compute(DspacingBins))


def test_workflow_is_deterministic(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = workflow.get(IofDspacing, scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_can_compute_intermediate_results(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(NormalizedByProtonCharge[SampleRun])
    assert set(result.dims) == {'segment', 'wire', 'counter', 'strip', 'module'}


def test_pipeline_group_by_two_theta(workflow):
    workflow[TwoThetaBins] = sc.linspace(
        dim='two_theta', unit='rad', start=0.8, stop=2.4, num=17
    )
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacingTwoTheta)
    assert result.sizes == {
        'two_theta': 16,
        'dspacing': len(workflow.compute(DspacingBins)) - 1,
    }
    assert sc.identical(result.coords['dspacing'], workflow.compute(DspacingBins))
    assert sc.allclose(result.coords['two_theta'], workflow.compute(TwoThetaBins))


def test_pipeline_wavelength_masking(workflow):
    wmin = sc.scalar(0.18, unit="angstrom")
    wmax = sc.scalar(0.21, unit="angstrom")
    workflow[WavelengthMask] = lambda x: (x > wmin) & (x < wmax)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    masked_sample = workflow.compute(MaskedData[SampleRun])
    assert 'wavelength' in masked_sample.bins.masks
    sum_in_masked_region = (
        masked_sample.bin(wavelength=sc.concat([wmin, wmax], dim='wavelength'))
        .sum()
        .data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


def test_pipeline_two_theta_masking(workflow):
    tmin = sc.scalar(1.0, unit="rad")
    tmax = sc.scalar(1.2, unit="rad")
    workflow[TwoThetaMask] = lambda x: (x > tmin) & (x < tmax)
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    masked_sample = workflow.compute(MaskedData[SampleRun])
    assert 'two_theta' in masked_sample.masks
    sum_in_masked_region = (
        masked_sample.bin(two_theta=sc.concat([tmin, tmax], dim='two_theta')).sum().data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


def test_use_workflow_helper(workflow):
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {
        'dspacing': len(workflow.compute(DspacingBins)) - 1,
    }
    assert sc.identical(result.coords['dspacing'], workflow.compute(DspacingBins))
