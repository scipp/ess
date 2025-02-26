# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import sciline
import scipp as sc
from scippneutron._utils import elem_unit

import ess.snspowder.powgen.data  # noqa: F401
from ess import powder
from ess.powder.types import (
    CalibrationFilename,
    DetectorBankSizes,
    DspacingBins,
    Filename,
    IofDspacing,
    IofDspacingTwoTheta,
    MaskedData,
    NeXusDetectorName,
    NormalizedRunData,
    SampleRun,
    TofMask,
    TwoThetaBins,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMask,
)
from ess.snspowder import powgen


@pytest.fixture
def providers():
    from ess import powder

    return [*powder.providers, *powgen.providers, *powgen.data.providers]


@pytest.fixture
def params():
    return {
        NeXusDetectorName: "powgen_detector",
        Filename[SampleRun]: powgen.data.powgen_tutorial_sample_file(small=True),
        Filename[VanadiumRun]: powgen.data.powgen_tutorial_vanadium_file(small=True),
        CalibrationFilename: powgen.data.powgen_tutorial_calibration_file(small=True),
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
        DspacingBins: sc.linspace('dspacing', 0.0, 2.3434, 200, unit='angstrom'),
        TofMask: lambda x: (x < sc.scalar(0.0, unit="us").to(unit=elem_unit(x)))
        | (x > sc.scalar(16666.67, unit="us").to(unit=elem_unit(x))),
        TwoThetaMask: None,
        WavelengthMask: None,
        # Use bank sizes for small files
        DetectorBankSizes: {"bank": 23, "column": 7, "row": 7},
    }


def test_can_create_pipeline(providers, params):
    sciline.Pipeline(providers, params=params)


def test_pipeline_can_compute_dspacing_result(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    result = pipeline.compute(IofDspacing)
    assert result.sizes == {
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_pipeline_can_compute_dspacing_result_without_calibration(providers, params):
    params[CalibrationFilename] = None
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    result = pipeline.compute(IofDspacing)
    assert result.sizes == {
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])


def test_pipeline_compare_with_and_without_calibration(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    result_w_cal = pipeline.compute(IofDspacing)

    params[CalibrationFilename] = None
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    result_wo_cal = pipeline.compute(IofDspacing)

    assert sc.identical(
        result_w_cal.coords['dspacing'], result_wo_cal.coords['dspacing']
    )
    assert not sc.allclose(result_w_cal.hist().data, result_wo_cal.hist().data)


def test_workflow_is_deterministic(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = pipeline.get(IofDspacing, scheduler=scheduler)
    reference = graph.compute().hist().data
    result = graph.compute().hist().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_can_compute_intermediate_results(providers, params):
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    result = pipeline.compute(NormalizedRunData[SampleRun])
    assert set(result.dims) == {'bank', 'column', 'row'}


def test_pipeline_group_by_two_theta(providers, params):
    params[TwoThetaBins] = sc.linspace(
        dim='two_theta', unit='deg', start=25.0, stop=90.0, num=16
    ).to(unit='rad')
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    result = pipeline.compute(IofDspacingTwoTheta)
    assert result.sizes == {
        'two_theta': 15,
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])
    assert sc.allclose(result.coords['two_theta'], params[TwoThetaBins])


def test_pipeline_wavelength_masking(providers, params):
    wmin = sc.scalar(0.18, unit="angstrom")
    wmax = sc.scalar(0.21, unit="angstrom")
    params[WavelengthMask] = lambda x: (x > wmin) & (x < wmax)
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    masked_sample = pipeline.compute(MaskedData[SampleRun])
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


def test_pipeline_two_theta_masking(providers, params):
    tmin = sc.scalar(0.8, unit="rad")
    tmax = sc.scalar(1.0, unit="rad")
    params[TwoThetaMask] = lambda x: (x > tmin) & (x < tmax)
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline = powder.with_pixel_mask_filenames(pipeline, [])
    masked_sample = pipeline.compute(MaskedData[SampleRun])
    assert 'two_theta' in masked_sample.masks
    sum_in_masked_region = (
        masked_sample.flatten(to='pixel')
        .hist(two_theta=sc.concat([tmin, tmax], dim='two_theta'))
        .sum()
        .data
    )
    assert sc.allclose(
        sum_in_masked_region,
        sc.scalar(0.0, unit=sum_in_masked_region.unit),
    )


def test_use_workflow_helper(params):
    workflow = powgen.PowgenWorkflow()
    for key, value in params.items():
        workflow[key] = value
    workflow = powder.with_pixel_mask_filenames(workflow, [])
    result = workflow.compute(IofDspacing)
    assert result.sizes == {
        'dspacing': len(params[DspacingBins]) - 1,
    }
    assert sc.identical(result.coords['dspacing'], params[DspacingBins])
