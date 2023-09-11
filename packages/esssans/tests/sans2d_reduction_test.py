# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc

import esssans as sans
from esssans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    CorrectForGravity,
    DirectBeamFilename,
    DirectRun,
    Filename,
    Incident,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    SampleRun,
    SolidAngle,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMask,
)


def make_params() -> dict:
    params = {}
    params[NeXusMonitorName[Incident]] = 'monitor2'
    params[NeXusMonitorName[Transmission]] = 'monitor4'
    band = sc.linspace('wavelength', 2.0, 16.0, num=2, unit='angstrom')
    params[WavelengthBands] = band
    params[WavelengthBins] = sc.linspace(
        'wavelength', start=band[0], stop=band[-1], num=141
    )
    params[WavelengthMask] = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[True]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[2.21, 2.59], unit='angstrom'
            )
        },
    )

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.6, num=141, unit='1/angstrom'
    )
    params[Filename[BackgroundRun]] = 'SANS2D00063159.hdf5'
    params[Filename[SampleRun]] = 'SANS2D00063114.hdf5'
    params[Filename[DirectRun]] = 'SANS2D00063091.hdf5'
    params[DirectBeamFilename] = 'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.hdf5'
    params[BeamCenter] = sc.vector(value=[0.0945643, -0.082074, 0.0], unit='m')
    params[NonBackgroundWavelengthRange] = sc.array(
        dims=['wavelength'], values=[0.7, 17.1], unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    return params


def sans2d_providers():
    return sans.providers + sans.sans2d.providers


def test_can_create_pipeline():
    sciline.Pipeline(sans2d_providers(), params=make_params())


def test_pipeline_can_compute_background_subtracted_IofQ():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_visualize_background_subtracted_IofQ():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    pipeline.visualize(BackgroundSubtractedIofQ)


def test_pipeline_can_compute_intermediate_results():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    result = pipeline.compute(SolidAngle[SampleRun])
    assert result.dims == ('spectrum',)
