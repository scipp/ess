# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List, Optional

import numpy as np
import pytest
import sciline
import scipp as sc

import esssans as sans
from esssans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    CorrectForGravity,
    DataWithLogicalDims,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    FinalDims,
    Incident,
    IofQ,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    SampleRun,
    SampleRunID,
    SolidAngle,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMask,
)


def make_params() -> dict:
    params = {}

    sample_runs = [60339]
    sample_filenames = [f'{i}-2022-02-28_2215.nxs' for i in sample_runs]
    param_table = sciline.ParamTable(
        SampleRunID, {Filename[SampleRun]: sample_filenames}, index=sample_runs
    )

    params[Filename[TransmissionRun[SampleRun]]] = '60394-2022-02-28_2215.nxs'
    params[Filename[EmptyBeamRun]] = '60392-2022-02-28_2215.nxs'

    params[NeXusMonitorName[Incident]] = 'monitor_1'
    params[NeXusMonitorName[Transmission]] = 'monitor_2'

    # Wavelength binning parameters
    band = sc.linspace('wavelength', 1.0, 13.0, num=2, unit='angstrom')
    params[WavelengthBands] = band
    params[WavelengthBins] = sc.linspace(
        'wavelength', start=band[0], stop=band[-1], num=141
    )

    # wavelength_min = sc.scalar(1.0, unit='angstrom')
    # wavelength_max = sc.scalar(13.0, unit='angstrom')
    # n_wavelength_bins = 200
    # # Wavelength bands parameters
    # n_wavelength_bands = 50
    # sampling_width = sc.scalar(0.25, unit='angstrom')

    # # Wavelength binning and bands
    # wavelength_bins, wavelength_bands = sans.directbeam.make_wavelength_bins_and_bands(
    #     wavelength_min=wavelength_min,
    #     wavelength_max=wavelength_max,
    #     n_wavelength_bins=n_wavelength_bins,
    #     n_wavelength_bands=n_wavelength_bands,
    #     sampling_width=sampling_width,
    # )

    # params[WavelengthBins] = wavelength_bins
    # params[WavelengthBands] = wavelength_bands

    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
    )

    # The final result should have dims of Q only
    params[FinalDims] = ['Q']

    return params, param_table


def loki_providers():
    return list(sans.providers + sans.loki.providers)


def test_can_create_pipeline():
    params, param_table = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_table(param_table)
    pipeline.get(IofQ[SampleRun])


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_IofQ(uncertainties):
    params, param_table = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_table(param_table)
    result = pipeline.compute(IofQ[SampleRun])
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQ_in_wavelength_slices():
    params, param_table = make_params()
    band = np.linspace(1.0, 13.0, num=11)
    params[WavelengthBands] = sc.array(
        dims=['band', 'wavelength'],
        values=np.vstack([band[:-1], band[1:]]).T,
        unit='angstrom',
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_table(param_table)
    result = pipeline.compute(IofQ[SampleRun])
    assert result.dims == ('band', 'Q')
    assert result.sizes['band'] == 10


def test_pipeline_can_compute_IofQ_in_layers():
    params, param_table = make_params()
    params[FinalDims] = ['layer', 'Q']
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    pipeline.set_param_table(param_table)
    result = pipeline.compute(IofQ[SampleRun])
    assert result.dims == ('layer', 'Q')
    assert result.sizes['layer'] == 4
