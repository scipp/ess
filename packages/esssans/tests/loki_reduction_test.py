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
    CorrectForGravity,
    EmptyBeamRun,
    Filename,
    FinalDims,
    Incident,
    NeXusMonitorName,
    QBins,
    RunID,
    SampleRun,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
)


def make_params(
    sample_runs: Optional[List[int]] = None, background_runs: Optional[List[int]] = None
) -> dict:
    params = {}
    suffix = '-2022-02-28_2215.nxs'

    if sample_runs is None:
        sample_runs = [60339]
    sample_filenames = [f'{i}{suffix}' for i in sample_runs]
    sample_runs_table = sciline.ParamTable(
        RunID[SampleRun], {Filename[SampleRun]: sample_filenames}, index=sample_runs
    )

    if background_runs is None:
        background_runs = [60393]
    background_filenames = [f'{i}{suffix}' for i in background_runs]
    background_runs_table = sciline.ParamTable(
        RunID[BackgroundRun],
        {Filename[BackgroundRun]: background_filenames},
        index=background_runs,
    )

    params[Filename[TransmissionRun[SampleRun]]] = f'60394{suffix}'
    params[Filename[TransmissionRun[BackgroundRun]]] = f'60392{suffix}'
    params[Filename[EmptyBeamRun]] = f'60392{suffix}'

    params[NeXusMonitorName[Incident]] = 'monitor_1'
    params[NeXusMonitorName[Transmission]] = 'monitor_2'

    # Wavelength binning parameters
    band = sc.linspace('wavelength', 1.0, 13.0, num=2, unit='angstrom')
    params[WavelengthBands] = band
    params[WavelengthBins] = sc.linspace(
        'wavelength', start=band[0], stop=band[-1], num=141
    )

    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
    )

    # The final result should have dims of Q only
    params[FinalDims] = ['Q']

    return params, [sample_runs_table, background_runs_table]


def loki_providers() -> List[Callable]:
    return list(sans.providers + sans.loki.providers)


def test_can_create_pipeline():
    params, param_tables = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in param_tables:
        pipeline.set_param_table(table)
    pipeline.get(BackgroundSubtractedIofQ)


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_IofQ(uncertainties):
    params, param_tables = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in param_tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQ_in_wavelength_slices():
    params, param_tables = make_params()
    band = np.linspace(1.0, 13.0, num=11)
    params[WavelengthBands] = sc.array(
        dims=['band', 'wavelength'],
        values=np.vstack([band[:-1], band[1:]]).T,
        unit='angstrom',
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in param_tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Q')
    assert result.sizes['band'] == 10


def test_pipeline_can_compute_IofQ_in_layers():
    params, param_tables = make_params()
    params[FinalDims] = ['layer', 'Q']
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in param_tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('layer', 'Q')
    assert result.sizes['layer'] == 4


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs():
    params, param_tables = make_params(
        sample_runs=[60250, 60339], background_runs=[60248, 60393]
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in param_tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
