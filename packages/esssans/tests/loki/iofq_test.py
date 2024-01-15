# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import numpy as np
import pytest
import sciline
import scipp as sc
from loki_common import make_params

import esssans as sans
from esssans.types import (
    BackgroundSubtractedIofQ,
    DimsToKeep,
    UncertaintyBroadcastMode,
    WavelengthBands,
)


def loki_providers() -> List[Callable]:
    return list(sans.providers + sans.loki.providers)


def test_can_create_pipeline():
    pipeline = sciline.Pipeline(loki_providers(), params=make_params())
    pipeline.get(BackgroundSubtractedIofQ)


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_IofQ(uncertainties):
    params = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQ_in_wavelength_slices():
    params = make_params()
    band = np.linspace(1.0, 13.0, num=11)
    params[WavelengthBands] = sc.array(
        dims=['band', 'wavelength'],
        values=np.vstack([band[:-1], band[1:]]).T,
        unit='angstrom',
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Q')
    assert result.sizes['band'] == 10


def test_pipeline_can_compute_IofQ_in_layers():
    params = make_params()
    params[DimsToKeep] = ['layer']
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('layer', 'Q')
    assert result.sizes['layer'] == 4


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs():
    params = make_params(
        sample_runs=['60250-2022-02-28_2215.nxs', '60339-2022-02-28_2215.nxs'],
        background_runs=['60248-2022-02-28_2215.nxs', '60393-2022-02-28_2215.nxs'],
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
