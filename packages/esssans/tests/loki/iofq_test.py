# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import pytest
import sciline
import scipp as sc

import esssans as sans
from esssans.types import (
    BackgroundSubtractedIofQ,
    BeamCenter,
    DimsToKeep,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_params  # noqa: E402


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
    params[WavelengthBands] = sc.linspace(
        'wavelength',
        params[WavelengthBins].min(),
        params[WavelengthBins].max(),
        11,
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


def test_beam_center_from_center_of_mass_is_close_to_verified_result():
    params = make_params()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    center = pipeline.compute(BeamCenter)
    reference = sc.vector([-0.0309889, -0.0168854, 0], unit='m')
    assert sc.allclose(center, reference)
