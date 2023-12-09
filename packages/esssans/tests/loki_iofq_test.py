# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import numpy as np
import pytest
import sciline
import scipp as sc

import esssans as sans
from esssans.types import (
    BackgroundSubtractedIofQ,
    FinalDims,
    UncertaintyBroadcastMode,
    WavelengthBands,
)

from loki_common import make_param_tables, make_params


def loki_providers() -> List[Callable]:
    return list(sans.providers + sans.loki.providers)


def test_can_create_pipeline():
    params = make_params()
    tables = make_param_tables()
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in tables:
        pipeline.set_param_table(table)
    pipeline.get(BackgroundSubtractedIofQ)


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_IofQ(uncertainties):
    params = make_params()
    tables = make_param_tables()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_IofQ_in_wavelength_slices():
    params = make_params()
    tables = make_param_tables()
    band = np.linspace(1.0, 13.0, num=11)
    params[WavelengthBands] = sc.array(
        dims=['band', 'wavelength'],
        values=np.vstack([band[:-1], band[1:]]).T,
        unit='angstrom',
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Q')
    assert result.sizes['band'] == 10


def test_pipeline_can_compute_IofQ_in_layers():
    params = make_params()
    tables = make_param_tables()
    params[FinalDims] = ['layer', 'Q']
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('layer', 'Q')
    assert result.sizes['layer'] == 4


def test_pipeline_can_compute_IofQ_merging_events_from_multiple_runs():
    params = make_params()
    tables = make_param_tables(
        sample_runs=[60250, 60339], background_runs=[60248, 60393]
    )
    pipeline = sciline.Pipeline(loki_providers(), params=params)
    for table in tables:
        pipeline.set_param_table(table)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
