# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List

import numpy as np
import sciline
import scipp as sc
from loki_common import make_params

import esssans as sans
from esssans.types import DimsToKeep, QBins
from esssans.direct_beam import get_I0


def loki_providers() -> List[Callable]:
    return list(sans.providers + sans.loki.providers)


def test_can_compute_direct_beam_for_all_pixels():
    n_wavelength_bands = 10
    params = make_params(n_wavelength_bands=n_wavelength_bands)
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = get_I0(filename='PolyGauss_I0-50_Rg-60.txt', q=sc.midpoints(params[QBins])[0])

    results = sans.direct_beam(pipeline=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('Q',)
    assert iofq_bands.dims == ('band', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands


def test_can_compute_direct_beam_per_layer():
    n_wavelength_bands = 10
    params = make_params(n_wavelength_bands=n_wavelength_bands)
    params[DimsToKeep] = ['layer']
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = get_I0(filename='PolyGauss_I0-50_Rg-60.txt', q=sc.midpoints(params[QBins])[0])

    results = sans.direct_beam(pipeline=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('layer', 'Q')
    assert iofq_bands.dims == ('band', 'layer', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert iofq_bands.sizes['layer'] == 4
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands
    assert direct_beam_function.sizes['layer'] == 4
