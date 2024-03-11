# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from pathlib import Path

import sciline
import scipp as sc
from scipp.scipy.interpolate import interp1d

from ess import sans
from ess.loki.data import get_path
from ess.sans.types import DimsToKeep, QBins, WavelengthBands, WavelengthBins

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import loki_providers, make_params  # noqa: E402


def _get_I0(qbins: sc.Variable) -> sc.Variable:
    Iq_theory = sc.io.load_hdf5(get_path('PolyGauss_I0-50_Rg-60.h5'))
    f = interp1d(Iq_theory, 'Q')
    return f(sc.midpoints(qbins)).data[0]


def test_can_compute_direct_beam_for_all_pixels():
    n_wavelength_bands = 10
    params = make_params()
    params[WavelengthBands] = sc.linspace(
        'wavelength',
        params[WavelengthBins].min(),
        params[WavelengthBins].max(),
        n_wavelength_bands + 1,
    )
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = _get_I0(qbins=params[QBins])

    results = sans.direct_beam(pipeline=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('Q',)
    assert iofq_bands.dims == ('band', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands


def test_can_compute_direct_beam_with_overlapping_wavelength_bands():
    n_wavelength_bands = 10
    params = make_params()
    # Bands have double the width
    edges = sc.linspace(
        'band',
        params[WavelengthBins].min(),
        params[WavelengthBins].max(),
        n_wavelength_bands + 2,
    )
    params[WavelengthBands] = sc.concat(
        [edges[:-2], edges[2::]], dim='wavelength'
    ).transpose()

    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = _get_I0(qbins=params[QBins])

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
    params = make_params()
    params[WavelengthBands] = sc.linspace(
        'wavelength',
        params[WavelengthBins].min(),
        params[WavelengthBins].max(),
        n_wavelength_bands + 1,
    )
    params[DimsToKeep] = ['layer']
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = _get_I0(qbins=params[QBins])

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


def test_can_compute_direct_beam_per_layer_and_straw():
    n_wavelength_bands = 10
    params = make_params()
    params[WavelengthBands] = sc.linspace(
        'wavelength',
        params[WavelengthBins].min(),
        params[WavelengthBins].max(),
        n_wavelength_bands + 1,
    )
    params[DimsToKeep] = ['layer', 'straw']
    providers = loki_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    I0 = _get_I0(qbins=params[QBins])

    results = sans.direct_beam(pipeline=pipeline, I0=I0, niter=4)
    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    assert iofq_full.dims == ('layer', 'straw', 'Q')
    assert iofq_bands.dims == ('band', 'layer', 'straw', 'Q')
    assert iofq_bands.sizes['band'] == n_wavelength_bands
    assert iofq_bands.sizes['layer'] == 4
    assert iofq_bands.sizes['straw'] == 7
    assert direct_beam_function.sizes['wavelength'] == n_wavelength_bands
    assert direct_beam_function.sizes['layer'] == 4
    assert direct_beam_function.sizes['straw'] == 7
