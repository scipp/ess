# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.dream import tools as tls


def make_fake_dream_data(nevents):
    params = {'module': 14, 'segment': 6, 'counter': 2, 'wire': 32, 'strip': 256}
    coords = {
        key: sc.array(
            dims=['row'], values=np.random.choice(np.arange(1, val + 1), size=nevents)
        )
        for key, val in params.items()
    }
    return sc.DataArray(data=sc.ones(sizes={'row': nevents}), coords=coords)


def test_to_logical_dims():
    da = tls.to_logical_dims(make_fake_dream_data(1000))
    assert set(da.dims) == {'module', 'segment', 'counter', 'wire', 'strip'}
    assert da.bins is not None


COMBINATIONS = [
    ('wire', 'strip'),
    ('module', 'segment'),
    ('module', 'wire'),
    ('module', 'strip'),
    ('segment', 'wire'),
    ('segment', 'strip'),
]


@pytest.mark.parametrize("dims", COMBINATIONS)
def test_wire_vs_strip_from_table(dims):
    func = getattr(tls, dims[0] + '_vs_' + dims[1])
    da = func(make_fake_dream_data(1000))
    assert set(da.dims) == set(dims)


@pytest.mark.parametrize("dims", COMBINATIONS)
def test_wire_vs_strip_from_logical(dims):
    func = getattr(tls, dims[0] + '_vs_' + dims[1])
    da = func(tls.to_logical_dims(make_fake_dream_data(1000)))
    assert set(da.dims) == set(dims)


@pytest.mark.parametrize("dims", COMBINATIONS)
def test_wire_vs_strip_first_dim_exists(dims):
    func = getattr(tls, dims[0] + '_vs_' + dims[1])
    da = func(make_fake_dream_data(1000).group(dims[0]))
    assert set(da.dims) == set(dims)


@pytest.mark.parametrize("dims", COMBINATIONS)
def test_wire_vs_strip_second_dim_exists(dims):
    func = getattr(tls, dims[0] + '_vs_' + dims[1])
    da = func(make_fake_dream_data(1000).group(dims[1]))
    assert set(da.dims) == set(dims)
